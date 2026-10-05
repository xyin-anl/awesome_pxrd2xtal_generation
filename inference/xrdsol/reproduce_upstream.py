# Reproduces XRDSol's MP-20 test result (paper: 82.3% success with known stoichiometry and cell)
# on random test structures, building each input exactly as upstream's training data
# (xrdsol/common/data_utils.py:process_one). The paper generates 25 candidates per structure,
# ranks them by R_cos (cosine similarity of simulated and target patterns), and scores the top one;
# upstream's repository code (compute_metrics.py) instead counts any matching run. Both are reported.
# Usage: python reproduce_upstream.py [--records 100] [--samples 25]  (uses .upstream/data/mp_20/test.csv)

from __future__ import annotations

import argparse
import os
import sys
import types
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=int, default=100)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-xrd", action="store_true",
                   help="Control: give each structure the next structure's pattern")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    upstream = os.path.join(HERE, ".upstream")
    model_dir = os.path.join(HERE, ".weights", "mp20")
    os.environ["PROJECT_ROOT"] = upstream
    sys.path[:0] = [upstream, os.path.join(upstream, "scripts")]
    pkg = types.ModuleType("scripts")
    pkg.__path__ = [os.path.join(upstream, "scripts")]
    sys.modules["scripts"] = pkg

    import hydra
    import pandas as pd
    import torch
    from eval_utils import get_crystals_list, lattices_to_params_shape
    from hydra import compose, initialize_config_dir
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    from torch_geometric.data import Batch, Data
    from xrdsol.common.data_utils import build_crystal, get_voigt_xrd

    torch.manual_seed(args.seed)
    with initialize_config_dir(model_dir):
        cfg = compose(config_name="hparams")
        model = hydra.utils.instantiate(cfg.model, optim=cfg.optim, data=cfg.data, logging=cfg.logging,
                                        _recursive_=False)
        model = model.load_from_checkpoint(os.path.join(model_dir, "epoch=189-step=5129.ckpt"),
                                           hparams_file=os.path.join(model_dir, "hparams.yaml"), strict=False)
    model = model.cuda().eval()

    rows = pd.read_csv(os.path.join(upstream, "data", "mp_20", "test.csv")).sample(args.records, random_state=args.seed)
    grid = np.arange(0, 90, 0.02)
    data, refs = [], []
    for cif in rows["cif"]:
        crystal = build_crystal(cif, niggli=False, primitive=True)  # process_one, hparams: primitive true
        xrd = XRDCalculator().get_pattern(crystal, scaled=False)
        profile = get_voigt_xrd(grid, xrd.x, xrd.y, 0.03, 1.54056)
        data.append(Data(atom_types=torch.LongTensor([s.specie.Z for s in crystal]), num_atoms=len(crystal),
                         num_nodes=len(crystal), lengths=torch.Tensor(crystal.lattice.abc).view(1, -1),
                         angles=torch.Tensor(crystal.lattice.angles).view(1, -1), xrd=torch.Tensor(profile)))
        refs.append(crystal)
    if args.shuffle_xrd:
        profiles = [d.xrd for d in data]
        for i, d in enumerate(data):
            d.xrd = profiles[(i + 1) % len(profiles)]
    batch = Batch.from_data_list(data).cuda()
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    sys.path.insert(0, HERE)
    from run import r_cos

    preds = [[] for _ in refs]
    for _ in range(args.samples):
        outputs, _ = model.sample(batch, step_lr=1e-5)
        lengths, angles = lattices_to_params_shape(outputs["lattices"])
        crystals = get_crystals_list(outputs["frac_coords"].cpu(), outputs["atom_types"].cpu(), lengths.cpu(),
                                     angles.cpu(), outputs["num_atoms"].cpu())
        for i, c in enumerate(crystals):
            preds[i].append(Structure(Lattice.from_parameters(*c["lengths"].tolist(), *c["angles"].tolist()),
                                      [int(t) for t in c["atom_types"]], c["frac_coords"]))
    first = any_match = top_rcos = 0
    for i, ref in enumerate(refs):
        match = [bool(matcher.fit(p, ref)) for p in preds[i]]
        first += match[0]
        any_match += any(match)
        scores = [r_cos(p, data[i].xrd.numpy(), get_voigt_xrd) for p in preds[i]]
        top_rcos += match[int(np.argmax(scores))]
    n = len(refs)
    print(f"records={n} samples={args.samples} one_sample={first / n:.3f} "
          f"top1_by_Rcos={top_rcos / n:.3f} (paper protocol) any_of_samples={any_match / n:.3f} (upstream code)")


if __name__ == "__main__":
    main()
