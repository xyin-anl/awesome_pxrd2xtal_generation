# PXRD inference script for PXRDnet (https://github.com/gabeguo/cdvae_xrd)
# Checkpoints: mp_20_sinc10 / mp_20_sinc100 from https://huggingface.co/therealgabeguo/cdvae_xrd_sinc10
# 1. Run setup.sh once (environment, upstream at the pinned commit, checkpoints)
# 2. Run: python run.py --pattern my_scan.xy --wavelength CuKa --composition KCaCO3F --z 1 --out results/
# Follows upstream's experimental pipeline: process_real_xrds/read_real_xrd.py:create_data puts
# the measured intensities on the model's 4096-point Q grid (Cu Ka, 0-180 deg); upstream's
# CrystDataset applies its sinc filter and subsampling; scripts/conditional_generation.py
# optimizes 100 latent codes against that target, decodes them with the known composition, and
# keeps the candidates whose simulated patterns fit best.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.cells import atom_list, primitive_z_candidates  # noqa: E402
from inference._common.pxrd_io import load_pattern  # noqa: E402

CHECKPOINTS = {"sinc100": "epoch=954-step=304645.ckpt", "sinc10": "epoch=939-step=299860.ckpt"}
CU_KA_PMG = 1.54184  # pymatgen WAVELENGTHS["CuKa"], hparams data.wavesource
N_PRESUBSAMPLE = 4096
MAX_ATOMS = 20

# scripts/conditional_generation_experimental.sh (the paper's experimental evaluation)
EXPERIMENTAL_ARGS = dict(num_starting_points=100, num_candidates=10, lr=0.1, min_lr=1e-4, l2_penalty=2e-4,
                         num_gradient_steps=5000, num_atom_lambda=0.1, composition_lambda=0.1, n_step_each=100,
                         l1_loss=True)


def experimental_tensor(pattern) -> "torch.Tensor":
    """read_real_xrd.py:create_data for a loaded pattern: bin by Q onto the Cu Ka 0-180 deg grid
    keeping the maximum per bin, then (x - max(min, 0)) / (max - min) clipped at 0."""
    import torch

    q = 4 * np.pi * np.sin(np.radians(pattern.two_theta) / 2) / pattern.wavelength
    q_min = 0.0
    q_max = 4 * np.pi * np.sin(np.radians(90.0)) / CU_KA_PMG
    xrd = torch.zeros(N_PRESUBSAMPLE)
    min_val, max_val = np.inf, -np.inf
    for curr_q, inten in zip(q, pattern.intensity):
        idx = int((curr_q - q_min) / (q_max - q_min) * N_PRESUBSAMPLE)
        if idx >= N_PRESUBSAMPLE:
            break  # upstream stops at the first point beyond the grid
        xrd[idx] = max(xrd[idx], float(inten))
        min_val, max_val = min(min_val, inten), max(max_val, inten)
    min_val = max(min_val, 0)
    return torch.maximum((xrd - min_val) / (max_val - min_val), torch.zeros_like(xrd))


def dummy_cif(symbols: list[str]) -> str:
    """Random cell holding the requested atoms. Only its atom count and types reach the model
    (as the composition constraint); upstream's experimental runs used the reference CIF here."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    rng = random.Random(0)
    lattice = Lattice.from_parameters(*(rng.uniform(4, 8) for _ in range(3)), 90, 90, 90)
    coords = [[rng.random() for _ in range(3)] for _ in symbols]
    return str(CifWriter(Structure(lattice, symbols, coords)))


def main() -> None:
    p = argparse.ArgumentParser(description="PXRDnet PXRD -> crystal structure inference")
    p.add_argument("--pattern", required=True, help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--composition", required=True, help="Reduced formula, e.g. KCaCO3F")
    p.add_argument("--z", type=int, required=True, help="Formula units per conventional cell (the model needs the atom list)")
    p.add_argument("--spacegroup", help="Optional; only used to convert Z to the primitive cell the model works in")
    p.add_argument("--n-samples", type=int, default=10, help="Candidates kept per primitive-Z hypothesis (upstream num_candidates)")
    p.add_argument("--checkpoint", choices=["sinc100", "sinc10"], default="sinc100",
                   help="sinc100 (100 A nanocrystal filter) was used for the paper's experimental results")
    p.add_argument("--num-starting-points", type=int, default=EXPERIMENTAL_ARGS["num_starting_points"])
    p.add_argument("--num-gradient-steps", type=int, default=EXPERIMENTAL_ARGS["num_gradient_steps"],
                   help="Cosine-restart period; total steps are 7x this (upstream)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", default=os.environ.get("PXRDNET_WEIGHTS", os.path.join(HERE, ".weights")))
    p.add_argument("--upstream", default=os.environ.get("PXRDNET_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    if args.z < 1:
        p.error("--z must be >= 1")
    if args.n_samples > args.num_starting_points:
        p.error("--n-samples cannot exceed --num-starting-points")

    # Upstream's experimental records come from CifParser.get_structures(), which returns the
    # primitive cell, so the atom list is that of the primitive cell. A conventional Z is converted
    # with the lattice centering of --spacegroup, or every compatible centering is tried.
    hypotheses = {}
    for zp in primitive_z_candidates(args.z, args.spacegroup):
        symbols = atom_list(args.composition, zp)
        if len(symbols) <= MAX_ATOMS:
            hypotheses[zp] = symbols
    if not hypotheses:
        raise SystemExit(f"Every atom-count hypothesis exceeds the model's {MAX_ATOMS}-atom limit (MP-20)")
    pattern = load_pattern(os.path.abspath(args.pattern), args.wavelength, args.x_unit)

    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)
    model_dir = os.path.abspath(os.path.join(args.weights, f"mp_20_{args.checkpoint}"))
    upstream = os.path.abspath(args.upstream)
    os.environ["PROJECT_ROOT"] = upstream
    os.environ["WANDB_MODE"] = "disabled"
    sys.path[:0] = [upstream, os.path.join(upstream, "scripts")]

    import hydra
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from pymatgen.core import Composition

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import wandb
        from conditional_generation import create_xrd_args, optimize_latent_code, smooth_xrds
        from visualization.visualize_materials import create_materials

        wandb.init(mode="disabled")
        with initialize_config_dir(model_dir, version_base=None):
            cfg = compose(config_name="hparams")
            model = hydra.utils.instantiate(cfg.model, optim=cfg.optim, data=cfg.data, logging=cfg.logging,
                                            _recursive_=False)
            # The stored hyperparameters point the GemNet scale file at the authors' machine; pass the
            # decoder config resolved from hparams.yaml inside the upstream checkout instead.
            model = type(model).load_from_checkpoint(os.path.join(model_dir, CHECKPOINTS[args.checkpoint]),
                                                     decoder=model.hparams.decoder)
            model.lattice_scaler = torch.load(os.path.join(model_dir, "lattice_scaler.pt"))
        model = model.cuda()
        xrd_tensor = experimental_tensor(pattern)
        opt_args = SimpleNamespace(**{**EXPERIMENTAL_ARGS, "num_starting_points": args.num_starting_points,
                                      "num_gradient_steps": args.num_gradient_steps, "num_candidates": args.n_samples,
                                      "start_from_init": None, "wave_source": "CuKa"})
        ld_kwargs = SimpleNamespace(n_step_each=opt_args.n_step_each, step_lr=1e-4, min_sigma=0,
                                    save_traj=False, disable_bar=True)

        results_by_z = {}
        for zp, symbols in hypotheses.items():
            with tempfile.TemporaryDirectory() as tmp:
                # One-row test set in upstream's format (a pickled DataFrame, despite the .csv name).
                pd.DataFrame([{
                    "material_id": "input",
                    "pretty_formula": Composition(args.composition).reduced_formula,
                    "elements": sorted(set(symbols)),
                    "cif": dummy_cif(symbols),
                    "spacegroup.number": 1,
                    "xrd": xrd_tensor,
                }]).to_pickle(os.path.join(tmp, "test.csv"))
                with initialize_config_dir(model_dir, version_base=None):
                    cfg = compose(config_name="hparams", overrides=[f"data.root_path={tmp}"])
                    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False, scaler_path=model_dir)
                    datamodule.setup("test")
                    loader = datamodule.test_dataloader()[0]
            batch = next(iter(loader)).to(model.device)
            target = batch.y.reshape(1, 512)
            z = optimize_latent_code(args=opt_args, model=model, batch=batch, target_noisy_xrd=target)
            crystals = model.langevin_dynamics(z, ld_kwargs,
                                               gt_num_atoms=batch.num_atoms.repeat(opt_args.num_starting_points),
                                               gt_atom_types=batch.atom_types.repeat(opt_args.num_starting_points))
            _, _, raw_xrds, crystal_list = create_materials(
                create_xrd_args(opt_args), crystals["frac_coords"], crystals["num_atoms"], crystals["atom_types"],
                crystals["lengths"], crystals["angles"], create_xrd=True, symprec=0.01)
            # create_materials substitutes an all-zero pattern when simulation fails; such candidates
            # (and any non-finite loss) are excluded instead of being ranked.
            simulated_ok = torch.tensor([bool(np.abs(np.asarray(x)).sum() > 0) for x in raw_xrds])
            smoothed, _ = smooth_xrds(opt_generated_xrds=raw_xrds, data_loader=loader)
            loss = F.l1_loss(smoothed.to(model.device), target.broadcast_to(smoothed.shape[0], 512),
                             reduction="none").mean(dim=-1).cpu()
            valid = simulated_ok & torch.isfinite(loss)
            order = [i for i in torch.argsort(torch.where(valid, loss, torch.tensor(float("inf")))).tolist() if valid[i]]
            results_by_z[zp] = (crystal_list, loss, order[: args.n_samples], int((~valid).sum()))

    with open(os.path.join(out, "upstream.log"), "w", encoding="utf-8") as fout:
        fout.write(log.getvalue())

    from pymatgen.core import Lattice, Structure

    candidates, details = [], []
    for zp, (crystal_list, loss, best, n_invalid) in results_by_z.items():
        for rank, idx in enumerate(best, start=1):
            c = crystal_list[idx]
            path = os.path.join(cand_dir, f"candidate_zp{zp}_{rank:03d}.cif")
            try:
                s = Structure(Lattice.from_parameters(*np.asarray(c["lengths"]).tolist(), *np.asarray(c["angles"]).tolist()),
                              [int(t) for t in c["atom_types"]], c["frac_coords"])
                s.to(filename=path)
                candidates.append(os.path.relpath(path, out))
                details.append({"file": candidates[-1], "primitive_z": zp, "xrd_l1_loss": float(loss[idx])})
            except Exception as exc:
                details.append({"file": None, "primitive_z": zp, "error": str(exc)[:200], "xrd_l1_loss": float(loss[idx])})

    results = {
        "model": "pxrdnet",
        "checkpoint": f"mp_20_{args.checkpoint}",
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "weights")},
        "atom_lists": {str(zp): v for zp, v in hypotheses.items()},
        "invalid_simulations": {str(zp): r[3] for zp, r in results_by_z.items()},
        "runtime_s": round(time.time() - start, 1),
        "candidates": candidates,
        "candidate_details": details,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(candidates)} candidate CIFs to {cand_dir}")


if __name__ == "__main__":
    main()
