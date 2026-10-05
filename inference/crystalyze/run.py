# PXRD inference script for Crystalyze (https://github.com/ML-PXRD/Crystalyze)
# Checkpoint: model_folder from the authors' Google Drive data folder (setup.sh downloads it)
# 1. Run setup.sh once (environment, upstream at the pinned commit, checkpoint)
# 2. Run: python run.py --pattern my_scan.xy --wavelength CuKa1 --composition LuOF --z 2 --out results/
# The pattern is prepared like upstream's cdvae/common/inference_utils.py:xy_data_prep (cubic
# interpolation onto 8500 points over 5-90 degrees 2theta, scaled to a maximum of 1), after
# converting the angles to the model's wavelength (1.5406 A). Generation uses upstream's
# scripts/evaluate.py:reconstructon, as upstream's solve_pxrd does.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.cells import atom_list, primitive_z_candidates  # noqa: E402
from inference._common.pxrd_io import convert_two_theta, load_pattern  # noqa: E402

MODEL_WAVELENGTH = 1.5406  # hparams.yaml: model.wavelength
GRID = np.linspace(5.0, 90.0, 8500)  # upstream xy_data_prep
N_MASKED = 1000  # upstream CrystDataset zeroes the first 1000 points: the effective range is 15-90 deg
MAX_ATOMS = 20  # hparams.yaml: data.max_atoms


def prepare_profile(args) -> tuple[np.ndarray, list[float]]:
    """Upstream xy_data_prep on angles converted to the model's wavelength."""
    from scipy.interpolate import interp1d

    pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
    tt = convert_two_theta(pattern.two_theta, pattern.wavelength, MODEL_WAVELENGTH)
    ok = np.isfinite(tt)
    tt, inten = tt[ok], pattern.intensity[ok]
    covered = [round(float(tt.min()), 2), round(float(tt.max()), 2)]
    if tt.max() <= GRID[0] or tt.min() >= GRID[-1]:
        raise SystemExit(f"The scan covers {covered[0]}-{covered[1]} deg at 1.5406 A, outside the model's 5-90 deg window")
    if covered[0] > 20 or covered[1] < 80:
        print(f"Warning: the scan covers only {covered[0]}-{covered[1]} deg 2theta (1.5406 A) of the "
              "model's effective 15-90 deg window; the rest is filled with zeros, as upstream does.")
    y = interp1d(tt, inten, kind="cubic", bounds_error=False, fill_value=0)(GRID)
    # Upstream's dataset later zeroes everything below 15 deg; require signal where the model looks.
    if not np.isfinite(y).all() or y[N_MASKED:].max() <= 0:
        raise SystemExit("The pattern has no positive intensity between 15 and 90 deg (1.5406 A), "
                         "the range the model actually sees")
    return (y / y.max()).astype(np.float32), covered


def _patch_dimenet_init() -> None:
    """torch-geometric 1.7.2's BesselBasisLayer fills its `freq` parameter in place at construction,
    which torch >= 1.9 rejects. Do the same fill under no_grad; the checkpoint overwrites `freq`
    right after, so the loaded model is unchanged."""
    import math

    import torch
    from torch_geometric.nn.models import dimenet

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(math.pi)

    dimenet.BesselBasisLayer.reset_parameters = reset_parameters


def load_model(model_dir: str):
    """Upstream scripts/eval_utils.load_model, except that the decoder config is passed explicitly:
    the checkpoint's stored hyperparameters point the GemNet scale file at the authors' cluster
    (/home/gridsan/...), while hparams.yaml resolves it inside the upstream checkout."""
    import hydra
    import torch
    from hydra import compose, initialize_config_dir

    _patch_dimenet_init()
    with initialize_config_dir(model_dir):
        cfg = compose(config_name="hparams")
        model = hydra.utils.instantiate(cfg.model, optim=cfg.optim, data=cfg.data, logging=cfg.logging,
                                        _recursive_=False)
        ckpt = os.path.join(model_dir, "epoch=804-step=57154.ckpt")
        model = model.load_from_checkpoint(ckpt, strict=False, data=model.hparams.data, decoder=model.hparams.decoder)
    # Upstream loads with strict=False; refuse a partial load instead of running with random weights.
    stored = set(torch.load(ckpt, map_location="cpu")["state_dict"])
    expected = set(model.state_dict())
    if stored != expected:
        raise SystemExit(f"Checkpoint does not match the model: {len(expected - stored)} missing, "
                         f"{len(stored - expected)} unexpected keys")
    model.lattice_scaler = torch.load(os.path.join(model_dir, "lattice_scaler.pt"))
    model.scaler = torch.load(os.path.join(model_dir, "prop_scaler.pt"))
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Crystalyze PXRD -> crystal structure inference")
    p.add_argument("--pattern", required=True, help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa1, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--composition", required=True, help="Reduced formula, e.g. LuOF")
    p.add_argument("--z", type=int, help="Formula units per conventional cell; fixes the atom count and types")
    p.add_argument("--spacegroup", help="Optional; only used to convert Z to the primitive cell")
    p.add_argument("--n-samples", type=int, default=10, help="Candidates per atom-count hypothesis")
    p.add_argument("--n-step-each", type=int, default=100, help="Langevin steps per noise level (upstream default)")
    p.add_argument("--batch-size", type=int, default=64, help="Samples generated in parallel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-dir", default=os.environ.get("CRYSTALYZE_MODEL", os.path.join(HERE, ".weights", "model_folder")))
    p.add_argument("--upstream", default=os.environ.get("CRYSTALYZE_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    if args.z is not None and args.z < 1:
        p.error("--z must be >= 1")

    # Importing upstream's cdvae.common.utils changes the working directory, so fix paths first.
    args.pattern = os.path.abspath(args.pattern)
    args.model_dir = os.path.abspath(args.model_dir)
    upstream = os.path.abspath(args.upstream)
    os.environ["PROJECT_ROOT"] = upstream  # hparams.yaml resolves the GemNet scale file through it
    sys.path[:0] = [upstream, os.path.join(upstream, "scripts")]
    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)

    import random

    import hydra
    import torch
    import yaml
    from omegaconf import OmegaConf
    from pymatgen.core import Composition, Element, Lattice, Structure

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from cdvae.common.inference_utils import (
        create_inference_dataframe,
        create_inference_graph_data,
        create_inference_xrd_data,
        load_and_modify_config,
    )
    from eval_utils import get_crystals_list
    from evaluate import reconstructon

    profile, covered = prepare_profile(args)

    # Atom-list hypotheses. With Z the atom count and types are forced (upstream's
    # --force_num_atoms --force_atom_types); training cells are Niggli-reduced MP cells, which are
    # primitive, so a conventional Z is converted. Without Z only the element set is constrained.
    if args.z is not None:
        hypotheses = {f"zp{zp}": atom_list(args.composition, zp) for zp in primitive_z_candidates(args.z, args.spacegroup)}
        hypotheses = {k: v for k, v in hypotheses.items() if len(v) <= MAX_ATOMS}
        if not hypotheses:
            raise SystemExit(f"Every atom-count hypothesis exceeds the model's {MAX_ATOMS}-atom limit")
        force = True
    else:
        hypotheses = {"elements": [el.symbol for el in Composition(args.composition).elements]}
        force = False

    start = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        model = load_model(args.model_dir)
        model.to("cuda")  # upstream's sampler hard-codes .cuda(0) when atom types are sampled
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each, step_lr=1e-4, min_sigma=0,
                                save_traj=False, disable_bar=True)

    candidates = []
    for label, symbols in hypotheses.items():
        numbers = [Element(s).Z for s in symbols]
        # n identical inputs run as one batch; each starts from its own random dummy cell, so this
        # draws the same independent samples as upstream's sequential num_evals loop, in parallel.
        inference_data = {f"{label}_{k}": (torch.tensor(profile), numbers) for k in range(args.n_samples)}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            # The three files upstream's solve_pxrd writes, in a scratch directory.
            create_inference_dataframe(inference_data).to_csv(os.path.join(tmp, "test.csv"), index=False)
            torch.save(create_inference_xrd_data(inference_data), os.path.join(tmp, "test_pv_xrd.pt"))
            torch.save(create_inference_graph_data(inference_data), os.path.join(tmp, "test.pt"))
            data_cfg = yaml.safe_load(load_and_modify_config(label).replace("${oc.env:PROJECT_ROOT}/data/" + label, tmp))
            data_cfg["datamodule"]["batch_size"]["test"] = args.batch_size
            # Upstream's data config refers to ${data.*}, so it must sit under a "data" root.
            datamodule = hydra.utils.instantiate(OmegaConf.create({"data": data_cfg}).data.datamodule, _recursive_=False,
                                                 scaler_path=os.path.abspath(args.model_dir))
            datamodule.setup("test")
            loader = datamodule.test_dataloader()[0]
            frac, natoms, types, lengths, angles, *_ = reconstructon(
                loader, model, ld_kwargs, 1, force, force, 10, num_batches=len(loader))
        crystals = get_crystals_list(frac[0], types[0], lengths[0], angles[0], natoms[0])
        for k, c in enumerate(crystals, start=1):
            path = os.path.join(cand_dir, f"candidate_{label}_{k:03d}.cif")
            try:
                s = Structure(Lattice.from_parameters(*c["lengths"].tolist(), *c["angles"].tolist()),
                              [int(z) for z in c["atom_types"]], c["frac_coords"])
                s.to(filename=path)
                candidates.append({"file": os.path.relpath(path, out), "hypothesis": label})
            except Exception as exc:  # degenerate lattices from the decoder
                candidates.append({"file": None, "hypothesis": label, "error": str(exc)[:200]})

    with open(os.path.join(out, "upstream.log"), "w", encoding="utf-8") as fout:
        fout.write(log.getvalue())
    written = [c["file"] for c in candidates if c["file"]]
    results = {
        "model": "crystalyze",
        "checkpoint": os.path.join(args.model_dir, "epoch=804-step=57154.ckpt"),
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "model_dir")},
        "hypotheses": hypotheses,
        "scan_coverage_2theta_at_1.5406": covered,
        "runtime_s": round(time.time() - start, 1),
        "candidates": written,
        "candidate_details": candidates,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(written)} candidate CIFs to {cand_dir}")


if __name__ == "__main__":
    main()
