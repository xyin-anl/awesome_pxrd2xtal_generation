# PXRD inference script for XRDSol (https://github.com/ai4mat-zhu/XRDSol)
# Checkpoint: xrdsol/prop_models/mp20/epoch=189-step=5129.ckpt (git LFS; setup.sh downloads it)
# 1. Run setup.sh once (environment, upstream at the pinned commit, checkpoint)
# 2. Run: python run.py --pattern scan.xy --wavelength CuKa --composition LuOF --z 2 --spacegroup P4/nmm \
#           --cell 3.849,3.849,5.31,90,90,90 --out results/
# XRDSol solves atomic coordinates for a known cell and composition. It was trained on primitive
# cells with pymatgen Cu Ka peak lists rendered as Voigt profiles (xrdsol/common/data_utils.py:
# process_one); this script picks peaks from the measured pattern, renders them the same way, and
# converts a conventional cell and Z to the primitive cell.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.cells import CENTERING, atom_list  # noqa: E402
from inference._common.pxrd_io import (  # noqa: E402
    CU_KA,
    convert_two_theta,
    load_pattern,
    parse_wavelength,
    pick_peaks,
)

MAX_ATOMS = 20  # hparams.yaml: data.max_atoms
GRID = np.arange(0, 90, 0.02)  # process_one: wider_x
ALPHAGAMMA = 0.03  # process_one: initial_alphagamma
STEP_LR = 1e-5  # scripts/eval_utils.py: recommand_step_lr["csp"/"csp_multi"]["mp_20"]

# Lattice-point translations of each centering (R in the hexagonal setting, obverse).
TRANSLATIONS = {
    "P": [],
    "A": [[0, 0.5, 0.5]],
    "B": [[0.5, 0, 0.5]],
    "C": [[0.5, 0.5, 0]],
    "I": [[0.5, 0.5, 0.5]],
    "F": [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    "R": [[2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]],
}


def primitive_cell(cell: list[float], spacegroup: str | None, z: int, cell_is_primitive: bool):
    """Return (primitive lattice parameters, primitive Z)."""
    from pymatgen.core import Lattice, Structure

    if cell_is_primitive:
        return cell, z
    if not spacegroup:
        raise SystemExit("XRDSol works in the primitive cell: give --spacegroup to convert the conventional "
                         "cell, or --cell-is-primitive if --cell and --z already describe the primitive cell")
    letter = spacegroup.strip()[0].upper()
    if letter == "R":
        a, b, c, al, be, ga = cell
        if np.allclose([al, be, ga], [90, 90, 120], atol=0.5) and abs(a - b) < 1e-3:
            pass  # hexagonal axes: three lattice points, converted below
        elif np.allclose([a, b], c, rtol=1e-3) and np.allclose([al, be], ga, atol=0.05):
            return cell, z  # rhombohedral axes are already primitive
        else:
            raise SystemExit("R space group with a cell in neither the hexagonal nor the rhombohedral setting")
    m = CENTERING[letter]
    if z % m:
        raise SystemExit(f"Z={z} is not divisible by the {m} lattice points of {spacegroup}")
    lattice = Lattice.from_parameters(*cell)
    points = [[0, 0, 0]] + TRANSLATIONS[letter]
    prim = Structure(lattice, ["H"] * len(points), points).get_primitive_structure(tolerance=0.01)
    if len(prim) != 1:
        raise SystemExit("Could not reduce the cell with the given centering; check --cell and --spacegroup")
    return list(prim.lattice.abc) + list(prim.lattice.angles), z // m


def r_cos(structure, target: np.ndarray, get_voigt_xrd) -> float:
    """Cosine similarity between a structure's training-style pattern and the target pattern."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    try:
        xrd = XRDCalculator().get_pattern(structure, scaled=False)
        sim = get_voigt_xrd(GRID, np.asarray(xrd.x), np.asarray(xrd.y), ALPHAGAMMA, 1.54056)
    except Exception:
        return -1.0
    denom = np.linalg.norm(sim) * np.linalg.norm(target)
    return float(sim @ target / denom) if denom > 0 else -1.0


def condition_pattern(args, get_voigt_xrd) -> tuple[np.ndarray, list]:
    """Peaks at pymatgen's averaged Cu Ka (XRDCalculator() default, used in training), rendered by
    upstream's get_voigt_xrd on the 0-90 deg, 0.02 deg grid."""
    if args.peaks:
        if not args.wavelength:
            raise SystemExit("--peaks needs --wavelength (the radiation the peak positions refer to)")
        arr = np.loadtxt(args.peaks, delimiter=",", skiprows=1, ndmin=2)
        two_theta, intensity, lam = arr[:, 0], arr[:, 1], parse_wavelength(args.wavelength)
    else:
        pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
        strip = {"auto": None, "on": True, "off": False}[args.strip_ka2]
        two_theta, intensity, lam = pick_peaks(pattern, strip_ka2=strip)
    tt = convert_two_theta(two_theta, lam, CU_KA)
    keep = np.isfinite(tt) & (tt > 0) & (tt < GRID[-1]) & np.isfinite(intensity) & (intensity > 0)
    if not keep.any():
        raise SystemExit("No peaks inside the model's 0-90 degree (Cu Ka) window")
    profile = get_voigt_xrd(GRID, tt[keep], intensity[keep], ALPHAGAMMA, 1.54056)
    peaks = sorted(zip(np.round(tt[keep], 4).tolist(), np.round(100 * intensity[keep] / intensity[keep].max(), 2).tolist()))
    return profile.astype(np.float32), peaks


def main() -> None:
    p = argparse.ArgumentParser(description="XRDSol PXRD + cell -> atomic coordinates")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pattern", help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    src.add_argument("--peaks", help="Pre-picked peaks CSV with header '2theta,intensity' (needs --wavelength)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--strip-ka2", choices=["auto", "on", "off"], default="auto",
                   help="Merge Cu Ka2 satellites into Ka1 peaks (auto: on for data declared at averaged Cu Ka and for other Cu data whose profile shows the doublet)")
    p.add_argument("--composition", required=True, help="Reduced formula, e.g. LuOF")
    p.add_argument("--z", type=int, required=True, help="Formula units per conventional cell")
    p.add_argument("--cell", required=True, help="Conventional cell a,b,c,alpha,beta,gamma (Angstrom, degrees)")
    p.add_argument("--spacegroup", help="Hermann-Mauguin symbol; used to convert the cell and Z to the primitive cell")
    p.add_argument("--cell-is-primitive", action="store_true", help="--cell and --z already describe the primitive cell")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-dir", default=os.environ.get("XRDSOL_MODEL", os.path.join(HERE, ".weights", "mp20")))
    p.add_argument("--upstream", default=os.environ.get("XRDSOL_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    if args.z < 1:
        p.error("--z must be >= 1")
    try:
        cell = [float(v) for v in args.cell.split(",")]
        assert len(cell) == 6 and all(v > 0 for v in cell)
    except (ValueError, AssertionError):
        p.error("--cell must be six positive numbers: a,b,c,alpha,beta,gamma")

    prim_cell, z_prim = primitive_cell(cell, args.spacegroup, args.z, args.cell_is_primitive)
    symbols = atom_list(args.composition, z_prim)
    if len(symbols) > MAX_ATOMS:
        raise SystemExit(f"{len(symbols)} atoms in the primitive cell exceeds the model's {MAX_ATOMS}-atom limit (MP-20)")

    args.pattern = os.path.abspath(args.pattern) if args.pattern else None
    args.peaks = os.path.abspath(args.peaks) if args.peaks else None
    model_dir = os.path.abspath(args.model_dir)
    upstream = os.path.abspath(args.upstream)
    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)
    os.environ["PROJECT_ROOT"] = upstream
    sys.path[:0] = [upstream, os.path.join(upstream, "scripts")]

    import torch
    from pymatgen.core import Element, Lattice, Structure
    from torch_geometric.data import Batch, Data

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log = io.StringIO()
    start = time.time()
    # Upstream imports "scripts.eval_utils"; this repository has its own scripts/ package, which
    # would shadow upstream's namespace package, so register upstream's directory explicitly.
    import types

    upstream_scripts = types.ModuleType("scripts")
    upstream_scripts.__path__ = [os.path.join(upstream, "scripts")]
    sys.modules["scripts"] = upstream_scripts

    # Upstream calls faulthandler.enable() on import, which needs the real stderr.
    import hydra
    from eval_utils import get_crystals_list, lattices_to_params_shape
    from hydra import compose, initialize_config_dir
    from xrdsol.common.data_utils import get_voigt_xrd

    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        profile, peaks = condition_pattern(args, get_voigt_xrd)
        # scripts/eval_utils.py:load_model, with a check that every weight is loaded.
        with initialize_config_dir(model_dir):
            cfg = compose(config_name="hparams")
            model = hydra.utils.instantiate(cfg.model, optim=cfg.optim, data=cfg.data, logging=cfg.logging,
                                            _recursive_=False)
            ckpt = os.path.join(model_dir, "epoch=189-step=5129.ckpt")
            model = model.load_from_checkpoint(ckpt, hparams_file=os.path.join(model_dir, "hparams.yaml"), strict=False)
        stored = set(torch.load(ckpt, map_location="cpu")["state_dict"])
        if stored != set(model.state_dict()):
            raise SystemExit("Checkpoint does not match the model (missing or unexpected weights)")
        model.lattice_scaler = torch.load(os.path.join(model_dir, "lattice_scaler.pt"))
        model.scaler = torch.load(os.path.join(model_dir, "prop_scaler.pt"))
        model = model.cuda().eval()

        # solution.py:SampleDataset, one copy of the input per sample.
        atom_types = torch.LongTensor([Element(s).Z for s in symbols])
        data = Data(atom_types=atom_types, num_atoms=len(symbols), num_nodes=len(symbols),
                    lengths=torch.Tensor(prim_cell[:3]).view(1, -1), angles=torch.Tensor(prim_cell[3:]).view(1, -1),
                    xrd=torch.Tensor(profile))
        batch = Batch.from_data_list([data.clone() for _ in range(args.n_samples)]).cuda()
        with torch.no_grad():
            outputs, _ = model.sample(batch, step_lr=STEP_LR)
        lengths, angles = lattices_to_params_shape(outputs["lattices"])  # the cell stays fixed during sampling
        crystals = get_crystals_list(outputs["frac_coords"].cpu(), outputs["atom_types"].cpu(),
                                     lengths.cpu(), angles.cpu(), outputs["num_atoms"].cpu())

    with open(os.path.join(out, "upstream.log"), "w", encoding="utf-8") as fout:
        fout.write(log.getvalue())
    # The paper ranks candidates by R_cos, the cosine similarity between each candidate's simulated
    # pattern (rendered as in training) and the target pattern, and reports the top one.
    structures = [Structure(Lattice.from_parameters(*np.asarray(c["lengths"]).tolist(), *np.asarray(c["angles"]).tolist()),
                            [int(t) for t in c["atom_types"]], c["frac_coords"]) for c in crystals]
    scores = [r_cos(s, profile, get_voigt_xrd) for s in structures]
    candidates, details = [], []
    for k, i in enumerate(np.argsort(scores)[::-1], start=1):
        path = os.path.join(cand_dir, f"candidate_{k:03d}.cif")
        structures[i].to(filename=path)
        candidates.append(os.path.relpath(path, out))
        details.append({"file": candidates[-1], "r_cos": round(float(scores[i]), 5)})

    results = {
        "model": "xrdsol",
        "checkpoint": os.path.join(model_dir, "epoch=189-step=5129.ckpt"),
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "model_dir")},
        "primitive_cell": [round(v, 5) for v in prim_cell],
        "primitive_z": z_prim,
        "atom_list": symbols,
        "peaks_cuka": peaks,
        "runtime_s": round(time.time() - start, 1),
        "candidates": candidates,
        "candidate_details": details,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(candidates)} candidate CIFs to {cand_dir}")


if __name__ == "__main__":
    main()
