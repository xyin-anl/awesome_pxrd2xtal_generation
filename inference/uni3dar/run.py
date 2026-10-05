# PXRD inference script for Uni-3DAR (https://github.com/dptech-corp/Uni-3DAR)
# Checkpoint: mp20_pxrd.pt from https://huggingface.co/dptech/Uni-3DAR (setup.sh downloads it)
# 1. Run setup.sh once (environment, Uni-Core, upstream at the pinned commit, checkpoint)
# 2. Run: python run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --z 2 --out results/
# Uni-3DAR conditions on a list of diffraction peaks (pymatgen Cu Ka, 0-120 degrees 2theta,
# intensities 0-100) plus the primitive-cell atom list. This script picks peaks with
# inference/_common/pxrd_io.py and follows upstream's PXRD protocol (uni3dar/inference.py):
# oversample 20x with the composition constraint, keep exact-composition samples, rank by score.
# Requires an NVIDIA GPU with compute capability >= 8.0 (flash-attn); no CPU support.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.cells import atom_list, primitive_z_candidates  # noqa: E402
from inference._common.pxrd_io import (  # noqa: E402
    CU_KA,
    convert_two_theta,
    load_pattern,
    parse_wavelength,
    pick_peaks,
    write_peaks_csv,
)

TWO_THETA_MAX = 120.0  # training window (pymatgen CuKa)
OVERSAMPLE = 20  # upstream: min_generated_samples = total_n * 20
MAX_TRY = 10  # upstream: max_try
DEFAULT_BATCH = 64  # benchmarked default; upstream's 256 needs more than 24 GB

# Flags from upstream scripts/inference_mp20_pxrd.sh (the configuration mp20_pxrd.pt was released with).
UPSTREAM_ARGS = [
    "--task", "uni3dar", "--loss", "ar", "--arch", "uni3dar_sampler", "--bf16",
    "--emb-dim", "1024", "--num-head", "16", "--layer", "24",
    "--data-type", "crystal", "--merge-level", "8",
    "--tree-temperature", "0.15", "--atom-temperature", "0.3", "--xyz-temperature", "0.3",
    "--count-temperature", "1.0", "--rank-ratio", "0.8", "--rank-by", "atom+xyz",
    "--grid-len", "0.24", "--xyz-resolution", "0.01", "--recycle", "1",
    "--atom-type-key", "atom_type", "--atom-pos-key", "atom_pos",
    "--lattice-matrix-key", "lattice_matrix", "--allow-atoms", "all", "--head-dropout", "0.1",
    "--crystal-pxrd", "4", "--crystal-pxrd-step", "0.1", "--crystal-pxrd-noise", "0.1",
    "--crystal-component", "1", "--crystal-component-sqrt", "--crystal-component-noise", "0.1",
    "--crystal-pxrd-threshold", "5", "--max-num-atom", "128",
]

def prepare_peaks(args, out: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Return Cu Ka peak positions/intensities and the Cu Ka 2theta range the scan covered."""
    if args.peaks:
        if not args.wavelength:
            raise SystemExit("--peaks needs --wavelength (the radiation the peak positions refer to)")
        arr = np.loadtxt(args.peaks, delimiter=",", skiprows=1, ndmin=2)
        two_theta, intensity = arr[:, 0], arr[:, 1]
        lam = parse_wavelength(args.wavelength)
        covered = [float(np.nanmin(two_theta)), float(np.nanmax(two_theta))]
    else:
        pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
        strip = {"auto": None, "on": True, "off": False}[args.strip_ka2]
        two_theta, intensity, lam = pick_peaks(pattern, strip_ka2=strip)
        covered = [float(pattern.two_theta.min()), float(pattern.two_theta.max())]
    tt = convert_two_theta(two_theta, lam, CU_KA)
    keep = np.isfinite(tt) & (tt > 0) & (tt <= TWO_THETA_MAX) & np.isfinite(intensity) & (intensity > 0)
    tt, inten = tt[keep], intensity[keep]
    if len(tt) == 0:
        raise SystemExit("No peaks found inside the model's 0-120 degree (Cu Ka) window.")
    # The model has no notion of unmeasured ranges: outside the scan it sees "no peaks".
    covered = [float(np.nan_to_num(v, nan=180.0)) for v in convert_two_theta(np.array(covered), lam, CU_KA)]
    covered = [round(covered[0], 2), round(min(covered[1], TWO_THETA_MAX), 2)]
    if covered[0] > 15 or covered[1] < 90:
        print(f"Warning: the scan covers only {covered[0]}-{covered[1]} deg 2theta (Cu Ka) of the model's "
              f"0-{TWO_THETA_MAX:.0f} deg window; missing ranges are seen as containing no peaks.")
    inten = 100.0 * inten / inten.max()
    order = np.argsort(tt)
    tt, inten = tt[order], inten[order]
    write_peaks_csv(os.path.join(out, "peaks_cuka_0-120.csv"), tt, inten)
    return tt, inten, covered


def load_model(checkpoint: str, upstream: str, batch_size: int, seed: int):
    import torch
    from unicore import options, tasks, utils

    argv = ["unused_data_path", "--user-dir", os.path.join(upstream, "uni3dar"),
            "--batch-size", str(batch_size), "--seed", str(seed), *UPSTREAM_ARGS]
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, input_args=argv)
    utils.import_user_module(args)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    args.model = "uni3dar_sampler"
    task = tasks.setup_task(args)
    model = task.build_model(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema"]["params"], strict=True)
    return model.cuda().bfloat16()


def main() -> None:
    p = argparse.ArgumentParser(description="Uni-3DAR PXRD -> crystal structure inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pattern", help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    src.add_argument("--peaks", help="Pre-picked peaks CSV with header '2theta,intensity'")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--strip-ka2", choices=["auto", "on", "off"], default="auto",
                   help="Merge Cu Ka2 satellites into Ka1 peaks (auto: on for data declared at averaged Cu Ka and for other Cu data whose profile shows the doublet)")
    p.add_argument("--composition", required=True, help="Reduced formula, e.g. TiO2 (required by this model)")
    p.add_argument("--z", type=int, required=True, help="Formula units per conventional cell")
    p.add_argument("--spacegroup", help="Optional; only used to convert Z to the primitive cell")
    p.add_argument("--n-samples", type=int, default=10, help="Top-ranked candidates to write per primitive-Z hypothesis")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="Structures per generation call; at 64 memory reaches ~22 GB for 20-atom cells, use 32 (~6 GB) or 16 on smaller GPUs")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--checkpoint", default=os.environ.get("UNI3DAR_CKPT", os.path.join(HERE, ".weights", "mp20_pxrd.pt")))
    p.add_argument("--upstream", default=os.environ.get("UNI3DAR_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    if args.z < 1:
        p.error("--z must be >= 1")

    # Upstream builds structures with ase inside a bare `except:`; without ase every sample
    # would vanish silently, so import it here to fail fast.
    import ase  # noqa: F401
    from pymatgen.core import Element
    from pymatgen.io.ase import AseAtomsAdaptor

    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)
    pxrd_x, pxrd_y, covered = prepare_peaks(args, out)

    sys.path.insert(0, os.path.abspath(args.upstream))
    start = time.time()
    model = load_model(args.checkpoint, os.path.abspath(args.upstream), args.batch_size, args.seed)

    # Scores are token perplexities and are not comparable across atom counts, so each
    # primitive-Z hypothesis is ranked on its own and contributes its own top --n-samples.
    groups = {}  # primitive Z -> list of (score, ase.Atoms)
    # Smaller batches (for smaller GPUs) get proportionally more calls, so the sample budget
    # matches the default batch size the wrapper was benchmarked with.
    max_try = MAX_TRY * max(1, DEFAULT_BATCH // args.batch_size)
    for z_prim in primitive_z_candidates(args.z, args.spacegroup):
        symbols = atom_list(args.composition, z_prim)
        if len(symbols) > 128:
            print(f"Skipping primitive Z={z_prim}: {len(symbols)} atoms exceeds the model's 128-atom limit")
            continue
        data = {"pxrd_x": pxrd_x, "pxrd_y": pxrd_y, "atom_type": symbols}
        # Upstream's atom_constraint: sorted atomic numbers minus one.
        target = np.array(sorted(Element(sym).Z for sym in symbols)) - 1
        pool, tries, needed = [], 0, args.n_samples * OVERSAMPLE
        while len(pool) < needed:
            res, score = model.generate(data=data, atom_constraint=target)
            for atoms, s in zip(res, score):
                if np.array_equal(np.sort(atoms.get_atomic_numbers()) - 1, target):
                    pool.append((float(s), atoms))
                if len(pool) >= needed:
                    break
            tries += 1
            kept = len(pool)
            # Same stopping rules as upstream inference_crystal_cond.
            if tries > max_try or (kept / (len(res) + 1e-5) <= 0.1 and tries > 2):
                break
        pool.sort(key=lambda item: item[0])  # upstream ranks ascending by score
        groups[z_prim] = pool

    if not any(groups.values()):
        raise SystemExit(
            "No structure with the requested composition was generated. Check the composition and Z; "
            "upstream hides construction errors, so also check that the environment matches environment.yml."
        )
    candidates = []
    for z_prim, pool in groups.items():
        for k, (s, atoms) in enumerate(pool[: args.n_samples], start=1):
            path = os.path.join(cand_dir, f"candidate_zp{z_prim}_{k:03d}.cif")
            AseAtomsAdaptor.get_structure(atoms).to(filename=path)
            candidates.append({"file": os.path.relpath(path, out), "score": s, "primitive_z": z_prim, "rank_within_z": k})

    results = {
        "model": "uni3dar",
        "checkpoint": os.path.basename(args.checkpoint),
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "checkpoint")},
        "n_exact_composition_samples": {str(z): len(pool) for z, pool in groups.items()},
        "scan_coverage_cuka_2theta": covered,
        "runtime_s": round(time.time() - start, 1),
        "candidates": [c["file"] for c in candidates],
        "candidate_details": candidates,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    n_pool = sum(len(pool) for pool in groups.values())
    print(f"Wrote {len(candidates)} candidate CIFs to {cand_dir} (from {n_pool} exact-composition samples)")


if __name__ == "__main__":
    main()
