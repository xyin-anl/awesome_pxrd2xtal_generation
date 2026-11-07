# PXRD inference script for Bragglite (PXRD-Bench, https://github.com/dongyeop3813/PXRD-Bench)
# Checkpoints: single-phase files from bragglite-checkpoints-v1.tar.gz, Zenodo record 23878332
# (setup.sh downloads and verifies them).
# 1. Run setup.sh once (environment, upstream at the pinned commit, checkpoints)
# 2. Run: python run.py --pattern scan.xy --wavelength CuKa1 --composition BaTiO3 --z 1 --out results/
# Single-phase modes only: known composition (exact primitive-cell multiplicities) and element set
# (elements plus the primitive-cell atom count). The pattern goes through upstream's own
# experimental-profile loader (src/data/experimental_pxrd.py, the opXRD path used by
# eval_bragglite: wavelength canonicalization to 1.54184 A, interpolation onto 850 bins over
# 5-90 deg, Savitzky-Golay smoothing, order-5 polynomial background, peaks_128x4 extraction), and
# candidates come from upstream's evaluation sampler (src/pxrd/eval_task_runtime.py:_prediction_rows,
# EMA weights, per-candidate stateless seeds). Candidates are unrefined and not ranked.
# License: Bragglite code is MIT, but upstream retains FlowMM-derived helpers under CC BY-NC 4.0
# (noncommercial); the checkpoints are CC BY 4.0. See README.md.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from inference._common.pxrd_io import CU_KA, CU_KA1, load_pattern  # noqa: E402

CHECKPOINTS = {  # (training set, mode) -> file in checkpoints/single_phase of the archive
    ("pxrd_bench", "composition"): "pxrd_bench_stoichiometry.pt",
    ("pxrd_bench", "element-set"): "pxrd_bench_element_set.pt",
    ("mp20", "composition"): "mp20_stoichiometry.pt",
    ("mp20", "element-set"): "mp20_element_set.pt",
}
TASK = {"composition": "stoich", "element-set": "element_set"}  # upstream task names
CHECKPOINT_SUBDIR = os.path.join("bragglite-checkpoints-v1", "checkpoints", "single_phase")  # archive layout
DEFAULT_WEIGHTS = os.path.join(os.environ.get("BRAGGLITE_WEIGHTS", os.path.join(HERE, ".weights")), CHECKPOINT_SUBDIR)
GRID_MIN, GRID_MAX = 5.0, 90.0  # experimental_pxrd.py: np.linspace(5, 90, 850, endpoint=False)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_wavelength(declared: float, mode: str) -> tuple[float, str]:
    """Wavelength the measured peak maxima refer to.

    Upstream maps each opXRD profile with its primary wavelength. For Cu radiation the maxima of
    a K-alpha1/K-alpha2 doublet sit at K-alpha1 even when the file declares the averaged Cu K-alpha
    wavelength (inference/AGENTS.md, lessons), so by default Cu data are mapped from K-alpha1.
    """
    is_cu = abs(declared - CU_KA) < 0.003 or abs(declared - CU_KA1) < 0.003
    if mode == "ka1" and is_cu:
        return CU_KA1, "Cu data: peak maxima treated as Cu K-alpha1 (1.54056 A)"
    return declared, "as declared"


def hypotheses_from_args(args, nmax: int) -> dict[str, list[str]]:
    """Atom lists, one per primitive-cell hypothesis. In element-set mode only their element set
    and length reach the model, as in upstream's oracle element-set evaluation."""
    from pymatgen.core import Composition

    if args.n_atoms is not None:
        if args.elements:
            symbols = [s.strip() for s in args.elements.split(",") if s.strip()]
            if len(symbols) > args.n_atoms:
                raise SystemExit("--n-atoms must be at least the number of elements")
            # Each element appears at least once; the split between them is left to the model.
            return {f"n{args.n_atoms}": [symbols[i % len(symbols)] for i in range(args.n_atoms)]}
        comp = Composition(args.composition)
        per_fu = int(round(comp.num_atoms))
        if args.n_atoms % per_fu:
            raise SystemExit(f"--n-atoms {args.n_atoms} is not a multiple of the {per_fu} atoms in {args.composition}")
        return {f"zp{args.n_atoms // per_fu}": atom_list(args.composition, args.n_atoms // per_fu)}
    hyps = {f"zp{zp}": atom_list(args.composition, zp) for zp in primitive_z_candidates(args.z, args.spacegroup)}
    kept = {k: v for k, v in hyps.items() if len(v) <= nmax}
    if not kept:
        raise SystemExit(f"Every primitive-cell hypothesis exceeds the checkpoint's {nmax}-atom limit")
    return kept


def write_profile_input(tmp: str, two_theta: np.ndarray, intensity: np.ndarray) -> None:
    """The CSV/CIF pair upstream's ExperimentalPXRDTokens reads from a directory. The CIF is a
    one-atom placeholder: only the pattern features of the loaded item are used."""
    with open(os.path.join(tmp, "pattern.csv"), "w", encoding="utf-8") as fout:
        fout.write("two_theta,intensity\n")
        for t, y in zip(two_theta, intensity):
            fout.write(f"{t:.6f},{y:.8g}\n")
    with open(os.path.join(tmp, "pattern.cif"), "w", encoding="utf-8") as fout:
        fout.write(
            "data_placeholder\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nH1 H 0 0 0\n"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Bragglite PXRD -> crystal structure inference (single phase)")
    p.add_argument("--pattern", required=True, help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa1, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--mode", choices=["composition", "element-set"], default="composition",
                   help="composition: exact atom multiplicities; element-set: elements and atom count only")
    p.add_argument("--composition", help="Reduced formula, e.g. BaTiO3")
    p.add_argument("--z", type=int, help="Formula units per conventional cell (converted to the primitive cell)")
    p.add_argument("--spacegroup", help="Optional; only used to convert Z to the primitive cell")
    p.add_argument("--n-atoms", type=int, help="Atoms in the primitive cell (instead of --z)")
    p.add_argument("--elements", help="Element-set mode without a composition: comma-separated symbols (needs --n-atoms)")
    p.add_argument("--training-set", choices=["pxrd_bench", "mp20"], default="pxrd_bench",
                   help="Checkpoint family: PXRD-Bench (<=52 atoms, default) or MP-20 (<=20 atoms)")
    p.add_argument("--n-samples", type=int, default=20, help="Candidates per atom-count hypothesis (upstream K=20)")
    p.add_argument("--seed", type=int, default=777, help="Upstream evaluation seed")
    p.add_argument("--steps", type=int, help="Sampler steps (default: the checkpoint's, 150)")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema", help="EMA weights (upstream default) or raw")
    p.add_argument("--fp32", action="store_true", help="Disable bfloat16 autocast on CUDA (upstream uses bf16)")
    p.add_argument("--no-background-subtraction", action="store_true",
                   help="Skip upstream's order-5 polynomial background (for already background-free profiles)")
    p.add_argument("--cu-positions", choices=["ka1", "declared"], default="ka1",
                   help="Wavelength Cu peak maxima refer to: K-alpha1 (default) or the declared value")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--batch-size", type=int, default=320, help="Candidates sampled per batch (the released settings use candidate_chunk_size=320)")
    p.add_argument("--checkpoint-dir", default=DEFAULT_WEIGHTS,
                   help="Directory with the single-phase .pt files (default: $BRAGGLITE_WEIGHTS or .weights, as setup.sh)")
    p.add_argument("--upstream", default=os.environ.get("BRAGGLITE_UPSTREAM", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()

    if args.elements and args.mode != "element-set":
        p.error("--elements is for --mode element-set")
    if args.elements and args.n_atoms is None:
        p.error("--elements needs --n-atoms")
    if not args.elements and not args.composition:
        p.error("--composition is required (or --elements with --n-atoms in element-set mode)")
    if args.n_atoms is None and args.z is None:
        p.error("Bragglite needs the primitive-cell atom count: give --z (formula units per conventional cell) or --n-atoms")
    if args.n_atoms is not None and args.z is not None:
        p.error("give --z or --n-atoms, not both")
    for name in ("z", "n_atoms", "n_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")

    args.pattern = os.path.abspath(args.pattern)
    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)
    ckpt_path = os.path.join(os.path.abspath(args.checkpoint_dir), CHECKPOINTS[(args.training_set, args.mode)])
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint {ckpt_path} not found; run setup.sh first")
    upstream = os.path.abspath(args.upstream)
    sys.path.insert(0, upstream)

    pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
    lam_src, lam_note = source_wavelength(pattern.wavelength, args.cu_positions)

    import torch
    from pymatgen.io.cif import CifWriter

    from src.data.experimental_pxrd import _canonicalize_wavelength
    from src.data.mp20_tokens import VZ
    from src.eval_util.checkpoint_runtime import (
        build_model_from_checkpoint,
        checkpoint_model_args,
        checkpoint_type_encoding_name,
        load_checkpoint,
        maybe_apply_checkpoint_ema,
    )
    from src.models.type_encoding import build_type_encoding
    from src.pxrd.eval_metrics import record_to_structure
    from src.pxrd.eval_task_runtime import _build_dataset, _prediction_rows, _sampler_config
    from src.pxrd.peaks import d_to_two_theta

    start = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu (slow)")
    device = torch.device(args.device)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        model_args = checkpoint_model_args(checkpoint)
        if not model_args.get("pxrd_cond"):
            raise SystemExit("checkpoint was not trained with pxrd_cond=true")
        model_lam = float(model_args.get("pxrd_wavelength", 1.54184))
        nmax = int(model_args.get("nmax") or 20)
        model, _ = build_model_from_checkpoint(checkpoint, device=device)
        ema_used = maybe_apply_checkpoint_ema(model, checkpoint, args.weights, log_prefix="[bragglite]")
        model.eval()
        type_encoding = build_type_encoding(checkpoint_type_encoding_name(checkpoint, model_args), vz=VZ)
    if args.mode == "element-set" and not model_args.get("element_set_cond"):
        raise SystemExit(f"{ckpt_path} is not an element-set checkpoint")
    hypotheses = hypotheses_from_args(args, nmax)

    # Upstream's experimental loader assumes the CSV is already at the checkpoint wavelength
    # (csv_records passes target_wavelength), so convert through d-spacing first with upstream's
    # own function, exactly as it does for opXRD records.
    tt, inten = _canonicalize_wavelength(pattern.two_theta, pattern.intensity, lam_src, model_lam)
    order = np.argsort(tt)
    tt, inten = tt[order], inten[order]
    if not len(tt) or tt.max() <= GRID_MIN or tt.min() >= GRID_MAX:
        raise SystemExit(f"The scan does not reach the model's {GRID_MIN:g}-{GRID_MAX:g} deg window at {model_lam} A")
    covered = [round(float(max(tt.min(), GRID_MIN)), 2), round(float(min(tt.max(), GRID_MAX)), 2)]
    if covered[0] > 20 or covered[1] < 70:
        print(f"Warning: the scan covers only {covered[0]}-{covered[1]} deg 2theta at {model_lam} A of the "
              f"model's {GRID_MIN:g}-{GRID_MAX:g} deg window; the rest is zero, as upstream does.")

    with tempfile.TemporaryDirectory() as tmp:
        write_profile_input(tmp, tt, inten)
        data_args = SimpleNamespace(
            dataset_class="experimental_pxrd", dataset_name="user_pattern", data_root=tmp, split="test",
            nmax=nmax, real_pxrd_variant="all", real_pxrd_peak_preprocessing="official",
            experimental_background_subtraction=not args.no_background_subtraction,
            structure_cache=None, simulated_profile_augmentation_seed=None, phase_reference_root=tmp,
            diagnostic_pxrd_features_path=None,
        )
        with contextlib.redirect_stdout(log):
            dataset, _, _ = _build_dataset(data_args, model_args)
        if len(dataset) != 1:
            raise SystemExit("upstream's experimental loader did not return the pattern")
        source = dict(dataset[0])
    features = source["pxrd_d"].reshape(-1, 4).numpy()  # width (deg), prominence, intensity, d
    n_peaks = int((features[:, 2] > 0).sum())
    if n_peaks == 0:
        raise SystemExit("Upstream's peak extraction found no peaks in the pattern")

    # One token item per hypothesis: the pattern features plus the atom types (only their set and
    # count in element-set mode). F1/Y1 are placeholders that only feed upstream's reference record.
    from pymatgen.core import Element

    items = []
    for label, symbols in hypotheses.items():
        a0 = torch.zeros(nmax, dtype=torch.long)
        a0[: len(symbols)] = torch.tensor([Element(s).Z for s in symbols])
        pad = torch.ones(nmax, dtype=torch.bool)
        pad[: len(symbols)] = False
        items.append({**source, "mp_id": label, "A0": a0, "pad_mask": pad, "num_atoms": len(symbols)})

    task = TASK[args.mode]
    sample_args = SimpleNamespace(
        seed=args.seed, num_candidates=args.n_samples, batch_candidates=True,
        candidate_chunk_size=max(args.batch_size, args.n_samples), chunk_size=256, compile=False,
        bf16=not args.fp32, sampling_backend="edm", rng_mode="stateless_batch",
        sample_num_steps=args.steps, solver=None,
        **{k: None for k in ("sigma_min", "sigma_max", "rho", "S_churn", "S_min", "S_max", "S_noise",
                             "aa_frac_max_scale", "aa_rho_types", "aa_rho_coords", "aa_rho_lattice",
                             "sigma_min_types", "sigma_max_types", "rho_types", "sigma_min_coords",
                             "sigma_max_coords", "rho_coords", "sigma_min_lattice", "sigma_max_lattice",
                             "rho_lattice")},
    )
    sampler = _sampler_config(sample_args, model_args)
    os.environ["BRAGGLITE_FAST_SAMPLER"] = "1"  # upstream runtime.fast_sampler default
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_sample = time.time()
    metadata: dict = {}
    with contextlib.redirect_stdout(log):
        rows = list(_prediction_rows(task=task, args=sample_args, model=model, type_encoding=type_encoding,
                                     dataset=items, indices=list(range(len(items))), model_args=model_args,
                                     sampler=sampler, device=device, metadata=metadata))
    sampling_s = time.time() - t_sample

    candidates = []
    for row in rows:
        for rank, record in enumerate(row["candidates"]):
            name = f"candidate_{row['sample_id']}_{rank + 1:03d}.cif"
            structure, error = record_to_structure(record)
            entry = {"file": None, "hypothesis": row["sample_id"], "rank": rank}
            if structure is None:
                entry["error"] = str(error)[:200]
            else:
                # CifWriter.write_file goes through monty.zopen, which the locked monty rejects.
                with open(os.path.join(cand_dir, name), "w", encoding="utf-8") as fout:
                    fout.write(str(CifWriter(structure)))
                entry.update(file=os.path.join("candidates", name), formula=structure.composition.formula)
            candidates.append(entry)

    with open(os.path.join(out, "conditioning_peaks.csv"), "w", encoding="utf-8") as fout:
        fout.write(f"two_theta_at_{model_lam},d_angstrom,intensity,prominence,width_deg\n")
        for w, prom, inten_, d in features[features[:, 2] > 0]:
            fout.write(f"{float(d_to_two_theta(np.array([d]), wavelength=model_lam)[0]):.4f},{d:.5f},"
                       f"{inten_:.5f},{prom:.5f},{w:.4f}\n")
    with open(os.path.join(out, "upstream.log"), "w", encoding="utf-8") as fout:
        fout.write(log.getvalue())
    written = [c["file"] for c in candidates if c["file"]]
    results = {
        "model": "bragglite",
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "training_set": args.training_set,
        "mode": args.mode,
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "checkpoint_dir")},
        "checkpoint_nmax": nmax,
        "hypotheses": {k: {"n_atoms": len(v), "atoms": v if args.mode == "composition" else sorted(set(v))}
                       for k, v in hypotheses.items()},
        "preprocessing": {
            "declared_wavelength": pattern.wavelength,
            "source_wavelength": lam_src,
            "source_wavelength_note": lam_note,
            "model_wavelength": model_lam,
            "scan_coverage_2theta_at_model_wavelength": covered,
            "background_subtraction": not args.no_background_subtraction,
            "n_conditioning_peaks": n_peaks,
        },
        "weights": "ema" if ema_used else "raw",
        "sampler_steps": sampler.sample_num_steps,
        "seed": args.seed,
        "device": str(device),
        "runtime_s": round(time.time() - start, 1),
        "sampling_s": round(sampling_s, 2),
        "gpu_peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
        "ranked": False,
        "candidates": written,
        "candidate_details": candidates,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(written)} candidate CIFs to {cand_dir}")


if __name__ == "__main__":
    main()
