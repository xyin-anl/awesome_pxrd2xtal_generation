# PXRD inference script for deCIFer (https://github.com/FrederikLizakJohansen/deCIFer)
# Checkpoint: decifer_v1_ckpt.pt from the deCIFer data archive on ERDA (setup.sh downloads it)
# 1. Run setup.sh once (creates the environment, clones upstream at the pinned commit, downloads the checkpoint)
# 2. Run: python run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --z 2 --out results/
# The pattern goes through upstream's own experimental preprocessing (bin/experimental_pipeline.py):
# conversion to Q, normalization, cropping, and interpolation onto the 0-10 1/Angstrom grid.
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.pxrd_io import load_pattern  # noqa: E402


def cell_formula_prompt(composition: str, z: int) -> str:
    """Return the data_ block name deCIFer was trained on, e.g. TiO2, Z=2 -> 'Ti2O4'.

    Training CIFs name the block after pymatgen's _chemical_formula_sum for the full cell,
    with explicit counts of 1 (KCaCO3F, Z=1 -> 'K1Ca1C1O3F1').
    """
    from pymatgen.core import Composition

    return (Composition(composition) * z).formula.replace(" ", "")


def main() -> None:
    p = argparse.ArgumentParser(description="deCIFer PXRD -> crystal structure inference")
    p.add_argument("--pattern", required=True, help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--composition", help="Reduced formula, e.g. TiO2; omit for composition-free generation")
    p.add_argument("--z", type=int, default=1, help="Formula units per cell (used with --composition)")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--q-min", type=float, default=0.0, help="Lower Q crop in 1/Angstrom (authors' experiments: 1.5)")
    p.add_argument("--q-max", type=float, default=10.0, help="Upper Q crop in 1/Angstrom (authors' experiments: 8)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=3000)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--checkpoint", default=os.environ.get("DECIFER_CKPT", os.path.join(HERE, ".weights", "decifer_v1_ckpt.pt")))
    p.add_argument("--upstream", default=os.environ.get("DECIFER_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()

    if not 0.0 <= args.q_min < args.q_max <= 10.0:
        p.error("need 0 <= --q-min < --q-max <= 10 (the model's Q grid ends at 10 1/Angstrom)")
    if args.composition and args.z < 1:
        p.error("--z must be >= 1")

    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)

    pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
    q = 4.0 * np.pi * np.sin(np.radians(pattern.two_theta) / 2.0) / pattern.wavelength
    in_window = (q > args.q_min) & (q < args.q_max)
    if in_window.sum() < 10 or np.ptp(pattern.intensity[in_window]) <= 0:
        raise SystemExit(
            f"The pattern covers Q {q.min():.2f}-{q.max():.2f} 1/Angstrom, which leaves no usable signal "
            f"inside --q-min {args.q_min} / --q-max {args.q_max}; the model would see an empty condition."
        )
    prompt_formula = cell_formula_prompt(args.composition, args.z) if args.composition else None

    sys.path.insert(0, os.path.abspath(args.upstream))
    import __main__
    from bin.experimental_pipeline import DeciferPipeline
    from bin.train import TrainConfig

    # The checkpoint pickles TrainConfig as __main__.TrainConfig (it was saved from train.py).
    __main__.TrainConfig = TrainConfig

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        # Upstream reads a directory of .xy files; hand it the pattern in that form.
        with open(os.path.join(tmp, "sample.xy"), "w", encoding="utf-8") as fout:
            for t, i in zip(pattern.two_theta, pattern.intensity):
                fout.write(f"{t} {i}\n")
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            pipe = DeciferPipeline(
                model_path=args.checkpoint,
                zip_path=tmp,
                device=args.device,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                results_output_folder=os.path.join(tmp, "results"),
            )
            # A U-deCIFer (unconditioned) checkpoint loads fine but ignores the pattern.
            cfg = pipe.model.config
            if not getattr(cfg, "condition", False) or getattr(cfg, "condition_size", None) != 1000:
                raise SystemExit(f"{args.checkpoint} is not a PXRD-conditioned deCIFer checkpoint")
            pipe.prepare_target_data(
                target_file="sample.xy",
                wavelength=pattern.wavelength,
                q_min_crop=args.q_min,
                q_max_crop=args.q_max,
            )
            pipe.run_experiment_protocol(n_trials=args.n_samples, composition=prompt_formula)
    with open(os.path.join(out, "upstream.log"), "w", encoding="utf-8") as fout:
        fout.write(log.getvalue())

    candidates = []
    for k, gen in enumerate(pipe.results["gens"], start=1):
        path = os.path.join(cand_dir, f"candidate_{k:03d}.cif")
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(gen["cif_str"])
        candidates.append(os.path.relpath(path, out))

    results = {
        "model": "decifer",
        "checkpoint": os.path.basename(args.checkpoint),
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "checkpoint")},
        "prompt_formula": prompt_formula,
        "runtime_s": round(time.time() - start, 1),
        "n_requested": args.n_samples,
        "candidates": candidates,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(candidates)} candidate CIFs to {cand_dir} ({args.n_samples} requested; unparsable outputs are dropped)")


if __name__ == "__main__":
    main()
