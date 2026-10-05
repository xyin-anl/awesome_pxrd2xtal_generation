# PXRD inference script for CrystaLLM-pi (https://github.com/C-Bone-UCL/CrystaLLM-pi)
# Weights: c-bone/CrystaLLM-pi_Mattergen-XRD and c-bone/CrystaLLM-pi_Chili100K-XRD on Hugging Face
# 1. Run setup.sh once (creates the environment and clones upstream at the pinned commit)
# 2. Run: python run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --z 2 --out results/
# CrystaLLM-pi conditions on the 20 strongest *picked peaks*, not on a raw profile. This script
# picks peaks with inference/_common/pxrd_io.py; pass --peaks to supply your own peak list.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.pxrd_io import (  # noqa: E402
    CU_KA,
    CU_KA1,
    convert_two_theta,
    load_pattern,
    parse_wavelength,
    pick_peaks,
    write_peaks_csv,
)

MODELS = {
    "mattergen": "c-bone/CrystaLLM-pi_Mattergen-XRD",
    "chili100k": "c-bone/CrystaLLM-pi_Chili100K-XRD",
}
MAX_PEAKS = 20
TWO_THETA_MAX = 90.0  # training window, CuKa
SEARCH_ZS = [1, 2, 3, 4, 6]  # upstream DEFAULT_Z_LIST


def prepare_peaks(args, workdir: str) -> str:
    """Write the top-20 peak CSV that upstream conditions on and return its path.

    Training peaks were simulated by pymatgen with the averaged Cu Ka wavelength (1.54184 A),
    while upstream's input path converts user data to Cu Ka1 (1.54056 A). To give the model
    training-consistent angles, peaks are converted to averaged Cu Ka here and upstream is told
    the data are already Ka1, which makes its own conversion a no-op.
    """
    if args.peaks:
        if not args.wavelength:
            raise SystemExit("--peaks needs --wavelength (the radiation the peak positions refer to)")
        arr = np.loadtxt(args.peaks, delimiter=",", skiprows=1, ndmin=2)
        two_theta, intensity = arr[:, 0], arr[:, 1]
        lam = parse_wavelength(args.wavelength)
    else:
        pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
        strip = {"auto": None, "on": True, "off": False}[args.strip_ka2]
        two_theta, intensity, lam = pick_peaks(pattern, strip_ka2=strip)

    # Keep only peaks inside the model's 0-90 degree window before taking the top 20, so
    # short-wavelength data do not lose peaks to upstream's later filter.
    tt = convert_two_theta(two_theta, lam, CU_KA)
    keep = np.isfinite(tt) & (tt > 0) & (tt <= TWO_THETA_MAX) & np.isfinite(intensity) & (intensity > 0)
    tt, inten = tt[keep], intensity[keep]
    if len(tt) == 0:
        raise SystemExit("No peaks found inside the model's 0-90 degree (Cu Ka) window.")
    order = np.lexsort((tt, -inten))[:MAX_PEAKS]  # upstream's tie-break: intensity, then angle
    tt, inten = tt[order], inten[order]
    inten = 100.0 * inten / inten.max()

    path = os.path.join(workdir, "peaks_cuka_top20.csv")
    write_peaks_csv(path, tt, inten)
    return path


def check_spacegroup(symbol: str, upstream: str) -> None:
    """Upstream inserts the symbol verbatim; unknown spellings tokenize as <unk> without error."""
    with open(os.path.join(upstream, "_utils", "HF-cif-tokenizer", "spacegroups.txt"), encoding="utf-8") as fin:
        known = {line.strip() for line in fin if line.strip()}
    if symbol not in known:
        raise SystemExit(f"Unknown space-group spelling {symbol!r}; use the tokenizer form, e.g. P4_2/mnm or Fm-3m")


def main() -> None:
    p = argparse.ArgumentParser(description="CrystaLLM-pi PXRD -> crystal structure inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pattern", help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    src.add_argument("--peaks", help="Pre-picked peaks CSV with header '2theta,intensity'")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--strip-ka2", choices=["auto", "on", "off"], default="auto",
                   help="Merge Cu Ka2 satellites into Ka1 peaks (auto: on only for data declared at averaged Cu Ka, 1.5418 A)")
    p.add_argument("--composition", required=True, help="Reduced formula, e.g. TiO2 (required by this model)")
    p.add_argument("--z", type=int, help="Formula units per cell; every Z in 1,2,3,4,6 is tried if omitted")
    p.add_argument("--spacegroup", help="Optional Hermann-Mauguin symbol, e.g. P4_2/mnm")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--model", choices=sorted(MODELS), default="mattergen")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--upstream", default=os.environ.get("CRYSTALLM_PI_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--weights", default=os.environ.get("CRYSTALLM_PI_WEIGHTS", os.path.join(HERE, ".weights")))
    args = p.parse_args()

    if args.z is not None and args.z < 1:
        p.error("--z must be >= 1")
    if args.spacegroup:
        check_spacegroup(args.spacegroup, args.upstream)

    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    peaks_csv = prepare_peaks(args, out)

    # Upstream's --search_zs keeps only the first valid structure per formula, so when Z is
    # unknown we prompt every candidate Z explicitly and get --n-samples structures for each.
    z_values = [args.z] if args.z is not None else SEARCH_ZS
    n_prompts = len(z_values)
    # Load the snapshot setup.sh downloaded at a pinned revision. Upstream looks models up in
    # its registry by path, so register the local directory under the hub entry's metadata.
    hub_id = MODELS[args.model]
    local_dir = os.path.join(args.weights, hub_id.split("/")[1])
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        raise SystemExit(f"Checkpoint not found in {local_dir}; run setup.sh first")
    with open(os.path.join(args.upstream, "_utils", "model_registry.json"), encoding="utf-8") as fin:
        registry_entry = json.load(fin)[hub_id]
    registry_path = os.path.join(out, "model_registry.json")
    with open(registry_path, "w", encoding="utf-8") as fout:
        json.dump({local_dir: registry_entry}, fout, indent=2)

    cmd = [
        sys.executable,
        "_load_and_generate.py",
        "--hf_model_path", local_dir,
        "--model_registry", registry_path,
        "--reduced_formula_list", ",".join([args.composition] * n_prompts),
        "--z_list", ",".join(str(z) for z in z_values),
        "--xrd_files", *([peaks_csv] * n_prompts),
        "--xrd_wavelength", str(CU_KA1),
        "--level", "level_4" if args.spacegroup else "level_2",
        "--temperature", str(args.temperature),
        "--num_return_sequences", str(args.n_samples),
        "--max_return_attempts", "3",
        "--target_valid_cifs", str(args.n_samples),
        "--num_workers_gpu", "1",
        "--output_cif_dir", cand_dir,
    ]
    if args.spacegroup:
        cmd += ["--spacegroups", ",".join([args.spacegroup] * n_prompts)]

    env = dict(os.environ)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    start = time.time()
    with open(os.path.join(out, "upstream.log"), "w") as log:
        proc = subprocess.run(cmd, cwd=args.upstream, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"Upstream generation failed; see {out}/upstream.log")

    results = {
        "model": "crystallm_pi",
        "checkpoint": MODELS[args.model],
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream", "weights")},
        "peaks_csv": os.path.relpath(peaks_csv, out),
        "runtime_s": round(time.time() - start, 1),
        "candidates": [os.path.relpath(f, out) for f in sorted(glob.glob(os.path.join(cand_dir, "*.cif")))],
    }
    with open(os.path.join(out, "results.json"), "w") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(results['candidates'])} candidate CIFs to {cand_dir}")


if __name__ == "__main__":
    main()
