# PXRD inference script for DiffractGPT (https://github.com/atomgptlab/atomgpt)
# Weights: LoRA adapter knc6/diffractgpt_mistral_chemical_formula on unsloth/mistral-7b-bnb-4bit
# 1. Run setup.sh once (environment and weights at pinned Hugging Face revisions)
# 2. Run: python run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --out results/
# The released adapter (re-uploaded 2025-10-24) was trained on peak-list prompts built by
# atomgpt/scripts/diffractgpt/dataset_atomgpt_spectra2.py:make_diffractgpt_prompt. This script picks
# peaks from the measured pattern (inference/_common/pxrd_io.py) and rebuilds the prompt with
# exactly those steps.
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

from inference._common.pxrd_io import (  # noqa: E402
    CU_KA,
    convert_two_theta,
    load_pattern,
    parse_wavelength,
    pick_peaks,
)

INSTRUCTION = "Below is a description of a material."
ALPACA_PROMPT = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"  # atomgpt TrainingPropConfig
TWO_THETA_MAX = 90.0  # make_diffractgpt_prompt: thetas=[0, 90]
NUM_PEAKS = 20


def build_prompt(two_theta: np.ndarray, intensity: np.ndarray, formula: str) -> tuple[str, str]:
    """make_diffractgpt_prompt applied to a peak list at averaged Cu Ka (1.54184 A, JARVIS's XRD
    default): Gaussian sticks (sigma 0.1 deg) on a 0.1 deg grid, find_peaks, top 20 by height,
    sorted by angle."""
    from scipy.signal import find_peaks

    intensity = intensity / intensity.max()
    x_new = np.arange(0, 90, 0.1)
    y_new = np.zeros_like(x_new, dtype=np.float64)
    for x0, amp in zip(two_theta, intensity):
        y_new += amp * np.exp(-0.5 * ((x_new - x0) / 0.1) ** 2)
    y_new /= y_new.max()
    peaks, props = find_peaks(y_new, height=0.01, distance=1, prominence=0.05)
    top = peaks[np.argsort(props["peak_heights"])[::-1][:NUM_PEAKS]]
    top = top[np.argsort(x_new[top])]
    peak_text = ", ".join(f"{round(x_new[p], 2)}°({round(y_new[p], 2)})" for p in top)
    return training_prompt(formula, peak_text), peak_text


def jarvis_formula(formula: str) -> str:
    """Training prompts use JARVIS's reduced formula (gcd reduction, electronegativity order, no
    polyanion grouping); pymatgen's differs for 703 of the 3,799 released test prompts."""
    from jarvis.core.composition import Composition as JarvisComposition
    from pymatgen.core import Composition

    return JarvisComposition(Composition(formula).get_el_amt_dict()).reduced_formula


def training_prompt(formula: str, peak_text: str) -> str:
    reduced = jarvis_formula(formula)
    return (
        f"The chemical formula is: {reduced}.\n"
        f"The XRD pattern shows main peaks at: {peak_text}.\n"
        "Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )


def upstream_peak_text(two_theta: np.ndarray, intensity: np.ndarray) -> str:
    """atomgpt inverse_models/utils.py:load_exp_file peak selection on a measured profile (angles
    already converted to Cu Ka): height-normalized, find_peaks(height=0.05, prominence=0.02,
    >= 0.5 deg apart), the 20 highest peaks sorted by angle."""
    from scipy.signal import find_peaks

    y = intensity / intensity.max()
    step = two_theta[1] - two_theta[0]
    distance = max(1, int(0.5 / step))
    peaks, props = find_peaks(y, height=0.05, prominence=0.02, distance=distance)
    top = peaks[np.argsort(props["peak_heights"])[::-1][:NUM_PEAKS]]
    top = top[np.argsort(two_theta[top])]
    return ", ".join(f"{round(two_theta[p], 2)}°({round(y[p], 2)})" for p in top)


def text_to_structure(response: str):
    """atomgpt inverse_models/utils.py:text2atoms, returning a pymatgen Structure."""
    from pymatgen.core import Lattice, Structure

    lines = response.strip("</s>").split("\n")
    lengths = np.array(lines[1].split(), dtype=float)
    angles = np.array(lines[2].split(), dtype=float)
    while lines and not lines[-1].strip():  # trailing blank lines only
        lines.pop()
    species, coords = [], []
    for line in lines[3:]:
        # Like upstream, any malformed atom row invalidates the whole structure.
        tokens = line.split()
        species.append(tokens[0])
        coords.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    return Structure(Lattice.from_parameters(*lengths, *angles), species, coords)


def main() -> None:
    p = argparse.ArgumentParser(description="DiffractGPT PXRD -> crystal structure inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pattern", help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    src.add_argument("--peaks", help="Pre-picked peaks CSV with header '2theta,intensity' (needs --wavelength)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--composition", required=True, help="Chemical formula; the prompt uses JARVIS's reduced formula")
    p.add_argument("--peak-method", choices=["harness", "upstream"], default="harness",
                   help="harness: shared peak picker + the training prompt generator; "
                        "upstream: atomgpt load_exp_file peak selection on the profile")
    p.add_argument("--strip-ka2", choices=["auto", "on", "off"], default="auto",
                   help="harness method: merge Cu Ka2 satellites (auto: on only for data declared at averaged Cu Ka, 1.5418 A)")
    p.add_argument("--n-samples", type=int, default=1,
                   help="1 = upstream's greedy decoding; >1 = that many sampled structures (not upstream)")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature when --n-samples > 1")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", default=os.environ.get("DIFFRACTGPT_WEIGHTS", os.path.join(HERE, ".weights")))
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    if args.n_samples < 1:
        p.error("--n-samples must be >= 1")

    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)

    if args.peak_method == "upstream":
        if args.peaks:
            raise SystemExit("--peak-method upstream needs a profile (--pattern)")
        pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
        tt = convert_two_theta(pattern.two_theta, pattern.wavelength, CU_KA)
        keep = np.isfinite(tt) & (tt < TWO_THETA_MAX)
        grid = np.arange(tt[keep].min(), tt[keep].max(), np.median(np.diff(tt[keep])))
        profile = np.clip(np.interp(grid, tt[keep], pattern.intensity[keep]), 0, None)
        peak_text = upstream_peak_text(grid, profile)
        if not peak_text:
            raise SystemExit("No peaks found by the upstream peak selection")
        prompt = training_prompt(args.composition, peak_text)
    else:
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
        keep = np.isfinite(tt) & (tt > 0) & (tt < TWO_THETA_MAX) & np.isfinite(intensity) & (intensity > 0)
        if not keep.any():
            raise SystemExit("No peaks inside the model's 0-90 degree (Cu Ka) window")
        prompt, peak_text = build_prompt(tt[keep], intensity[keep], args.composition)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    start = time.time()
    adapter = os.path.join(args.weights, "diffractgpt_mistral_chemical_formula")
    base = os.path.join(args.weights, "mistral-7b-bnb-4bit")
    if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
        raise SystemExit(f"Weights not found in {args.weights}; run setup.sh first")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    # As upstream's loader: bf16 where supported (else fp16), with a matching 4-bit compute dtype.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(base, device_map="cuda", torch_dtype=dtype, quantization_config=quant)
    model = PeftModel.from_pretrained(model, adapter).eval()

    text = ALPACA_PROMPT.format(INSTRUCTION, prompt, "")
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    gen = dict(max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    if args.n_samples == 1:
        gen.update(do_sample=False)
    else:
        gen.update(do_sample=True, temperature=args.temperature, num_return_sequences=args.n_samples)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen)
    responses = tokenizer.batch_decode(outputs)

    candidates, details = [], []
    for k, full in enumerate(responses, start=1):
        response = full.split("# Output:")[1].strip("</s>")
        entry = {"response": response}
        try:
            s = text_to_structure(response)
            path = os.path.join(cand_dir, f"candidate_{k:03d}.cif")
            s.to(filename=path)
            entry["file"] = os.path.relpath(path, out)
            candidates.append(entry["file"])
        except Exception as exc:  # unparsable generation
            entry["error"] = str(exc)[:200]
        details.append(entry)

    results = {
        "model": "diffractgpt",
        "checkpoint": "knc6/diffractgpt_mistral_chemical_formula",
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "weights")},
        "prompt": prompt,
        "peak_text": peak_text,
        "decoding": "greedy" if args.n_samples == 1 else f"sampling, temperature {args.temperature}",
        "runtime_s": round(time.time() - start, 1),
        "candidates": candidates,
        "candidate_details": details,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"Wrote {len(candidates)} candidate CIFs to {cand_dir} ({len(responses)} generations)")


if __name__ == "__main__":
    main()
