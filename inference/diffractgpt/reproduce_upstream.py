# Checks this wrapper against DiffractGPT's released test set (knc6/diffractgpt_jarvis_dft):
#  1. model and generation: greedy output on the authors' own prompts, scored against the answers;
#  2. prompt builder: run.build_prompt on pymatgen peaks of each answer structure, compared with the
#     authors' peak text (pymatgen stands in for JARVIS's XRD simulator, so small differences remain).
# Usage: python reproduce_upstream.py /path/to/test.parquet [--records 30]

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run as wrapper  # noqa: E402


def peaks_of(text: str) -> list[tuple[float, float]]:
    return [(float(t), float(i)) for t, i in re.findall(r"([0-9.]+)°\(([0-9.]+)\)", text)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test_parquet")
    p.add_argument("--records", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    import pandas as pd
    import torch
    from peft import PeftModel
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = pd.read_parquet(args.test_parquet).sample(args.records, random_state=args.seed)
    weights = os.path.join(HERE, ".weights")
    adapter = os.path.join(weights, "diffractgpt_mistral_chemical_formula")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    from transformers import BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(os.path.join(weights, "mistral-7b-bnb-4bit"),
                                                 device_map="cuda", torch_dtype=dtype, quantization_config=quant)
    model = PeftModel.from_pretrained(model, adapter).eval()
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    matched = parsed = 0
    peak_recall, abc_err, rms, formula_ok = [], [], [], []
    for _, row in rows.iterrows():
        answer = wrapper.text_to_structure("\n" + row["output"])
        # 1. the authors' own prompt through this wrapper's loading and decoding
        text = wrapper.ALPACA_PROMPT.format(wrapper.INSTRUCTION, row["input"], "")
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(out)[0].split("# Output:")[1].strip("</s>")
        try:
            pred = wrapper.text_to_structure(response)
            parsed += 1
            abc_err.append(np.abs(np.array(pred.lattice.abc) - np.array(answer.lattice.abc)))
            dist = matcher.get_rms_dist(pred, answer)
            if dist is not None:
                matched += 1
                rms.append(dist[0] * (answer.volume / len(answer)) ** (1 / 3))  # normalized -> Angstrom
        except Exception:
            pass
        # 2. this wrapper's prompt builder on simulated peaks of the answer
        pattern = XRDCalculator("CuKa").get_pattern(answer, two_theta_range=(0, 90))
        prompt_ours, ours = wrapper.build_prompt(np.array(pattern.x), np.array(pattern.y), answer.composition.formula)
        formula_ok.append(prompt_ours.split("\n")[0] == row["input"].split("\n")[0])
        theirs = peaks_of(row["input"])
        mine = peaks_of(ours)
        hits = sum(any(abs(t - u) <= 0.15 for u, _ in mine) for t, _ in theirs)
        peak_recall.append(hits / max(1, len(theirs)))
    mae = np.mean(abc_err, axis=0) if abc_err else [np.nan] * 3
    print(f"records={args.records} parsed={parsed} match_greedy={matched / args.records:.3f} "
          f"lattice_MAE_abc={mae[0]:.2f},{mae[1]:.2f},{mae[2]:.2f} A "
          f"rms_matched={np.mean(rms) if rms else float('nan'):.3f} A prompt_peak_recall={np.mean(peak_recall):.3f} "
          f"prompt_formula_exact={np.mean(formula_ok):.3f}")


if __name__ == "__main__":
    main()
