# Model-agnostic benchmark for the inference scripts in this repository.
# Runs <model>/run.py on every case in cases.json under each input setting declared in the
# model's manifest.yaml, then scores the candidate CIFs against the reference structures.
#
# Usage:
#   python inference/_benchmark/benchmark.py inference/crystallm_pi \
#       --python ~/micromamba/envs/crystallm_pi/bin/python --n-samples 20
#
# --python is the interpreter of the model's environment (run.py is executed with it).
# This script itself needs numpy, pymatgen, and pyyaml.
#
# Settings: every setting in manifest.benchmark.settings lists the side information passed to
# the model (composition, z, spacegroup). With --control, each case is also run with another
# case's pattern ("mismatched pattern"); a model that actually uses the diffraction data should
# score clearly lower there.

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# Matching tolerances used by CDVAE/DiffCSP-style crystal structure prediction benchmarks.
MATCHER_KW = {"stol": 0.5, "angle_tol": 10, "ltol": 0.3}

FLAG_FOR_INPUT = {"composition": "--composition", "z": "--z", "spacegroup": "--spacegroup", "cell": "--cell"}


def failure(output, out_dir):
    # Keep the tail of the output without machine-specific paths; wrappers that record why they
    # stopped (timeout, memory limit) in results.json before exiting non-zero contribute that too.
    tail = output[-2000:].replace(REPO_ROOT, "<repo>").replace(os.path.expanduser("~"), "~")
    res = {"error": tail}
    path = os.path.join(out_dir, "results.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fin:
            status = json.load(fin).get("status")
        if status:
            res["status"] = status
    return res


def build_cmd(model_dir, python, case, pattern_case, inputs, n_samples, out_dir, extra_args):
    cmd = [python, os.path.join(model_dir, "run.py"), "--n-samples", str(n_samples), "--out", out_dir]
    cmd += ["--pattern", os.path.join(REPO_ROOT, pattern_case["pattern"])]
    if not pattern_case["pattern"].endswith(".cif"):  # pdCIF files carry their own wavelength
        cmd += ["--wavelength", str(pattern_case["wavelength"])]
    for name in inputs:
        cmd += [FLAG_FOR_INPUT[name], str(case[name])]
    return cmd + list(extra_args)


def effective_n_samples(extra_args, default):
    # argparse keeps the last value, so an --n-samples in a setting's extra_args wins.
    values = [extra_args[i + 1] for i, arg in enumerate(extra_args[:-1]) if arg == "--n-samples"]
    return int(values[-1]) if values else default


def fingerprint(cmd, model_dir):
    """Identifies a run for --resume: the exact command plus the wrapper and shared code."""
    digest = hashlib.sha256(json.dumps(cmd).encode())
    for path in sorted(glob.glob(os.path.join(model_dir, "*.py")) + glob.glob(os.path.join(REPO_ROOT, "inference", "_common", "*.py"))):
        with open(path, "rb") as fin:
            digest.update(fin.read())
    return digest.hexdigest()


def run_case(cmd, out_dir):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return failure(proc.stderr or proc.stdout, out_dir)
    path = os.path.join(out_dir, "results.json")
    if not os.path.exists(path):
        return failure("run.py exited 0 without writing results.json\n" + (proc.stderr or proc.stdout), out_dir)
    with open(path, encoding="utf-8") as fin:
        return json.load(fin)


def score(out_dir, results, gt_path):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    gt = Structure.from_file(gt_path)
    gt_sg = SpacegroupAnalyzer(gt, symprec=0.1).get_space_group_number()
    matcher = StructureMatcher(**MATCHER_KW)
    rows = []
    for rel in results.get("candidates", []):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                s = Structure.from_file(os.path.join(out_dir, rel))
            sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
            rms = matcher.get_rms_dist(s, gt)
        except Exception:
            rows.append({"file": rel, "valid": False})
            continue
        rows.append(
            {
                "file": rel,
                "valid": True,
                "spacegroup_number": sg,
                "spacegroup_correct": sg == gt_sg,
                "match": rms is not None,
                "rms": None if rms is None else round(float(rms[0]), 4),
            }
        )
    valid = [r for r in rows if r["valid"]]
    matched = [r for r in valid if r["match"]]
    return {
        "n_candidates": len(rows),
        "n_valid": len(valid),
        "match_any": bool(matched),
        # first candidate as returned; meaningful for wrappers that rank their output
        "match_top1": bool(rows and rows[0]["valid"] and rows[0]["match"]),
        # Both fractions use all returned candidates; unreadable CIFs count as wrong.
        "match_fraction": round(len(matched) / len(rows), 3) if rows else 0.0,
        "best_rms": min((r["rms"] for r in matched), default=None),
        "spacegroup_fraction": round(sum(r["spacegroup_correct"] for r in valid) / len(rows), 3) if rows else 0.0,
        "candidates": rows,
    }


def summarize(per_case):
    # Failed runs stay in every denominator and count as unsuccessful.
    ok = [c for c in per_case if "error" not in c]
    n = len(per_case)
    rms = [c["best_rms"] for c in ok if c["best_rms"] is not None]
    return {
        "n_cases": n,
        "n_failed_runs": n - len(ok),
        "match_rate_any": round(sum(c["match_any"] for c in ok) / n, 3) if n else 0.0,
        "match_rate_top1": round(sum(c.get("match_top1", False) for c in ok) / n, 3) if n else 0.0,
        "mean_match_fraction": round(sum(c["match_fraction"] for c in ok) / n, 3) if n else 0.0,
        "mean_spacegroup_fraction": round(sum(c["spacegroup_fraction"] for c in ok) / n, 3) if n else 0.0,
        "mean_candidates": round(sum(c["n_candidates"] for c in ok) / n, 1) if n else 0.0,
        "mean_best_rms": round(float(np.mean(rms)), 4) if rms else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir")
    p.add_argument("--python", required=True, help="Interpreter of the model's environment")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    p.add_argument("--only", nargs="*", help="Run only these case ids")
    p.add_argument("--control", action="store_true", help="Also run the mismatched-pattern control")
    p.add_argument("--resume", action="store_true",
                   help="Reuse completed cases whose command and wrapper code are unchanged")
    p.add_argument("--settings", nargs="*", help="Run only these settings and merge them into an existing report")
    p.add_argument("--work", default=os.path.join(HERE, "runs"), help="Scratch directory for model outputs")
    p.add_argument("--report", help="Where to write the JSON report (default: <model_dir>/benchmark.json)")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    model_dir = os.path.abspath(args.model_dir)
    with open(os.path.join(model_dir, "manifest.yaml"), encoding="utf-8") as fin:
        manifest = yaml.safe_load(fin)
    bench = manifest.get("benchmark", {})
    all_cases = json.load(open(args.cases, encoding="utf-8"))["cases"]
    # Control donors are fixed on the full case list, so --only cannot create self-pairs.
    donor = {c["id"]: all_cases[(i + len(all_cases) // 2) % len(all_cases)] for i, c in enumerate(all_cases)}
    cases = [c for c in all_cases if c["id"] in args.only] if args.only else all_cases
    if args.only and (args.settings or not args.report):
        p.error("--only produces a partial run; write it to a separate --report (and do not merge with --settings)")

    settings = [(s["name"], s["inputs"], s.get("extra_args", []), False) for s in bench["settings"]]
    if args.settings:
        settings = [st for st in settings if st[0] in args.settings]
        if not settings:
            p.error("none of --settings are declared in the manifest")
    if args.control:
        name, inputs, extra, _ = settings[0]
        settings.append((f"{name}__mismatched_pattern", inputs, extra, True))
    out_path = args.report or os.path.join(model_dir, "benchmark.json")
    previous = {}
    if args.settings and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fin:
            old = json.load(fin)
        if old.get("matcher") != MATCHER_KW or [c["id"] for c in all_cases] != old.get("case_ids"):
            p.error(f"{out_path} used a different matcher or case set; rerun all settings instead of merging")
        previous = old.get("settings", {})

    report = {
        "model": manifest["id"],
        "date": datetime.date.today().isoformat(),
        "n_samples": args.n_samples,
        "matcher": MATCHER_KW,
        "case_ids": [c["id"] for c in cases],
        "settings": dict(previous),
    }
    for name, inputs, extra, mismatched in settings:
        per_case = []
        for case in cases:
            pattern_case = donor[case["id"]] if mismatched else case
            out_dir = os.path.join(args.work, manifest["id"], name, case["id"])
            marker = os.path.join(out_dir, ".benchmark_complete")
            cmd = build_cmd(model_dir, args.python, case, pattern_case, inputs, args.n_samples, out_dir, extra)
            key = fingerprint(cmd, model_dir)
            if args.resume and os.path.exists(marker) and open(marker, encoding="utf-8").read() == key:
                with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as fin:
                    res = json.load(fin)
            else:
                for stale in (marker, os.path.join(out_dir, "results.json")):
                    if os.path.exists(stale):
                        os.remove(stale)
                res = run_case(cmd, out_dir)
                if "error" not in res:
                    with open(marker, "w", encoding="utf-8") as fout:
                        fout.write(key)
            gt_path = os.path.join(HERE, case["ground_truth"])
            entry = {"id": case["id"], "pattern_from": pattern_case["id"]}
            entry.update(res if "error" in res else score(out_dir, res, gt_path))
            entry["runtime_s"] = res.get("runtime_s")
            per_case.append(entry)
            status = "ERROR" if "error" in entry else (
                f"match={entry['match_any']!s:5} frac={entry['match_fraction']:.2f} "
                f"sg={entry['spacegroup_fraction']:.2f} n={entry['n_candidates']}"
            )
            print(f"[{name}] {case['id']:26s} {status}", flush=True)
        report["settings"][name] = {
            "inputs": inputs,
            "extra_args": extra,
            "date": datetime.date.today().isoformat(),
            "n_samples": effective_n_samples(extra, args.n_samples),
            "summary": summarize(per_case),
            "cases": per_case,
        }
        print(f"[{name}] SUMMARY {json.dumps(report['settings'][name]['summary'])}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2)
        fout.write("\n")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
