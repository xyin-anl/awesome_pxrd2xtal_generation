# Approximates the paper's simulated MP-20 test evaluation (Table 1: MP-20-trained Bragglite,
# Top-1/Top-20 match rates with the strict matcher) on the public MP-20 test split, with
# upstream's own code for every step: Bragg simulation (src/pxrd/bragg_simulator.py) and
# peaks_128x4 features (src/pxrd/peaks.py:xrdiff_feature_vector) at 1.54184 A over 5-90 deg, the
# evaluation sampler (src/pxrd/eval_task_runtime.py:_prediction_rows, EMA weights, K=20, seed 777,
# stateless per-candidate seeds), and the scorer (src/pxrd/eval_metrics.py:score_prediction_row,
# ltol/stol/angle_tol 0.2/0.3/5, matcher only). Upstream's frozen MP-20 token, structure, and
# feature caches are not public, so this is not upstream's exact data path: references are the
# CDVAE CIFs reduced to primitive Niggli cells here, and features are recomputed from them.
# Targets follow upstream's order (torch.randperm with the evaluation seed) and get upstream's
# per-candidate seeds for their row index; results are appended per target, so an interrupted run
# continues with --resume (refused if the configuration changed).
#
# With --via-wrapper the same targets instead go through run.py as experimental patterns: each
# reference is rendered as a dense pseudo-Voigt profile (upstream's pseudo_voigt_profile, 0.02 deg
# step) and written as a two-column file, which tests the wrapper's experimental path end to end.
# run.py seeds each pattern as a single target, so these samples are independent of a
# simulated-feature run on the same targets (--compare-with records that run's counts).
#
# Usage (bragglite environment):
#   python reproduce_upstream.py --mp20-csv test.csv --training-set mp20 --mode composition \
#       --num-targets 1000 --out runs/repro_mp20_stoich [--resume]
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from inference.bragglite.run import CHECKPOINTS, DEFAULT_WEIGHTS, TASK  # noqa: E402

CDVAE_COMMIT = "f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea"
MP20_TEST_URL = f"https://raw.githubusercontent.com/txie-93/cdvae/{CDVAE_COMMIT}/data/mp_20/test.csv"
MP20_TEST_SHA256 = "dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a"
PAPER_TABLE1 = {  # MP-20-trained Bragglite on simulated MP-20 test, Top-1 / Top-20 (%)
    ("mp20", "composition"): (70.85, 82.59),
    ("mp20", "element-set"): (47.48, 71.51),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(cif: str):
    from pymatgen.core import Structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Structure.from_str(cif, fmt="cif")
    # Upstream's primitive_niggli_v1 cache builder is not public; 0.1 A is the primitive-cell tolerance
    # upstream's own opXRD loader uses (experimental_pxrd.py). The default 0.25 A merges near-translations
    # and changed the atom count of 27 of the first 2000 targets.
    return s.get_primitive_structure(tolerance=0.1).get_reduced_structure()


def make_item(material_id: str, cif: str, nmax: int):
    import torch

    from src.data.mp20_tokens import pack_structure_tokens
    from src.pxrd.bragg_simulator import fast_xrd_pattern
    from src.pxrd.peaks import xrdiff_feature_vector

    s = canonical(cif)
    item = pack_structure_tokens(s.atomic_numbers, s.frac_coords, s.lattice.abc, s.lattice.angles,
                                 nmax=nmax, num_atoms=len(s), mp_id=material_id)
    tt, inten = fast_xrd_pattern(s, wavelength=1.54184, two_theta_range=(5.0, 90.0))
    feats = xrdiff_feature_vector(tt, inten, "peaks_128x4", peak_count=128)
    item["pxrd_d"] = torch.from_numpy(feats)
    item["pxrd_i"] = torch.zeros_like(item["pxrd_d"])
    item["pxrd_peak_pad_mask"] = torch.zeros_like(item["pxrd_d"], dtype=torch.bool)
    return item, (tt, inten)


def score_row(row: dict) -> dict:
    from src.pxrd.eval_metrics import score_prediction_row

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scored = score_prediction_row(row, topk=(1, 20), matcher_profile="strict", match_gate="matcher_only",
                                      pdd_k=100, include_formula=False, match_only=True)
    return {"sample_id": row["sample_id"], "match@1": bool(scored["topk"]["1"]["matched"]),
            "match@20": bool(scored["topk"]["20"]["matched"]), "n_candidates": len(row["candidates"]),
            "failed": bool(row.get("failed", False))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mp20-csv", required=True, help="CDVAE data/mp_20/test.csv (sha256 checked)")
    p.add_argument("--training-set", choices=["mp20", "pxrd_bench"], default="mp20")
    p.add_argument("--mode", choices=["composition", "element-set"], default="composition")
    p.add_argument("--num-targets", type=int, default=1000)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--block", type=int, default=16, help="Targets sampled together (x n-samples candidates)")
    p.add_argument("--workers", type=int, default=4, help="Scoring processes")
    p.add_argument("--via-wrapper", action="store_true", help="Run each target through run.py as an .xy pattern")
    p.add_argument("--wrapper-python", default=sys.executable)
    p.add_argument("--checkpoint-dir", default=DEFAULT_WEIGHTS)
    p.add_argument("--upstream", default=os.environ.get("BRAGGLITE_UPSTREAM", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compare-with", help="Another run's output directory; its rates on the same targets go into the summary")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.abspath(args.upstream))
    if sha256(args.mp20_csv) != MP20_TEST_SHA256:
        raise SystemExit(f"{args.mp20_csv} is not the CDVAE MP-20 test split ({MP20_TEST_URL})")
    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "scored.jsonl")

    import torch

    from src.data.mp20_tokens import VZ
    from src.eval_util.checkpoint_runtime import (build_model_from_checkpoint, checkpoint_model_args,
                                                  checkpoint_type_encoding_name, load_checkpoint,
                                                  maybe_apply_checkpoint_ema)
    from src.models.type_encoding import build_type_encoding
    from src.pxrd.eval_task_runtime import _prediction_rows, _sampler_config
    from src.pxrd.peaks import pseudo_voigt_profile

    with open(args.mp20_csv, newline="", encoding="utf-8") as fin:
        records = list(csv.DictReader(fin))
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(args.seed)).tolist()
    targets = order[: args.num_targets]
    config = {k: getattr(args, k) for k in ("training_set", "mode", "n_samples", "seed", "via_wrapper")}
    config["data_sha256"] = MP20_TEST_SHA256
    config_path = os.path.join(args.out, "config.json")
    if args.resume and os.path.exists(rows_path):
        with open(config_path, encoding="utf-8") as fin:
            if json.load(fin) != config:
                raise SystemExit(f"{args.out} holds rows from a different configuration; use a new --out")
    with open(config_path, "w", encoding="utf-8") as fout:
        json.dump(config, fout, indent=2)
    done = {}
    if args.resume and os.path.exists(rows_path):
        with open(rows_path, encoding="utf-8") as fin:
            for line in fin:
                r = json.loads(line)
                done[r["sample_id"]] = r
    elif os.path.exists(rows_path):
        os.remove(rows_path)
    todo = [i for i in targets if records[i]["material_id"] not in done]
    print(f"{len(targets)} targets, {len(done)} done, {len(todo)} to run", flush=True)

    ckpt = os.path.join(args.checkpoint_dir, CHECKPOINTS[(args.training_set, args.mode)])
    task = TASK[args.mode]
    device = torch.device("cuda")
    start = time.time()
    if not args.via_wrapper:
        checkpoint = load_checkpoint(ckpt, map_location=device)
        model_args = checkpoint_model_args(checkpoint)
        nmax = int(model_args["nmax"])
        model, _ = build_model_from_checkpoint(checkpoint, device=device)
        maybe_apply_checkpoint_ema(model, checkpoint, "ema")
        model.eval()
        type_encoding = build_type_encoding(checkpoint_type_encoding_name(checkpoint, model_args), vz=VZ)
        sample_args = SimpleNamespace(seed=args.seed, num_candidates=args.n_samples, batch_candidates=True,
                                      candidate_chunk_size=320, chunk_size=256, compile=False, bf16=True,
                                      sampling_backend="edm", rng_mode="stateless_batch")
        sampler = _sampler_config(sample_args, model_args)
        os.environ["BRAGGLITE_FAST_SAMPLER"] = "1"
    else:
        nmax = 20 if args.training_set == "mp20" else 52

    # spawn, not fork: the parent holds a CUDA context.
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
    pending = []
    for b in range(0, len(todo), args.block):
        block = todo[b: b + args.block]
        if not args.via_wrapper:
            # Keyed by the row index in the test split, so each target gets upstream's per-candidate
            # seeds stable_seed(domain, task, seed, dataset_index, rank) for its own index.
            items = {i: make_item(records[i]["material_id"], records[i]["cif"], nmax)[0] for i in block}
            rows = list(_prediction_rows(task=task, args=sample_args, model=model, type_encoding=type_encoding,
                                         dataset=items, indices=list(block), model_args=model_args,
                                         sampler=sampler, device=device, metadata={}))
        else:
            from src.pxrd.eval_metrics import structure_to_record
            from pymatgen.core import Structure

            rows = []
            for i in block:
                s = canonical(records[i]["cif"])
                _, (tt, inten) = make_item(records[i]["material_id"], records[i]["cif"], nmax)
                grid = np.arange(5.0, 90.0, 0.02)
                prof = pseudo_voigt_profile(tt, inten, two_theta_min=5.0, two_theta_max=90.0, step=0.02)
                with tempfile.TemporaryDirectory(dir=args.out) as tmp:
                    xy = os.path.join(tmp, "pattern.xy")
                    np.savetxt(xy, np.column_stack([grid, 1000 * prof]), fmt="%.4f")
                    out = os.path.join(tmp, "out")
                    cmd = [args.wrapper_python, os.path.join(HERE, "run.py"), "--pattern", xy, "--wavelength", "1.54184",
                           "--cu-positions", "declared", "--mode", args.mode, "--composition", s.composition.reduced_formula,
                           "--n-atoms", str(len(s)), "--training-set", args.training_set, "--n-samples", str(args.n_samples),
                           "--seed", str(args.seed), "--out", out, "--checkpoint-dir", args.checkpoint_dir,
                           "--upstream", args.upstream]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    cands = []
                    failed = proc.returncode != 0
                    if not failed:
                        res = json.load(open(os.path.join(out, "results.json"), encoding="utf-8"))
                        for f in res["candidates"]:
                            cands.append(structure_to_record(Structure.from_file(os.path.join(out, f))))
                    else:
                        print(proc.stderr[-500:], flush=True)
                rows.append({"sample_id": records[i]["material_id"], "reference": structure_to_record(s),
                             "candidates": cands, "failed": failed})
        pending += [pool.submit(score_row, row) for row in rows]
        finished = [f for f in pending if f.done()]
        pending = [f for f in pending if not f.done()]
        with open(rows_path, "a", encoding="utf-8") as fout:
            for f in finished:
                fout.write(json.dumps(f.result()) + "\n")
        n_done = len(done) + b + len(block)
        rate = (b + len(block)) / max(time.time() - start, 1e-9)
        print(f"{n_done}/{len(targets)} sampled, ETA {(len(todo) - b - len(block)) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    with open(rows_path, "a", encoding="utf-8") as fout:
        for f in pending:
            fout.write(json.dumps(f.result()) + "\n")
    pool.shutdown()

    with open(rows_path, encoding="utf-8") as fin:
        scored = {r["sample_id"]: r for r in map(json.loads, fin)}
    ids = [records[i]["material_id"] for i in targets]
    n = len(ids)
    top1 = sum(scored[i]["match@1"] for i in ids) / n
    top20 = sum(scored[i]["match@20"] for i in ids) / n
    compare = None
    if args.compare_with:
        with open(os.path.join(args.compare_with, "scored.jsonl"), encoding="utf-8") as fin:
            other = {r["sample_id"]: r for r in map(json.loads, fin)}
        with open(os.path.join(args.compare_with, "config.json"), encoding="utf-8") as fin:
            other_config = json.load(fin)
        shared = [i for i in ids if i in other]
        compare = {"config": other_config, "targets": len(shared),
                   "matched@1": sum(other[i]["match@1"] for i in shared),
                   "matched@20": sum(other[i]["match@20"] for i in shared),
                   "this_run_matched@1": sum(scored[i]["match@1"] for i in shared),
                   "this_run_matched@20": sum(scored[i]["match@20"] for i in shared)}
    summary = {
        "checkpoint": os.path.basename(ckpt),
        "mode": args.mode,
        "path": "run.py on rendered profiles" if args.via_wrapper else "upstream simulated peaks_128x4",
        "data": {"url": MP20_TEST_URL, "sha256": MP20_TEST_SHA256, "rows": len(records)},
        "targets": n,
        "failed_runs": sum(scored[i].get("failed", False) for i in ids),
        "candidates_per_target": sorted({scored[i]["n_candidates"] for i in ids}),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "match_rate@1": round(100 * top1, 2),
        "match_rate@20": round(100 * top20, 2),
        "binomial_se@1": round(100 * (top1 * (1 - top1) / n) ** 0.5, 2),
        "binomial_se@20": round(100 * (top20 * (1 - top20) / n) ** 0.5, 2),
        "paper_table1": PAPER_TABLE1.get((args.training_set, args.mode)),
        "runtime_min": round((time.time() - start) / 60, 1),
        "same_targets_other_run": compare,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
