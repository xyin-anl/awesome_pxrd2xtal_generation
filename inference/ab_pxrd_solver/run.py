# PXRD inference script for Ab-PXRD-Solver (https://github.com/MaterSim/Ab-PXRD-Solver)
# Models (peak detector, space-group predictor, Roost density ensemble) ship with upstream; MACE
# downloads its foundation model on first use.
# 1. Run setup.sh once (environment with GSAS-II built from source, upstream at the pinned commit)
# 2. Run: python run.py --pattern scan.xy --wavelength CuKa --composition PrYMg2 --spacegroup P4/mmm --out results/
# Ab-PXRD-Solver indexes the pattern, enumerates Wyckoff assignments, relaxes trial structures with
# MACE, and refines with GSAS-II, stopping at the first solution that fits well. This script puts
# the pattern in the form upstream's examples use (averaged Cu Ka, 10-80 deg 2theta on a 0.02 deg
# grid, maximum 100) and runs upstream's own PXRD_solve.py.
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inference._common.pxrd_io import CU_KA, CU_KA1, convert_two_theta, load_pattern  # noqa: E402

GRID = np.round(np.arange(10.0, 80.0 + 1e-9, 0.02), 2)  # upstream examples and GSAS-II simulation range


def prepare_csv(args, path: str) -> list[float]:
    pattern = load_pattern(args.pattern, args.wavelength, args.x_unit)
    # The solver refines against a fixed Cu Ka instrument model (INST_XRY.PRM, 1.54184 A), so only
    # Cu data are accepted; moving the angles of Mo or synchrotron data would not make their
    # intensities and peak shapes comparable.
    if min(abs(pattern.wavelength - CU_KA), abs(pattern.wavelength - CU_KA1)) > 0.003:
        raise SystemExit(f"Ab-PXRD-Solver models a Cu Ka laboratory instrument; got wavelength {pattern.wavelength:.5f} A")
    tt = convert_two_theta(pattern.two_theta, pattern.wavelength, CU_KA)
    ok = np.isfinite(tt)
    tt, y = tt[ok], pattern.intensity[ok]
    covered = [round(float(tt.min()), 2), round(float(tt.max()), 2)]
    if tt.max() <= GRID[0] or tt.min() >= GRID[-1]:
        raise SystemExit(f"The scan covers {covered[0]}-{covered[1]} deg at Cu Ka, outside the solver's 10-80 deg range")
    # Upstream picks background handling and peak thresholds from min(intensity), so unmeasured
    # parts of 10-80 deg are filled with the measured baseline (1st percentile), not with zeros.
    baseline = float(np.percentile(y, 1))
    if covered[0] > GRID[0] + 0.05 or covered[1] < GRID[-1] - 0.05:
        print(f"Warning: the scan covers {covered[0]}-{covered[1]} deg 2theta (Cu Ka) of the 10-80 deg range; "
              "missing ranges are filled with the measured baseline.")
    profile = np.interp(GRID, tt, y, left=baseline, right=baseline)
    profile = 100.0 * profile / profile.max()
    with open(path, "w", encoding="utf-8", newline="") as fout:
        fout.write("# 2Theta,Intensity\n")
        for x, v in zip(GRID, profile):
            fout.write(f"{x:.18e},{v:.18e}\n")
    return covered


def main() -> None:
    p = argparse.ArgumentParser(description="Ab-PXRD-Solver PXRD -> crystal structure")
    p.add_argument("--pattern", required=True, help="Measured pattern (.xy/.xye/.dat/.csv, or pdCIF)")
    p.add_argument("--wavelength", help="Angstrom or name (CuKa, MoKa, ...); read from pdCIF if omitted")
    p.add_argument("--x-unit", choices=["2theta", "q"], default="2theta", help="Unit of the pattern's first column")
    p.add_argument("--composition", required=True, help="Chemical formula, e.g. PrYMg2 (Z is searched by the solver)")
    p.add_argument("--spacegroup", help="Hermann-Mauguin symbol or number; otherwise the solver infers it (--infer-spg)")
    p.add_argument("--timeout", type=int, default=3600, help="Wall-time limit in seconds")
    p.add_argument("--memory-limit", type=float, default=12.0,
                   help="Memory cap in GB for the solver and its workers (via systemd-run when available; 0 = none). "
                        "Some structure searches grow past 25 GB; without a cap the kernel may kill unrelated processes")
    p.add_argument("--upstream", default=os.environ.get("AB_PXRD_SOLVER_DIR", os.path.join(HERE, ".upstream")))
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--n-samples", type=int, default=1, help="Ignored: the solver returns its single best solution")
    p.add_argument("--z", type=int, help="Ignored: the solver chooses Z itself (capped by max_Z 24, 20 atoms per "
                                         "primitive cell, and its predicted density range)")
    args = p.parse_args()

    from pymatgen.core import Composition
    from pymatgen.symmetry.groups import SpaceGroup

    if any(not float(n).is_integer() for n in Composition(args.composition).reduced_composition.values()):
        raise SystemExit("Upstream parses integer element counts only; give a stoichiometric formula")
    # Plain element-count string (no polyanion parentheses), as in upstream's example file names.
    formula = "".join(f"{el}{'' if n == 1 else int(n) if float(n).is_integer() else n}"
                      for el, n in Composition(args.composition).reduced_composition.items())
    spg = None
    if args.spacegroup:
        spg = int(args.spacegroup) if args.spacegroup.isdigit() else SpaceGroup(args.spacegroup).int_number

    out = os.path.abspath(args.out)
    cand_dir = os.path.join(out, "candidates")
    shutil.rmtree(cand_dir, ignore_errors=True)
    os.makedirs(cand_dir, exist_ok=True)
    upstream = os.path.abspath(args.upstream)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, f"PXRD_{formula}_{spg or 1}.csv")
        args.pattern = os.path.abspath(args.pattern)
        covered = prepare_csv(args, csv_path)
        shutil.copy(csv_path, os.path.join(out, "input_cuka_10-80.csv"))
        solver_out = os.path.join(tmp, "Results")
        cmd = [sys.executable, os.path.join(HERE, "solve_driver.py"), "--input", csv_path, "--output", solver_out,
               "--formula", formula]
        cmd += ["--spg", str(spg)] if spg else ["--infer-spg"]
        # A cgroup memory cap confines an out-of-memory kill to the solver instead of the whole session.
        scope = None
        if args.memory_limit > 0 and shutil.which("systemd-run"):
            scope = f"ab-pxrd-solver-{os.getpid()}-{int(time.time())}"
            cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={scope}",
                   "-p", f"MemoryMax={int(args.memory_limit * 1024)}M", "-p", "MemorySwapMax=0"] + cmd
        # MACE relaxations run in forked workers, which cannot use CUDA: run on CPU as upstream does.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        start = time.time()
        with open(os.path.join(out, "upstream.log"), "w", encoding="utf-8") as log:
            # Upstream starts persistent MACE and GSAS-II worker processes; run it in its own
            # process group so a timeout can stop all of them.
            proc = subprocess.Popen(cmd, cwd=upstream, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                proc.wait(timeout=args.timeout)
                status = "finished" if proc.returncode == 0 else (
                    f"killed (memory limit {args.memory_limit:g} GB)" if proc.returncode in (137, -9)
                    else f"exit code {proc.returncode}")
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status = f"timed out after {args.timeout} s"
            finally:
                # Spawned MACE/GSAS-II workers can outlive the process group; stopping the scope
                # kills everything left in its cgroup.
                if scope:
                    subprocess.run(["systemctl", "--user", "stop", f"{scope}.scope"], capture_output=True)
        runtime = round(time.time() - start, 1)

        summary = {}
        summary_csv = os.path.join(solver_out, "summary.csv")
        if os.path.exists(summary_csv):
            with open(summary_csv, encoding="utf-8") as fin:
                rows = list(csv.DictReader(fin))
            summary = rows[-1] if rows else {}
        # Only the solution the pipeline selected (written by solve_driver.py), and nothing from a
        # run that did not finish.
        candidates, selected = [], {}
        selected_json = os.path.join(solver_out, "selected.json")
        if status == "finished" and os.path.exists(selected_json):
            with open(selected_json, encoding="utf-8") as fin:
                selected = json.load(fin)
            if selected.get("cif"):
                dest = os.path.join(cand_dir, "candidate_001.cif")
                shutil.copy(selected["cif"], dest)
                candidates.append(os.path.relpath(dest, out))
        if os.path.isdir(os.path.join(solver_out, "logs")):
            shutil.copytree(os.path.join(solver_out, "logs"), os.path.join(out, "solver_logs"), dirs_exist_ok=True)

    results = {
        "model": "ab_pxrd_solver",
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out", "upstream")},
        "solver_formula": formula,
        "solver_spacegroup": spg or "inferred",
        "scan_coverage_cuka_2theta": covered,
        "status": status,
        "solver_summary": summary,
        "selected": selected,
        "runtime_s": runtime,
        "candidates": candidates,
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"{status}: wrote {len(candidates)} candidate CIF(s) to {cand_dir} "
          f"(status {summary.get('Status', 'n/a')}, R2 {summary.get('R2', 'n/a')})")
    if status != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
