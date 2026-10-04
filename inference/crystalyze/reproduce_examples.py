# Runs this wrapper on the experimental compounds from the Crystalyze paper that ship in
# exp_pxrd_data/crystalyze (pattern .xy, TOPAS .inp with the wavelength, solved .cif) and scores
# the candidates against the published structures.
# Usage: python reproduce_examples.py [--n-samples 20] [--out runs/]

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO_ROOT, "exp_pxrd_data", "crystalyze")


def wavelength_from_inp(path: str) -> float:
    """First emission line with la 1 in the TOPAS input (K-alpha1 for lab sources)."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    match = re.search(r"la\s+1\s+lo\s+([0-9.]+)", text)
    if not match:
        raise ValueError(f"No wavelength in {path}")
    return float(match.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--out", default=os.path.join(HERE, "..", "_benchmark", "runs", "crystalyze_examples"))
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    summary = []
    for case in sorted(os.listdir(DATA)):
        folder = os.path.join(DATA, case)
        xy = glob.glob(os.path.join(folder, "*.xy"))[0]
        lam = wavelength_from_inp(glob.glob(os.path.join(folder, "*.inp"))[0])
        gt = SpacegroupAnalyzer(Structure.from_file(os.path.join(folder, f"{case}.cif")), 0.1).get_conventional_standard_structure()
        sga = SpacegroupAnalyzer(gt, 0.1)
        formula, z = gt.composition.get_reduced_formula_and_factor()
        out = os.path.abspath(os.path.join(args.out, case))
        cmd = [sys.executable, os.path.join(HERE, "run.py"), "--pattern", xy, "--wavelength", str(lam),
               "--composition", formula, "--z", str(int(z)), "--spacegroup", sga.get_space_group_symbol(),
               "--n-samples", str(args.n_samples), "--out", out]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        row = {"case": case, "formula": formula, "z": int(z), "spacegroup": sga.get_space_group_symbol(),
               "atoms": len(gt), "wavelength": lam}
        if proc.returncode != 0:
            row["error"] = (proc.stderr or proc.stdout).strip().splitlines()[-1][:200]
        else:
            files = json.load(open(os.path.join(out, "results.json")))["candidates"]
            matches = [f for f in files if matcher.fit(Structure.from_file(os.path.join(out, f)), gt)]
            row.update({"n_candidates": len(files), "n_match": len(matches)})
        summary.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)


if __name__ == "__main__":
    main()
