# Builds cases.json and ground_truth/*.cif from experimental pdCIFs that contain both a measured
# pattern and the refined structure. Run once when the benchmark set changes.
# Usage: python inference/_benchmark/build_cases.py

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

from pymatgen.io.cif import CifParser, CifWriter  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from utils.parse_cifs import read_experimental_cif  # noqa: E402

# Files whose parsed reference structure is not trustworthy, with the reason.
EXCLUDED = {
    "sh0123Xraysup5.rtv.combined.cif": "utils/parse_cifs.py cannot parse the pattern loop",
    "ck5030Vsup6.rtv.combined.cif": "partial-occupancy Er2Ge3.17 phase parses as a 2-atom ErGe cell",
    "av5088sup4.rtv.combined.cif": "two-phase sample (monoclinic ~57.5 wt% + orthorhombic C4Br4S); one reference cannot score it",
}


def main() -> None:
    warnings.filterwarnings("ignore")
    gt_dir = os.path.join(HERE, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    cases = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "exp_pxrd_data", "pxrdnet", "*.cif"))):
        name = os.path.basename(path)
        if name in EXCLUDED:
            continue
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = read_experimental_cif(filepath=path)
            structure = CifParser.from_str(result[1]).parse_structures(primitive=False)[0]
        conv = SpacegroupAnalyzer(structure, symprec=0.1).get_conventional_standard_structure()
        reduced, z = conv.composition.get_reduced_formula_and_factor()
        sga = SpacegroupAnalyzer(conv, symprec=0.1)
        case_id = name.split(".")[0]
        CifWriter(conv).write_file(os.path.join(gt_dir, f"{case_id}.cif"))
        two_theta = result[6]
        cases.append(
            {
                "id": case_id,
                "pattern": os.path.relpath(path, REPO_ROOT),
                "wavelength": round(float(result[9]), 5),
                "two_theta_range": [round(float(min(two_theta)), 2), round(float(max(two_theta)), 2)],
                "composition": reduced,
                "z": int(z),
                "spacegroup": sga.get_space_group_symbol(),
                "spacegroup_number": sga.get_space_group_number(),
                # conventional standard cell (same setting as z and spacegroup), for models that need the cell
                "cell": ",".join(f"{v:.5f}" for v in list(conv.lattice.abc) + list(conv.lattice.angles)),
                "ground_truth": f"ground_truth/{case_id}.cif",
                "source": "exp_pxrd_data/pxrdnet (IUCr pdCIF, as used by PXRDnet)",
            }
        )
    with open(os.path.join(HERE, "cases.json"), "w", encoding="utf-8") as fout:
        json.dump({"excluded": EXCLUDED, "cases": cases}, fout, indent=2)
        fout.write("\n")
    print(f"Wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
