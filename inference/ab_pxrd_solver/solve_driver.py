# Single-pattern driver for Ab-PXRD-Solver. Mirrors upstream PXRD_solve.py:run_deterministic,
# then writes the solution the pipeline actually selected (state["best_result"]["xtal"]). Upstream
# writes Results/cifs/Match_<formula>_<spg>.cif after each trial, so that file can belong to a
# trial other than the final selection.
# Usage (run with the upstream checkout as working directory): python solve_driver.py <PXRD_solve args>

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from pxrd_app.cli import build_common_parser, build_run_state  # noqa: E402
from pxrd_app.constants import DEFAULT_STATE as default_state  # noqa: E402
from pxrd_app.core import attach_run_log, detach_run_log, logger, run_pipeline  # noqa: E402
from pxrd_app.runtime import emit_timing_summary, write_results_csv  # noqa: E402


def main() -> None:
    args = build_common_parser("Ab-PXRD-Solver single-pattern driver").parse_args()
    for sub in ("cifs", "logs", "tmp"):
        os.makedirs(os.path.join(args.output, sub), exist_ok=True)
    os.environ["PXRD_TMP_ROOT"] = os.path.join(args.output, "tmp")

    state = build_run_state(default_state, logger, args, args.input)
    handler = attach_run_log(state)
    try:
        state = run_pipeline(state)
    finally:
        detach_run_log(handler)
        emit_timing_summary(logger, state)
        write_results_csv(args.input, state)

    best = state.get("best_result") or {}
    selected = {
        "status": state.get("status"),
        "message": state.get("msg"),
        "spg": state.get("spg"),
        "r2": best.get("r2"),
        "chi2": best.get("chi2"),
        "wr": best.get("wr"),
        "accepted": best.get("accepted"),
        "cif": None,
    }
    if best.get("xtal") is not None:
        path = os.path.join(args.output, "selected.cif")
        best["xtal"].to_file(path)
        selected["cif"] = path
    with open(os.path.join(args.output, "selected.json"), "w", encoding="utf-8") as fout:
        json.dump(selected, fout, indent=2, default=str)


if __name__ == "__main__":
    main()
