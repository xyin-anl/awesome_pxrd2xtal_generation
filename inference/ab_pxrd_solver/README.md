# Ab-PXRD-Solver

Search-based structure solution from PXRD and a chemical formula. It indexes the pattern, enumerates
Wyckoff assignments, relaxes trial structures with the MACE foundation potential, refines the
best trials against the pattern with GSAS-II, and stops at the first solution that fits well.
Upstream: [MaterSim/Ab-PXRD-Solver](https://github.com/MaterSim/Ab-PXRD-Solver) (MIT) ·
Paper: [arXiv 2605.24594](https://arxiv.org/abs/2605.24594)

## Quick start

```bash
./setup.sh                      # conda env "ab_pxrd_solver" (GSAS-II compiled from source), upstream, MACE model
conda activate ab_pxrd_solver
python run.py --pattern scan.xy --wavelength CuKa --composition PrYMg2 --spacegroup P4/mmm --out results/
```

Output: at most one structure, `results/candidates/candidate_001.cif`, written only when the solver
finishes and selects a solution. `results/results.json` records the run status, the solver's
summary row (status, R², χ², Rwp, space group, Wyckoff sites, cell), and the selected solution's
metrics. `results/upstream.log` and `results/solver_logs/` hold upstream's logs, and
`results/input_cuka_10-80.csv` is the exact profile the solver received.

| Option | Notes |
|--------|-------|
| `--pattern` | Measured full profile: `.xy`, `.xye`, `.dat`, `.csv`, or pdCIF. Cu radiation only |
| `--wavelength`, `--x-unit` | Required for column files (`CuKa`, `CuKa1`, or Å), read from pdCIF. Anything further than 0.003 Å from Cu Kα or Kα1 is rejected |
| `--composition` | Required; integer stoichiometry. The solver chooses Z itself |
| `--spacegroup` | Hermann–Mauguin symbol or number; omit to let upstream infer it (`--infer-spg`) |
| `--timeout` | Wall-time limit in seconds (default 3600) |
| `--memory-limit` | Memory cap in GB for the solver and its workers (default 12; 0 disables it); see below |
| `--upstream` | Override `.upstream/`; also settable through `AB_PXRD_SOLVER_DIR` |
| `--out` | Required output directory |

`--n-samples` and `--z` are accepted for interface compatibility and ignored. Runs on CPU; no GPU
is used.

## How the input is prepared

Upstream's examples (`Examples/PXRD_*.csv`) are GSAS-II simulations at averaged Cu Kα (1.54184 Å),
10–80° 2θ on a 0.02° grid, scaled to a maximum of 100; refinement uses a fixed single-wavelength
Cu Kα instrument file (`pxrd_app/tools/INST_XRY.PRM`). [`run.py`](run.py) writes the measured
profile in the same form:

1. Loads the pattern with [`inference/_common/pxrd_io.py`](../_common/pxrd_io.py) and rejects non-Cu
   radiation. Moving the angles of Mo or synchrotron data would not make their intensities and
   peak shapes match the Cu instrument model.
2. Converts Cu Kα1 angles to averaged Cu Kα, interpolates onto the 10–80° grid, and scales to 100.
   Unmeasured parts of the range are filled with the pattern's 1st-percentile intensity rather
   than zero, because upstream chooses its background handling and peak thresholds from the
   minimum intensity; a warning is printed.
3. Writes the formula as a plain element-count string (`KCaCO3F`, not `KCa(CO3)F`) and the space
   group as a number, then runs [`solve_driver.py`](solve_driver.py) inside a capped systemd scope.

`solve_driver.py` mirrors upstream's `PXRD_solve.py:run_deterministic` and additionally writes the
structure the pipeline selected (`state["best_result"]`). Upstream rewrites
`Results/cifs/Match_<formula>_<spg>.cif` after every trial, so that file can belong to a trial
other than the final selection.

MACE relaxations run in forked worker processes, which cannot initialize CUDA, so the wrapper
hides the GPU, as upstream's examples do. The environment pins numpy 1.26.4 (numpy 2 breaks an
upstream `linalg.solve` call), torch 2.6.0, mace-torch 0.3.14, and GSAS-II at commit `6e98f63`.

### Memory

On monoclinic cases, upstream's cell search grows past any cap available on a 32 GB machine within
seconds; without a given space group, most benchmark cases exceed 12 GB within a few minutes. For CdBiClO₂ (P2₁/m), the scope held 4.4 GB after 5 s and 15.2 GB after 10 s, and it hit a
20 GB cap after about 12 s, before any structure was generated. An uncapped run reached about
26 GB, and the kernel's out-of-memory killer then also killed unrelated processes. `run.py`
therefore runs the solver under `systemd-run --user --scope -p MemoryMax=…` (default 12 GB, no
swap). When the cap is reached, only the solver is killed and the status is
`killed (memory limit … GB)`. On timeout, or after any run, the scope is stopped, so no MACE or
GSAS-II worker outlives the run. Without `systemd-run` there is no cap.

## Verification (2026-10-05, CPU, 30 GB RAM, upstream commit `0d67964`)

**Upstream example.** `Examples/PXRD_PrYMg2_123.csv` through `run.py`:

```bash
python run.py --pattern .upstream/Examples/PXRD_PrYMg2_123.csv --wavelength CuKa \
  --composition PrYMg2 --spacegroup 123 --out pryMg2/
```

Result: Success in 72 s, P4/mmm with Wyckoff sites 1d/1c/2g, R² 0.9984, χ² 0.0234, Rwp 3.877. This is
identical to upstream's own `PXRD_solve.py` on the same file (R² 0.9984, χ² 0.0234, Rwp 3.8739,
23 structures and 41 attempts).

**Input check.** Upstream's own pattern similarity (`pxrd_app/tools/XRD.py:Similarity`, with the
simulation settings the search uses) between each reference structure and the profile the wrapper
prepared is 0.986 for both BaTiO₃ and KCaCO₃F; the upstream example's solution scores 0.985 against
its input. The prepared profiles therefore carry the information the search needs; the failures
below happen before structure generation.

**Fresh setup.** `setup.sh` into a new environment (`ENV_NAME=ab_pxrd_solver_verify`) builds
GSAS-II, verifies the MACE checksum, and reproduces the example above exactly (R² 0.9984,
χ² 0.0234, Wyckoff 1d/1c/2g, 23 structures in 41 attempts).

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
[`inference/_benchmark/cases.json`](../_benchmark/cases.json)), 30-minute limit and 12 GB memory cap
per run, scored with `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`. The solver returns one
structure, so any-match equals top-1. Full report: [`benchmark.json`](benchmark.json).

| Setting | Matched | Solver finished without a structure | Memory cap | Timeout | Upstream crash | Rejected (not Cu) |
|---------|---------|------|------|------|------|------|
| composition + space group | 1/12 | 2 | 2 | 2 | 1 | 4 |
| composition only | 0/12 | 0 | 7 | 1 | 0 | 4 |
| control: composition + space group, another case's pattern | 0/12 | 3 | 2 | 1 | 2 | 4 |

- **Solved:** AlPO₄ (P6₃mc), status C-Success, R² 0.990, RMS 0.007 against the reference, 435 s.
- **Finished without a structure:** BaTiO₃ and KCaCO₃F, in about 7 s each. Indexing found only
  supercells (smallest 257.6 Å³ for BaTiO₃, whose cell is 64.4 Å³; 822.8 Å³ for KCaCO₃F, cell
  100.4 Å³), so there were "No viable (cell, SPG) pairs". The KCaCO₃F peaks match the reference
  except for one weak extra reflection at 29.44° (4%, probably a calcite impurity); the BaTiO₃
  data contain Kα2 shoulders.
- **Memory cap:** CdBiClO₂ (P2₁/m) and Na₂LiAlF₆ (P2₁/c) within 8 s; without a space group,
  every Cu case except BaTiO₃ within 94–308 s, before any structure was generated (the logs that
  were flushed show the cell search over candidate space groups).
- **Timeout:** LaInO₃ (Pnma) and KLaTiO₄ (P4/nmm) with a space group; BaTiO₃ without one.
- **Upstream crash:** `UnboundLocalError: total_count` (`pxrd_app/core.py:1058`) for EuI₂, and in
  two control runs.
- **Rejected:** Mg₂Si, Mg₂Sn (0.499 Å synchrotron) and both Rb₂S phases (Mo Kα), by design.

The control solves nothing, but with one success in the matched setting this does not
demonstrate pattern dependence. The pipeline does refine against the pattern, and the input check
above shows the prepared patterns are correct.

**Status: limited.** The upstream example reproduces exactly from a fresh setup, but the solver
does not complete on most benchmark cases, for reasons inside upstream (memory, indexing
robustness, a crash) or by design (Cu only); see [the status definitions](../README.md#what-verified-means).

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol). Fixed as a
result: only Cu data are accepted (an earlier version converted any wavelength and Q data);
only the selected solution is returned (an earlier version copied every `Match_*.cif`);
partial scans are padded with the measured baseline instead of zeros; environment pins; whole
process-tree cleanup on timeout; and corrected documentation of the Z search (`max_Z` 24, at most
20 atoms per primitive cell, and the predicted density range).

## Known limitations

- Cu laboratory data only, 10–80° 2θ. The instrument model has a single wavelength, so Kα2
  shoulders in unstripped data appear as extra peaks; prefer Kα2-stripped or Kα1-monochromated
  data. Check the data rather than the label: the BaTiO₃ benchmark file declares 1.54056 Å
  (Kα1), but its high-angle peaks have Kα2 shoulders 0.14–0.22° above them, matching the
  doublet splitting.
- Indexing is not robust to extra peaks. A single weak impurity reflection or unresolved
  Kα2 shoulders leave only supercells, and the run ends with "No viable (cell, SPG) pairs".
- Monoclinic (and likely triclinic) searches need more memory than a 32 GB machine provides.
- Some runs end in an upstream `UnboundLocalError: total_count`
  (`pxrd_app/core.py:1058`); the wrapper reports the exit code.
- Long searches: orthorhombic and tetragonal cases with several sites reached the 30-minute
  timeout without a solution.
- Returns one structure, so match-rate metrics are top-1 by construction.
