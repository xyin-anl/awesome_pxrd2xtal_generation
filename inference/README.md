# Runnable inference

Each subdirectory wraps one published PXRD-to-structure model behind the same command-line
interface, so you can run it on your own GPU or CPU against your own data. Upstream code is never
copied here: `setup.sh` clones it at a pinned, tested commit and the weights download from their
official location.

## Using a model

```bash
cd inference/<model>
./setup.sh                                   # once: conda env + upstream checkout
conda activate <env>                         # name in manifest.yaml
python run.py --pattern scan.xy --wavelength CuKa --composition TiO2 --out results/
```

Common options (a model may not support all of them; see its `manifest.yaml`):

| Option | Meaning |
|--------|---------|
| `--pattern` | Measured pattern: `.xy`, `.xye`, `.dat`, `.csv` (2theta, intensity) or a pdCIF |
| `--wavelength` | Angstrom or a name (`CuKa`, `CuKa1`, `MoKa`, ...); read from pdCIF if omitted |
| `--composition` | Reduced formula, e.g. `KCaCO3F` |
| `--z` | Formula units per cell |
| `--spacegroup` | Hermann-Mauguin symbol |
| `--n-samples` | Candidates to generate |
| `--device` | `cuda` or `cpu` |
| `--out` | Output directory |

Every script writes `out/candidates/*.cif` and `out/results.json` (inputs, checkpoint, runtime,
candidate list). Some models write extra files such as the peak list they were conditioned on.

## Directory layout

| Path | Purpose |
|------|---------|
| `_common/pxrd_io.py` | Pattern loading, wavelength conversion, background removal, peak picking |
| `_benchmark/` | Shared experimental benchmark (`cases.json`, reference structures) and `benchmark.py` |
| `<model>/manifest.yaml` | Upstream repo + commit, weights, license, required inputs, hardware, benchmark settings, verification status |
| `<model>/environment.yml`, `setup.sh` | Reproducible environment and upstream checkout |
| `<model>/run.py` | The common command-line interface |
| `<model>/README.md` | Model-specific notes, caveats, and the latest benchmark result |
| `<model>/benchmark.json` | Full benchmark report from the last verification run |

All wrappers follow this layout. [`TRIAGE.md`](TRIAGE.md) lists which other catalogued models are
feasible to wrap next and which are blocked.

## What "verified" means

A model is marked `verified` in its manifest only when all of these hold on the recorded hardware:

1. `setup.sh` succeeds in a fresh environment.
2. The upstream authors' own example reproduces their reported behaviour.
3. `benchmark.py` completes on every case, and the report is committed as `benchmark.json`.
4. The mismatched-pattern control (`--control`) scores clearly below the matched setting, showing
   that the diffraction data actually influences the output.
5. An independent review of the wrapper against the pinned upstream code found no open issues
   (findings and fixes are summarized in the model README).

A model is marked `reproduced` when 1-3 and 5 hold but the control is not clearly lower: the
wrapper behaves like upstream, yet on this benchmark the output barely depends on the pattern.

A model is marked `limited` when 1, 2 and 5 hold but most benchmark runs cannot complete for
reasons inside upstream or by design (for example, memory use beyond a workstation, an upstream
crash, or a single supported radiation). The model README lists each failure and its cause.

Benchmark numbers come from a small set of 12 experimental patterns. They are a sanity check for
the wrapper, not a ranking of models, and they are not comparable to numbers in the papers.

## Benchmark

`_benchmark/cases.json` is built from the experimental pdCIFs in `exp_pxrd_data/pxrdnet/` by
`_benchmark/build_cases.py`; files whose reference structure cannot be parsed reliably are listed
under `excluded` with the reason. Matching uses pymatgen `StructureMatcher(stol=0.5, angle_tol=10,
ltol=0.3)`, the setting used by CDVAE/DiffCSP-style benchmarks. Reported metrics:

- `match_rate_any`: fraction of cases where at least one candidate matches the reference
- `match_rate_top1`: fraction of cases where the first candidate matches; meaningful only for
  wrappers that rank their output (Uni-3DAR, PXRDnet, XRDSol)
- `mean_match_fraction`: average fraction of returned candidates that match
- `mean_spacegroup_fraction`: average fraction of returned candidates with the reference space
  group (unreadable CIFs count as wrong)
- `mean_candidates`: average number of candidates scored; settings that search over Z return
  more candidates and therefore get more chances in `match_rate_any`
- `mean_best_rms`: average normalized RMS displacement of the best match

Failed runs count as misses in every rate. Each setting records its date, sample count (an
`--n-samples` in its extra arguments takes precedence), and extra arguments; `--settings` reruns
merge into an existing report only if the case set and matcher are unchanged. `--resume` reuses
a completed case only when its command and the wrapper and shared code are unchanged. `--only`
runs a subset of cases and must write to a separate `--report`.
