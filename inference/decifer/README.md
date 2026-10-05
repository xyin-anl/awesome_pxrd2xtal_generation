# deCIFer

Autoregressive CIF language model conditioned on the full PXRD profile, optionally prompted with
the cell composition.
Upstream: [FrederikLizakJohansen/deCIFer](https://github.com/FrederikLizakJohansen/deCIFer) (MIT) ·
Paper: [arXiv:2502.02189](https://arxiv.org/abs/2502.02189)

## Quick start

```bash
./setup.sh                      # conda env "decifer", upstream at the pinned commit, checkpoint (665 MB)
conda activate decifer
python run.py --pattern scan.xy --wavelength CuKa --composition CeO2 --z 4 --n-samples 25 --out results/
```

Outputs: `results/candidates/*.cif` and `results/results.json` (includes the exact
`prompt_formula`). Candidates that do not parse as CIF are dropped, so you may get fewer than
`--n-samples`.

| Option | Notes |
|--------|-------|
| `--pattern` | Full profile; `.xy/.xye/.dat/.csv` or pdCIF |
| `--wavelength`, `--x-unit` | Wavelength for 2θ data; use `--x-unit q` for data already in Q (1/Å) |
| `--composition`, `--z` | Optional. Prompted as the full-cell formula (CeO2 with Z = 4 → `Ce4O8`) |
| `--q-min`, `--q-max` | Q crop; default 0-10 1/Å (the training range). The authors' experiments used 1.5-8 |

Space-group prompting is not available: the upstream experimental pipeline accepts a
`spacegroup` argument but does not use it. GPU memory: generation recomputes the full context
each step, so memory grows with CIF length; we measured peaks up to ~20 GB on the benchmark.

## How the input is prepared

`run.py` hands the pattern to upstream's own `DeciferPipeline` (`bin/experimental_pipeline.py`),
which converts 2θ to Q, min-max normalizes, crops to the Q window, and interpolates onto 1000
points over 0-10 1/Å. No background subtraction or smoothing is applied, matching the paper.
For pdCIF input, `utils/parse_cifs.py` uses the file's net intensities, or subtracts the file's
own background column from total intensities. The default keeps the full 0-10 1/Å range the
model was trained on. We compared it with the authors' experimental 1.5-8 1/Å crop in two
benchmark runs and found no consistent winner (see below); use `--q-min 1.5 --q-max 8` when
low-angle artifacts (beam stop, air scatter, sample holder) dominate the low-Q region.

Two details matter and were wrong in this repository's previous deCIFer script:

- **Composition prompt.** Training CIFs name the data block after the full-cell
  `_chemical_formula_sum` with explicit 1s (`Ce4O8`, `K1Ca1C1O3F1`). Prompting with a reduced
  formula breaks generation: on crystalline CeO2, `CeO2` gave 0/24 correct structures (every
  output was O2), while `Ce4O8` gave 22/25.
- **Checkpoint.** The published `decifer_v1_ckpt.pt` must be loaded with torch < 2.6 (it stores
  omegaconf objects and `__main__.TrainConfig`), so the environment pins torch 2.5.1.

## Verification (2026-10-04, RTX 4090, upstream commit `5b22c01`)

**Upstream example.** The authors' crystalline CeO2 result, stored as
`pickles/crystalline_CeO2.pkl` in the deCIFer data archive. The scan is identical to
`exp_pxrd_data/decifer/HEO/crystalline_CeO2_BM31.xye`. We used the prompts and Q crop
(`--q-min 1.5 --q-max 8`) from upstream `bin/run_protocol.py` (which itself targets the
nanoparticle scan), 25 samples each,
matched against fluorite CeO2:

| Prompt | Our run | Authors' stored sample (1 per protocol) |
|--------|---------|------------------------|
| `Ce4O8` | 22/25 match (22 in Fm-3m) | correct |
| `Ce1O2` | 0/25 | incorrect |
| none | 0/24 (10/24 in Fm-3m, wrong elements) | incorrect |

The upstream README labels this file as already in Q, but it is 2θ: the conditioning vector in
the authors' result pickle is reproduced (max deviation 0.015) only with λ = 0.2545 Å.
Use `--wavelength 0.2545` with this file.

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet`, 20 samples per run,
`StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`. Fractions are over all returned
candidates. Full report: [`benchmark.json`](benchmark.json).

| Setting | Any match | Mean match fraction | Mean correct-space-group fraction | Mean best RMS |
|---------|-----------|---------------------|-----------------------------------|---------------|
| composition + Z (Q 0-10, default) | 8/12 | 0.34 | 0.37 | 0.043 |
| composition + Z, Q 1.5-8 (authors' crop) | 12/12 | 0.33 | 0.38 | 0.135 |
| pattern only (no composition) | 0/12 | 0.00 | 0.09 | - |
| control: composition + Z, another case's pattern | 6/12 | 0.09 | 0.20 | 0.146 |

The control cuts the match fraction from 0.34 to 0.09, so the pattern strongly drives the
output (KCaCO3F 0.55 → 0, LaInO3 0.95 → 0, BaTiO3 1.00 → 0.25). Without a composition prompt the
model gets the elements wrong in every case, as the paper also reports.

The two Q windows give the same mean match fraction. The crop finds at least one match in more
cases, but its matches are much looser (mean best RMS 0.135 vs 0.043). An earlier run on a
13-case version of the benchmark, before the review fixes, favoured the full range (0.35 vs 0.26
match fraction; 11/13 vs 8/13 any-match). With 20 unseeded samples per case these differences
are within sampling noise, so neither window is clearly better.

Runtime: 20-80 s per case for 20 samples.

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol, high).
Fixed as a result: Q-window validation (an empty window would silently give an all-zero
condition), a check that the checkpoint is the PXRD-conditioned model (not U-deCIFer), Z ≥ 1,
portable checksum verification, the pdCIF wavelength and background handling in the shared
parser, and README wording. The reviewer's suggestion to use the full training Q range was
tested as described above.

## Known limitations

- Generation is sequential and slow for large or low-symmetry cells (minutes per 20 samples).
- Cu Kα data up to 90° 2θ reaches only Q ≈ 5.8 1/Å of the 0-10 1/Å the model was trained on.
- Without a composition prompt the model reproduces the symmetry but usually not the elements.
