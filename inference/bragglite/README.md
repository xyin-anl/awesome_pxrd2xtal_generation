# Bragglite

Diffusion Transformer (Crystalite backbone, EDM sampler) that generates primitive-cell crystal
structures from a PXRD profile plus either the exact composition or the element set and atom
count. The pattern enters as up to 128 peak tokens `[q, intensity, q-width, prominence]` with
Fourier position features, by in-context conditioning.
Upstream: [dongyeop3813/PXRD-Bench](https://github.com/dongyeop3813/PXRD-Bench) ·
Paper: [arXiv:2610.16480](https://arxiv.org/abs/2610.16480) ·
Weights: [Zenodo 23878332](https://zenodo.org/records/23878332) (CC BY 4.0)

> **License: noncommercial component.** Original Bragglite code is MIT, but upstream retains
> FlowMM-derived helpers under **CC BY-NC 4.0** (`licenses/FlowMM-CC-BY-NC-4.0.txt` in the
> upstream checkout); upstream states that the noncommercial restriction applies when those
> components are used. CDVAE-derived helpers keep their MIT notice, and the checkpoint archive is
> CC BY 4.0. This wrapper vendors none of it: `setup.sh` clones upstream and downloads the archive.
> Check the upstream terms before any commercial use.

## Quick start

```bash
./setup.sh                      # conda env "bragglite", upstream at the pinned commit, checkpoints (1.58 GB archive)
conda activate bragglite
python run.py --pattern scan.xy --wavelength CuKa1 --composition BaTiO3 --z 1 --spacegroup P4mm --out results/
python run.py --pattern scan.xy --wavelength CuKa1 --mode element-set --composition BaTiO3 --z 1 --out results_es/
python run.py --pattern scan.xy --wavelength CuKa1 --mode element-set --elements Ba,Ti,O --n-atoms 5 --out results_es/
```

Outputs: `results/candidates/candidate_<hypothesis>_<k>.cif`, `results/results.json`, and
`results/conditioning_peaks.csv` (the exact peak tokens the model received).

| Option | Notes |
|--------|-------|
| `--pattern`, `--wavelength`, `--x-unit` | Full measured profile; see "How the input is prepared" |
| `--mode composition` (default) | Exact primitive-cell multiplicities; atom types are fixed and only coordinates and lattice are sampled |
| `--mode element-set` | Only the element set and the atom count reach the model; it samples the multiplicities |
| `--composition`, `--z`, `--spacegroup` | The atom list is `composition x Z_primitive`; `--spacegroup` only converts a conventional Z to the primitive cell, otherwise every compatible centering is tried and each hypothesis gets its own `--n-samples` |
| `--n-atoms`, `--elements` | Primitive-cell atom count instead of `--z`; `--elements` (element-set mode) needs no composition |
| `--training-set` | `pxrd_bench` (default, at most 52 atoms) or `mp20` (at most 20 atoms) checkpoints |
| `--n-samples`, `--seed` | Candidates per hypothesis (upstream K = 20) and the evaluation seed (777) |
| `--cu-positions` | `ka1` (default): Cu peak maxima are taken as Cu Kα1; `declared`: use the declared wavelength |
| `--no-background-subtraction` | Skip upstream's order-5 polynomial background, for profiles that are already background-free |
| `--device`, `--fp32` | `cuda` (default, bfloat16 autocast as upstream) or `cpu` (float32, slow) |

Bragglite needs the primitive-cell atom count: give `--z` or `--n-atoms`. Candidates are
unrefined and are returned in generation order; upstream does not rank them, so "top-1" below
means the first sample, as in the paper's Match@1.

## What the model is conditioned on

Read from the pinned upstream code and the settings files shipped with each checkpoint:

- **Training patterns** are simulated, not measured: Bragg reflections from upstream's
  pymatgen-compatible simulator (`src/pxrd/bragg_simulator.py`) at **1.54184 Å** (averaged Cu Kα,
  one line, no Kα2), broadened with a pseudo-Voigt (Caglioti U, V, W = 0.1, 0.01, 0.1; η = 0.1)
  onto **850 bins, 5-90° 2θ, 0.1° step**, and max-normalized. Training adds profile randomization
  (peak jitter ±0.2°, zero shift ±0.3°, width ×0.4-2.2, η 0-0.8, intensity ×0.6-1.4, dropping up
  to 20% of the weakest peaks, smooth background up to 0.35, noise up to 0.04;
  `configs/data_augmentation/pxrd.yaml`).
- **Peak tokens** (`src/pxrd/peaks.py: xrdiff_feature_vector_from_profile`, `peaks_128x4`):
  `scipy.signal.find_peaks` with prominence ≥ 0 on the normalized profile, FWHM from
  `peak_widths` (bins × 0.1° 2θ), the 128 highest peaks sorted by angle, each stored as
  `[width, prominence, height, d]`. The encoder turns them into `[q = 2π/d, height, width in q,
  prominence]` and adds Fourier features of sin θ at 1.54184 Å (16 log-spaced frequencies 1-16).
- **Chemistry**: the primitive-cell atom list (training cells are primitive Niggli cells). In
  composition mode the atom types are pinned; in element-set mode an allowed-element mask is
  given and the atom count fixes the number of slots. At most 52 atoms (PXRD-Bench) or 20
  (MP-20).
- **Sampling** (from each checkpoint's stored arguments, identical to the shipped settings files):
  EMA weights, stochastic Heun, 150 steps, Karras schedule ρ = 7, S_churn 60, S_noise 1.003,
  σ 0.002-80, anti-annealing (types, coordinates, lattice) = (0, 10, 10), bfloat16, stateless
  per-candidate seeds from seed 777.

## How the input is prepared

`run.py` hands the pattern to upstream's own experimental loader, the path `eval_bragglite` uses
for measured opXRD profiles (`src/data/experimental_pxrd.py`, called through
`src/pxrd/eval_task_runtime.py:_build_dataset` with the checkpoint's arguments):

1. the pattern is loaded with `inference/_common/pxrd_io.py` (column files, pdCIF, or Q);
2. angles are mapped to 1.54184 Å through d-spacing with upstream's `_canonicalize_wavelength`
   (upstream maps each opXRD profile from its primary wavelength). For Cu radiation the source is
   taken as **Cu Kα1** by default, also for files declared as averaged Cu Kα: the maxima of a
   Kα1/Kα2 doublet sit at Kα1 (see `inference/AGENTS.md`). `--cu-positions declared` turns this off;
3. upstream then interpolates the profile linearly onto its 850-bin grid (zero outside the scan),
   applies a Savitzky-Golay filter (7 points, order 3), subtracts an order-5 polynomial fitted over
   the scanned range, clips at zero, normalizes, and extracts the peak tokens as above. Upstream's
   loader reads a CSV/CIF pair from a directory, so `run.py` writes the converted profile there
   with a one-atom placeholder CIF and keeps only the pattern features;
4. one token item per atom-count hypothesis gets the atom list, and upstream's evaluation sampler
   (`_prediction_rows`, `batch_candidates`, chunk 320 as in the shipped settings) draws the
   candidates, which upstream's `record_to_structure` turns into structures.

Nothing in upstream is patched. The upstream dataset caches (PXRD-Bench, MP-20, RRUFF, opXRD) are
not needed at inference time.

## Checkpoints

| File (archive `checkpoints/single_phase/`) | Training data | Mode | Max atoms | sha256 |
|------|---------------|------|-----------|--------|
| `pxrd_bench_stoichiometry.pt` | PXRD-Bench (≈4.1 M simulated) | composition | 52 | `2b6fd3ff…a1400` |
| `pxrd_bench_element_set.pt` | PXRD-Bench | element set | 52 | `fd274604…40805` |
| `mp20_stoichiometry.pt` | MP-20 | composition | 20 | `241bde11…39296` |
| `mp20_element_set.pt` | MP-20 | element set | 20 | `253a47ef…3f883` |

Archive `bragglite-checkpoints-v1.tar.gz`: 1,583,482,073 bytes, md5
`6f1f8e2d60e438b79349e8f84f8eb292` (as deposited), sha256
`ccc981c46af394b61e4193d488a1cdb5b876c7bfab2659e8926637252c31a39c`. Full checksums are in
`manifest.yaml` and `setup.sh`. The paper's headline experimental result (44.2% top-1 on RRUFF
with known composition) is the PXRD-Bench model, which is the default here.

## Verification (2026-11-07, RTX 4090, upstream commit `2a791ca`)

`setup.sh` was run from scratch (new environment, clone, and 1.58 GB download; 9.6 min), and a
second environment built from it gives byte-identical candidates and peak tokens to the
development environment for the same input.

**Upstream reproduction.** Upstream ships no single-pattern example, and the paper's
experimental sets (RRUFF-148 as provided by the XRDiff authors, frozen opXRD archives) are not
public. The closest published number with public data is the simulated MP-20 test split
(Table 1, MP-20-trained Bragglite, strict matcher ltol/stol/angle_tol 0.2/0.3/5°, no refinement).
`reproduce_upstream.py` rebuilds it with upstream's own simulator, features, sampler, and
scorer; upstream's frozen MP-20 token and feature caches are not public, so features are
recomputed from the CDVAE CIFs (primitive Niggli cells). Targets are taken in upstream's order
(`torch.randperm`, seed 777), 20 candidates each. Summaries: [`reproduce_upstream.json`](reproduce_upstream.json).

TBD_REPRO

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
`inference/_benchmark/cases.json`), 20 samples per atom-count hypothesis, scored with the
harness's looser `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)` (the paper uses
0.3/0.2/5°), fractions over all returned candidates, averaged across cases; no runs failed (full
report: [`benchmark.json`](benchmark.json)). PXRD-Bench checkpoints unless marked MP-20.

| Setting | Any match | First candidate matches | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|-------------------------|---------------------|-----------------------------------|-----------------|
| composition + Z + space group | 10/12 | 8/12 | 0.73 | 0.33 | 20 |
| same, seed 778 (run-to-run check) | 10/12 | 7/12 | 0.72 | 0.32 | 20 |
| same, Cu positions as declared | 10/12 | 8/12 | 0.72 | 0.34 | 20 |
| composition + Z (all compatible centerings tried) | 10/12 | 4/12 | 0.48 | 0.22 | 45 |
| element set + atom count (Z + space group) | 9/12 | 4/12 | 0.37 | 0.19 | 20 |
| composition + Z + space group, MP-20 checkpoint | 11/12 | 9/12 | 0.81 | 0.46 | 20 |
| element set + atom count, MP-20 checkpoint | 9/12 | 8/12 | 0.57 | 0.14 | 20 |
| control: composition + Z + space group, another case's pattern | 7/12 | 3/12 | 0.17 | 0.01 | 20 |

The pattern matters. Replacing it with another case's pattern lowers the mean match fraction from
0.73 to 0.17 (a second seed gives 0.72, so run-to-run variation is about 0.01) and the
correct-space-group fraction from 0.33 to 0.01. It drops to 0.00-0.15 for AlPO4, Mg2Si, Mg2Sn,
LaInO3, Rb2S phase II, and KCaCO3F. Matches that survive the wrong pattern come from the
composition prior: Na2LiAlF6 (1.00 to 0.65), BaTiO3 (1.00 to 0.40), EuI2 (0.75 to 0.35), and
CdBiClO2 (0.70 to 0.25), with best RMS worsening for EuI2 (0.117 to 0.391).

Both misses with the default checkpoint have few usable peaks or a long axis: Rb2S phase III
(Mo, 5-18° 2θ, which maps to about 11-40° at 1.54184 Å with three strong peaks) and KLaTiO4
(c = 13.5 Å). The MP-20 checkpoint solves Rb2S phase III (0.95) but misses CdBiClO2, which
the PXRD-Bench checkpoint solves (0.70). On these 12 small cells the MP-20 checkpoints score
higher, consistent with the paper's RRUFF results (52.4% vs 44.2% top-1 with known
composition); the PXRD-Bench checkpoints stay the default because they cover up to 52 atoms and a
much wider training set. Mapping Cu data from Kα1 or from the declared wavelength makes no
difference here (0.73 vs 0.72): the shift is about 0.1° at 90°, within the training jitter
(±0.2°) and zero shift (±0.3°).

Loose matching hides symmetry errors: BaTiO3 matches in all 20 samples but none has the
tetragonal P4mm symmetry, and KCaCO3F matches 18/20 with 0.00 correct space group. Without a
space group, composition + Z tries every compatible centering (45 candidates on average), and
fewer of them match (0.48), because the wrong-cell hypotheses contribute misses.

**Runtime and memory.** TBD_RUNTIME

## Known limitations

- Single phase only. The two-phase pipeline (XDecomposer decomposition, reference and
  predicted-phase caches) is later scope; its checkpoints are in the same archive but unused.
- The primitive-cell atom count is required (the paper's oracle-N setting); there is no Z search.
- Candidates are unrefined and unranked.
- Only angles are converted between wavelengths. Intensities are not corrected for the
  wavelength-dependent Lorentz-polarization factor, and Mo, synchrotron, or Q data are mapped
  onto the 5-90° window at 1.54184 Å (a Mo scan to 18° covers only ~5-40° there).
- Kα2 is not stripped: resolved Kα2 satellites become extra peak tokens, as in upstream's own
  opXRD path. The model was trained on single-line profiles.
- Upstream's linear interpolation onto the 0.1° grid subsamples finer scans rather than
  averaging them.
- RRUFF-148 and the frozen opXRD archives, the paper's experimental sets, are not public, so the
  experimental numbers in the paper could not be rerun; the upstream reproduction below uses the
  simulated MP-20 test split.
