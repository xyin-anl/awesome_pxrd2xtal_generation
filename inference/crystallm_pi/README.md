# CrystaLLM-π

Property-conditioned CIF language model; the XRD checkpoints generate structures from a formula
plus the 20 strongest diffraction peaks.
Upstream: [C-Bone-UCL/CrystaLLM-pi](https://github.com/C-Bone-UCL/CrystaLLM-pi) (MIT) ·
Paper: [arXiv:2511.21299](https://arxiv.org/abs/2511.21299)

## Quick start

```bash
./setup.sh                      # conda env "crystallm_pi", upstream at the pinned commit, pinned weights
conda activate crystallm_pi
python run.py --pattern scan.xy --wavelength CuKa --composition KCaCO3F --z 1 --n-samples 20 --out results/
```

Outputs: `results/candidates/*.cif`, `results/results.json`, and `results/peaks_cuka_top20.csv`
(the exact peak list the model was conditioned on; check it before trusting the output).

| Option | Notes |
|--------|-------|
| `--pattern` / `--peaks` | Raw profile (peak-picked here) or your own `2theta,intensity` CSV of picked peaks |
| `--wavelength`, `--x-unit` | Needed for column files; peaks are converted to averaged Cu Kα |
| `--strip-ka2` | `auto` (default: merge resolved Cu Kα2 satellites for Cu data), `on`, or `off` for monochromated Kα1 |
| `--composition` | Required |
| `--z` | Optional; if omitted, Z = 1, 2, 3, 4, 6 are each sampled `--n-samples` times |
| `--spacegroup` | Optional; tokenizer spelling such as `P4_2/mnm` (other spellings are rejected) |
| `--model` | `mattergen` (default) or `chili100k` |

GPU memory: about 2 GB. CPU works but is slower.

## How the input is prepared

The XRD checkpoints were trained on pymatgen-simulated Cu Kα peak lists: the 20 most intense
peaks between 0 and 90° 2θ, intensities scaled to 100. Upstream expects already picked peaks and
does not check this, so a raw profile passed straight through produces meaningless conditioning.
`run.py` therefore:

1. removes the background and picks peaks with `inference/_common/pxrd_io.py` (for Cu data,
   resolved Kα2 satellites with 30-75% of their parent's area are merged into the Kα1 peak;
   intensities are height × FWHM area estimates);
2. converts peak positions to the averaged Cu Kα wavelength used in training (1.54184 Å), drops
   those beyond 90°, and keeps the 20 strongest (ties broken by angle, as upstream does).

Upstream's own input path converts user data to Cu Kα1 (1.54056 Å), which shifts every peak
relative to training by up to 0.095° at 90°. `run.py` passes the converted peaks with
`--xrd_wavelength 1.54056` so that upstream's conversion is a no-op and the model sees
training-consistent angles.

On the 12 benchmark patterns the picker recovers on average 81% of the 10 strongest simulated
reference peaks within 0.2° at Cu Kα (mean precision 0.89; 10 of 12 patterns at 100%). It is
weakest for the two high-pressure diamond-anvil-cell patterns (Rb2S) and
patterns with few peaks. If you have a refined peak list, pass it with `--peaks`.

## Verification (2026-10-04, RTX 4090, upstream commit `12f1d72`)

**Upstream example.** The README's rutile recovery (TiO2, Z = 2,
`tests/fixtures/test_rutile_processed.csv`, Mattergen-XRD) gave 8/10 candidates matching rutile
P4₂/mnm (RMS ≤ 0.008), both through upstream's script and through `run.py --peaks` with the
pinned weights; the other two were hexagonal polymorphs.

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
`inference/_benchmark/cases.json`), 20 samples per run, default `mattergen` checkpoint,
`StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`. Fractions are over all returned
candidates; failed runs would count as misses (there were none). Full report:
[`benchmark.json`](benchmark.json).

| Setting | Any match | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|---------------------|-----------------------------------|-----------------|
| composition + Z | 10/12 | 0.60 | 0.59 | 19 |
| composition + Z + space group | 12/12 | 0.89 | 1.00 | 20 |
| composition only (Z searched) | 12/12 | 0.18 | 0.13 | 92 |
| control: composition + Z, another case's pattern | 8/12 | 0.29 | 0.28 | 18 |

The control halves the match fraction. Cases that depend on the pattern (Mg2Si, Mg2Sn,
Na2LiAlF6, Rb2S phase II, KLaTiO4, LaInO3) drop sharply with the wrong pattern, while BaTiO3,
KCaCO3F, and CdBiClO2 are recovered from composition + Z alone, so their matches say little
about use of the diffraction data. The composition-only row samples five Z values and therefore
has about five times more candidates than the other rows.

Misses without a space group: AlPO4 (P6₃mc) and the high-pressure Rb2S phase III. The loose
matcher accepts near-cubic candidates for tetragonal BaTiO3 (match fraction 1.00 but only 50% in
P4mm), so read the space-group column alongside the match rate.

Median runtime per run: about 11 s for 20 samples with a known Z (model load included). GPU
memory: about 1-2 GB.

**Independent review.** The wrapper was reviewed against the pinned upstream by Codex
(gpt-5.6-sol, high). Fixed as a result: peaks converted to averaged Cu Kα as in training
(upstream's own Kα1 conversion is bypassed), stricter Kα2 merging, upstream's tie-break order,
space-group spelling validation, Z ≥ 1, pinned Hugging Face revisions, and full dependency pins.
The previous benchmark (13 cases, before these fixes) gave 10/13 any-match and a 0.57 match
fraction for composition + Z.

## Known limitations

- Peak picking is heuristic; overlapped or weak peaks and impurity phases change the condition.
- Patterns are restricted to what maps into 0-90° 2θ at Cu Kα1 (d > 1.09 Å).
- Training conditions counted a reflection once per contributing (hkl) family, so overlapping
  reflections (common in high-symmetry cells) could fill several of the 20 slots. Measured peak
  lists cannot reproduce this without indexing; expect weaker conditioning for such patterns.
- Weights are loaded from local snapshots at the Hugging Face revisions in `manifest.yaml`.
