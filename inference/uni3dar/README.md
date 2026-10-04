# Uni-3DAR

Autoregressive 3D structure generator; the `mp20_pxrd.pt` checkpoint predicts crystal structures
from a diffraction peak list plus the cell composition.
Upstream: [dptech-corp/Uni-3DAR](https://github.com/dptech-corp/Uni-3DAR) (MIT) ·
Paper: [arXiv:2503.16278](https://arxiv.org/abs/2503.16278)

## Quick start

```bash
./setup.sh                      # conda env "uni3dar", Uni-Core, upstream at the pinned commit, checkpoint (1.3 GB)
conda activate uni3dar
python run.py --pattern scan.xy --wavelength CuKa --composition KCaCO3F --z 1 --n-samples 20 --out results/
```

Outputs: `results/candidates/candidate_zp<Z>_<rank>.cif` (ranked best first within each
primitive-cell Z), `results/results.json` (model score, primitive Z, and the scan's Cu Kα
coverage), and `results/peaks_cuka_0-120.csv` (the peak list the model was conditioned on).

| Option | Notes |
|--------|-------|
| `--pattern` / `--peaks` | Raw profile (peak-picked here) or your own `2theta,intensity` CSV of picked peaks (needs `--wavelength`) |
| `--wavelength`, `--x-unit` | Needed for column files; peaks are converted to averaged Cu Kα |
| `--composition`, `--z` | Both required; Z is per conventional cell |
| `--spacegroup` | Optional; only used to convert Z to the primitive cell (R groups: Z for hexagonal axes) |
| `--n-samples` | Candidates per primitive-Z hypothesis |
| `--batch-size` | 64 by default: ~8 GB for small cells but up to ~22 GB for 20-atom cells. Use 32 (~6 GB) or 16 on smaller GPUs; the number of generation calls is scaled up so the sample budget stays the same. Upstream's 256 does not fit in 24 GB |

Requires Linux x86-64 and an NVIDIA GPU with compute capability ≥ 8.0 (the pinned
flash-attention wheel); there is no CPU path.

## How the input is prepared

The checkpoint was trained on stick patterns (0-120° 2θ, intensities scaled to 100) and on the
atom list of the **primitive** cell. Upstream ships the training data but not the script that
made it; peak positions in its MP-20 records match pymatgen's averaged Cu Kα
(`XRDCalculator("CuKa")`) to within 0.01°, while Cu Kα1 is off by ~0.06-0.08°. Upstream thresholds the peaks at
intensity 5, fills the gaps with zeros, and interpolates onto a 0.1° grid. `run.py`:

1. picks peaks with `inference/_common/pxrd_io.py` and converts them to averaged Cu Kα;
2. converts the conventional-cell Z to the primitive cell using the lattice centering of
   `--spacegroup` (P 1, A/B/C/I 2, R 3, F 4); without a space group every compatible centering is
   tried. Model scores are not comparable across atom counts, so each hypothesis is ranked
   separately and contributes its own top `--n-samples`;
3. follows upstream's PXRD protocol (`uni3dar/inference.py`): oversamples 20× with the exact
   composition as a constraint, keeps only exact-composition structures, and ranks them by the
   model's score.

Peak picking from a raw profile is our addition, not part of upstream; it is checked only by
the benchmark below. The model has no notion of unmeasured ranges: anything outside the scan is
seen as "no peaks", so `run.py` warns when the scan does not span roughly 15-90° (Cu Kα).
Intensities are not corrected for wavelength-dependent factors (Lorentz-polarization,
absorption) when converting from other radiation.

The previous scripts in this directory (`uni3dar_inference.py`, `uni3dar_modal.py`) passed the
raw profile in the measurement's own wavelength, conditioned on the set of distinct elements
instead of the cell's atom list, skipped composition filtering and ranking, and loaded weights
with `strict=False`. They were replaced by `run.py`.

## Verification (2026-10-04, RTX 4090, upstream commit `bba82f5`)

**Upstream example.** Upstream's PXRD protocol on 40 random records (seed 0) of its own MP-20
test set (`mp20_data/test.lmdb`), using this wrapper's model loading and generation loop and
scored with upstream's `match_rate_at_k`: top-1 match rate 0.750 and top-20 0.800 at the default
batch size 64 (0.750 / 0.775 at 128). The paper reports a 75.08% top-1 match rate (Table 3).

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet`, 20 candidates per
primitive-Z hypothesis, batch size 64, `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`.
Fractions are over all returned candidates. Full report: [`benchmark.json`](benchmark.json).

| Setting | Any match | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|---------------------|-----------------------------------|-----------------|
| composition + Z (all centerings tried) | 8/12 | 0.42 | 0.29 | 45 |
| composition + Z + space group (exact primitive Z) | 8/12 | 0.67 | 0.54 | 20 |
| control: composition + Z, another case's pattern | 7/12 | 0.30 | 0.10 | 44 |

Without a space group, candidates from wrong-Z hypotheses lower the match fraction; the
space-group row is the fairer view of the model. The pattern clearly drives Mg2Si (0.92 → 0.37
with the wrong pattern), Mg2Sn (0.67 → 0.13), Rb2S phase II (0.33 → 0), and the space-group
fractions overall (0.29 → 0.10). BaTiO3, KCaCO3F, LaInO3, and EuI2 come out the same with either
pattern, i.e. composition alone decides them. Uni-3DAR found no match for AlPO4, CdBiClO2,
KLaTiO4, or Rb2S phase III in any setting.

Runtime: 17-66 s per case (model load included). Peak GPU memory 22.4 GB on LaInO3
(20-atom primitive cell) at batch size 64; 6.2 GB at batch size 32.

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol, high).
Fixed as a result: separate ranking per primitive-Z hypothesis (scores are not comparable across
atom counts), rejection of non-integer atom counts, `--peaks` requires `--wavelength` and finite
positive values, the oversampling loop stops at the target count, scan-coverage warning, and the
Linux x86-64 requirement is documented.

## Known limitations

- Trained on MP-20 (at most 20 atoms per primitive cell); larger cells are out of distribution
  even though the model accepts up to 128 atoms.
- Peak picking is heuristic; see `inference/crystallm_pi/README.md` for its benchmark accuracy.
- Upstream builds structures inside a bare `except:`, so environment problems (for example a
  missing `ase`) would show up as zero candidates. `run.py` imports `ase` up front and exits with
  an error when no exact-composition structure is generated.
