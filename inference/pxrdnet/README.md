# PXRDnet

CDVAE-based model that searches its latent space for structures whose predicted patterns match a
measured profile, constrained by the known atom list; trained on nanocrystal-broadened (sinc
filtered) MP-20 patterns.
Upstream: [gabeguo/cdvae_xrd](https://github.com/gabeguo/cdvae_xrd) (MIT) ·
Weights: [therealgabeguo/cdvae_xrd_sinc10](https://huggingface.co/therealgabeguo/cdvae_xrd_sinc10) ·
Paper: [Nature Materials 2025](https://www.nature.com/articles/s41563-025-02220-y)

## Quick start

```bash
./setup.sh                      # conda env "pxrdnet", upstream at the pinned commit, both checkpoints
conda activate pxrdnet
python run.py --pattern scan.xy --wavelength CuKa --composition KCaCO3F --z 1 --spacegroup P-6m2 --out results/
```

Outputs: the 10 best candidates per primitive-Z hypothesis in `results/candidates/` (ranked by the
L1 distance between their simulated and the measured pattern) and `results/results.json`.

| Option | Notes |
|--------|-------|
| `--pattern`, `--wavelength`, `--x-unit` | Full profile; binned by Q, so any wavelength works |
| `--composition`, `--z` | Both required; Z per conventional cell, converted to the primitive cell (at most 20 atoms) |
| `--spacegroup` | Optional; only used for that conversion. Without it every compatible centering is tried, each run separately |
| `--checkpoint` | `sinc100` (default; 100 Å filter, used for the paper's experimental results) or `sinc10` |
| `--num-starting-points`, `--num-gradient-steps` | Paper's experimental settings by default (100; 5000 per cosine cycle, 35,000 in total) |

Runtime is about 4 minutes per pattern with the default settings.

## How the input is prepared

`run.py` reproduces upstream's experimental pipeline with upstream's own code wherever possible:

1. `process_real_xrds/read_real_xrd.py:create_data`: each point's Q is computed from its angle and
   wavelength and binned onto the model's 4096-point grid (Cu Kα, 0-180° 2θ), keeping the
   maximum per bin, then normalized. For the 15 IUCr files in `exp_pxrd_data/pxrdnet` this
   produces exactly upstream's tensor (maximum difference 0) for 12 files; BaTiO3 and KLaTiO4
   differ by one bin because `utils/parse_cifs.py` uses the files' processed wavelength
   (`_pd_proc_wavelength`, Kα1) while upstream uses the source wavelength.
2. Upstream's `CrystDataset` applies the sinc filter and subsampling to produce the target.
3. Upstream's `optimize_latent_code`, `langevin_dynamics`, `create_materials`, and `smooth_xrds`
   run with the settings of `scripts/conditional_generation_experimental.sh`; the 10 candidates
   with the lowest L1 pattern loss are kept, as upstream does.

Upstream builds the test record from the reference CIF read with `CifParser.get_structures()`,
which returns the **primitive** cell (pymatgen 2023.3.10 default), so the model is conditioned on
primitive-cell atom counts (3 atoms for Mg2Si, not 12). `run.py` converts a conventional Z with
the lattice centering of `--spacegroup`, or runs every compatible centering separately. For user
data it builds a random cell holding those atoms, since only the atom count and types reach the
model. Candidates whose simulated pattern fails (upstream substitutes an all-zero pattern) are
excluded from the ranking.

The environment follows upstream `requirements.txt` with torch 2.0.1 instead of 2.0.0 (the
2.0.0 wheels lack `libnvrtc` for cuDNN convolutions). The checkpoints' stored hyperparameters
point the GemNet scale file at the authors' machine, so the decoder configuration resolved from
`hparams.yaml` is passed when loading.

The previous script in this directory reimplemented the same steps and was close to upstream; it
was replaced to call upstream directly and to share the common interface.

## Verification (2026-10-04, RTX 4090, upstream commit `1ce3845`)

**Upstream example.** The paper's experimental evaluation uses the 15 IUCr patterns in
`exp_pxrd_data/pxrdnet`; the benchmark below uses 12 of them (three are excluded for unreliable
reference structures, see `inference/_benchmark/cases.json`), so it doubles as the authors'
example. The paper
reports R-factors rather than a match rate for this set (mean Rwp² 0.383 ± 0.330) and shows
AlPO4, CdBiClO2, and LaInO3 among its examples.

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
`inference/_benchmark/cases.json`), 20 returned candidates per case, scored with
`StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)` and fractions over all returned candidates,
averaged across cases; no runs failed (full report: [`benchmark.json`](benchmark.json)).

| Setting | Any match | Top-1 match | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|-------------|---------------------|-----------------------------------|-----------------|
| composition + Z + space group (paper's experimental settings, sinc100) | 11/12 | 6/12 | 0.25 | 0.01 | 20 |
| control: composition + Z + space group, another case's pattern | 5/12 | 0/12 | 0.09 | 0.00 | 20 |

Replacing each case's pattern with another case's lowers any-match from 11/12 to 5/12 and the
mean match fraction from 0.25 to 0.09; the top-ranked candidate (lowest pattern loss) matches in 6/12
cases with the correct pattern and in none with the wrong one. The pattern clearly helps Mg2Si (0.95 to 0.25), Mg2Sn
(1.00 to 0.50), and BaTiO3 (0.65 to 0.10). AlPO4, CdBiClO2, Na2LiAlF6, EuI2, and KCaCO3F
each drop from 0.05 to 0.00, and LaInO3 from 0.10 to 0.00. There is no match-fraction benefit
for Rb2S phase II (0.05 in both), though its best RMS worsens from 0.167 to 0.466 with the
wrong pattern; Rb2S phase III actually improves from 0.05 to 0.15, while KLaTiO4 has no matches
in either run.

Candidates are unrefined, which explains the low correct-space-group fractions even among
structural matches; the control's 0.00 is rounded, not exactly zero. The space-group input
selects the primitive-cell atom count without enforcing symmetry. The discarded `composition_z`
run with conventional-cell atom counts gave 7/12 any-match and a 0.10 mean match fraction,
versus 11/12 and 0.25 here, evidence that the primitive-cell convention matters.

Mean best RMS is 0.182 over the 11 cases with a match. The loosest best matches are LaInO3
(0.494), CdBiClO2 (0.486), EuI2 (0.395), and Rb2S phase III (0.360), so an accepted match
need not be a close solution. Of the paper's highlighted examples, our AlPO4 run gives 1/20
matches with best RMS 0.028, CdBiClO2 gives 1/20, and LaInO3 gives 2/20; the latter two are
loose matches, and these structure-match results do not establish reproduction of the paper's
R-factors.

Median per-case `runtime_s`: 176.5 s for composition + Z + space group; peak GPU memory: about 8 GB.

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol, high).
Fixed as a result: primitive-cell atom counts (the first benchmark run used conventional-cell
counts and was discarded), exclusion of candidates whose pattern simulation failed, and loading
the checkpoint by its exact filename. Intensities recorded at identical angles are averaged by the
shared loader before binning (upstream keeps every point and takes the bin maximum); this only
matters for files with duplicate angles.

## Known limitations

- Requires the full atom list; at most 20 atoms.
- No wavelength-dependent intensity correction; the model expects nanocrystal-broadened patterns.
