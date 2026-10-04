# Crystalyze

CDVAE-based generator conditioned on a full PXRD profile (CNN encoder) with the element set as
a constraint; the paper refines its candidates afterwards (StructSnap/Rietveld).
Upstream: [ML-PXRD/Crystalyze](https://github.com/ML-PXRD/Crystalyze) (MIT) ·
Paper: [J. Am. Chem. Soc. 2024](https://pubs.acs.org/doi/10.1021/jacs.4c10244)

## Quick start

```bash
./setup.sh                      # conda env "crystalyze", upstream at the pinned commit, checkpoint (415 MB, Google Drive)
conda activate crystalyze
python run.py --pattern scan.xy --wavelength CuKa1 --composition LuOF --z 2 --spacegroup P4/nmm --n-samples 20 --out results/
```

Outputs: `results/candidates/candidate_<hypothesis>_<k>.cif` and `results/results.json`.

| Option | Notes |
|--------|-------|
| `--pattern`, `--wavelength`, `--x-unit` | Full profile; angles are converted to the model's 1.5406 Å |
| `--composition` | Required; without `--z` only the element set is constrained (upstream default) |
| `--z` | Fixes the atom count and types (upstream `--force_num_atoms --force_atom_types`); see below |
| `--spacegroup` | Only used to convert a conventional Z to the primitive cell |
| `--batch-size` | Samples generated in parallel (default 64) |

At most 20 atoms per cell (MP-20). Needs a CUDA GPU (upstream's sampler hard-codes `.cuda(0)`);
GPU memory: about 1.2 GB.

## How the input is prepared

`run.py` follows upstream's own experimental entry point (`cdvae/common/inference_utils.py`:
`xy_data_prep` and `solve_pxrd`):

1. angles are converted to 1.5406 Å (`hparams.yaml: model.wavelength`); the previous script in
   this directory skipped this, so Mo or synchrotron data were read as if they were Cu data;
2. the profile is cubic-interpolated onto 8500 points over 5-90° 2θ (zero outside the scan) and
   scaled to a maximum of 1. Upstream's dataset then zeroes the first 1000 points, so the model
   only sees **15-90°**; `run.py` rejects patterns with no signal in that range;
3. upstream's own functions write the inference files, its datamodule loads them, and its
   `reconstructon` runs the Langevin sampler.

Training cells are Niggli-reduced MP-20 cells, which are primitive, so a conventional Z is
converted using the lattice centering of `--spacegroup`; without a space group every compatible
centering is tried and each hypothesis gets its own `--n-samples` candidates.

Three changes were needed to run upstream at all on current hardware; none changes the model:

- **Environment.** Upstream pins torch 1.8.1 (CUDA 10.2). Its CUDA libraries cannot create
  cuSPARSE/cuSOLVER handles on current GPUs and drivers, so the environment uses torch 1.13.1
  (CUDA 11.7) with upstream's PyTorch Lightning 1.3.8 and torch-geometric 1.7.2.
- **DimeNet shim.** torch-geometric 1.7.2 fills a parameter in place during construction, which
  torch ≥ 1.9 rejects; `run.py` performs the same fill under `no_grad`. The checkpoint then
  overwrites every weight (verified: no missing or unexpected keys, maximum difference 0).
- **Checkpoint loading.** The checkpoint's stored hyperparameters point the GemNet scale file at
  the authors' cluster (`/home/gridsan/...`); `run.py` passes the decoder and data configuration
  resolved from `hparams.yaml` instead, as the previous script did.

Upstream draws `--n-samples` samples one after another; `run.py` puts `--n-samples` copies of the
input in one batch, each with its own random starting cell, which draws the same kind of
independent samples in parallel (4.5 min → under 1 min for 8 samples).

## Verification (2026-10-04, RTX 4090, upstream commit `d265e8b`)

**Upstream examples.** The eight experimental compounds from the paper that ship in
`exp_pxrd_data/crystalyze` (`reproduce_examples.py`; wavelength from each TOPAS `.inp`;
composition, Z, and space group from the published CIF; 20 samples; no refinement):

| Compound | Source | Matching candidates |
|----------|--------|---------------------|
| NaCu2P2 | Cu | 14/20 |
| LuOF | Cu | 6/20 |
| Rh3Bi | synchrotron 0.4066 Å | 5/20 |
| Ca2MnTeO6 | Cu | 1/20 |
| RuBi2 | synchrotron 0.4066 Å | 1/20 |
| HoNdV2O8, KBi3, ZrGe6Ni6 | Cu / synchrotron / Cu | 0/20 |

The paper solves these compounds after refining candidates with StructSnap and Rietveld
refinement, which this wrapper does not do; raw matches for 5 of 8 are consistent with that.

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
`inference/_benchmark/cases.json`), 20 samples per centering hypothesis, scored with
`StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)` and fractions over all returned candidates,
averaged across cases; no runs failed (full report: [`benchmark.json`](benchmark.json)).

| Setting | Any match | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|---------------------|-----------------------------------|-----------------|
| composition + Z (all compatible centerings tried) | 10/12 | 0.11 | 0.00 | 45 |
| composition + Z + space group (centering supplied) | 9/12 | 0.26 | 0.01 | 20 |
| composition only (element set constrained) | 5/12 | 0.18 | 0.03 | 20 |
| control: composition + Z, another case's pattern | 5/12 | 0.06 | 0.00 | 45 |

Replacing the pattern lowers the composition + Z mean match fraction from 0.11 to 0.06 and
any-match from 10/12 to 5/12. The pattern clearly helps AlPO4 (0.25 to 0.00), Mg2Si (0.38 to
0.10), LaInO3 (0.10 to 0.00), and KCaCO3F (0.10 to 0.00). Rb2S phase II also drops from 0.12
to 0.02, and its best RMS worsens from 0.061 to 0.307. There is no benefit for Na2LiAlF6
(the match fraction is unchanged), while BaTiO3 actually improves with the wrong pattern
(0.05 to 0.50); these matches do not demonstrate use of the correct diffraction data.

Candidates are unrefined CDVAE outputs, which explains the very low correct-space-group
fractions even among structural matches; the 0.00 table entries are rounded, not exactly zero.
The space-group input selects the primitive-cell atom count, without enforcing symmetry.
Composition + Z tries every compatible centering, including in the control, so it returns a
mean of 45 candidates per case versus 20 in the other settings. These different candidate
counts matter when comparing any-match rates.

Mean best RMS for composition + Z is 0.131 over the cases with a match. Some accepted matches
are loose: Rb2S phase III has best RMS 0.419, compared with 0.002 for Mg2Si. Read the match
rates alongside RMS and the space-group fractions, rather than as refined solutions.

Median runtime per case: 126 s for composition + Z (median of per-case `runtime_s`).

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol, high).
Fixed as a result: rejection of patterns with no signal in the 15-90° range the model actually
sees (upstream zeroes 5-15°), removal of the broken background-subtraction option and of the CPU
option (upstream hard-codes CUDA), a strict check that the checkpoint fills every model weight,
and the actual checkpoint path in `results.json`. The reviewer also noted that converting other
wavelengths only moves angles; this is documented below rather than claimed as equivalent.

## Known limitations

- Candidates are unrefined; the paper's workflow refines them before judging a solution.
- At most 20 atoms per cell.
- Only angles are converted between wavelengths. Intensities and peak widths are not corrected
  for wavelength-dependent factors, so Mo, synchrotron, and Q-space data are an approximation of
  the training input rather than an equivalent of it.
- Upstream's optional background subtraction is not exposed: as shipped it expects a 2-D tensor,
  runs after interpolation in this code path, and is hard-coded to CUDA.
- Upstream loads the checkpoint with `strict=False`; `run.py` aborts if any weight is missing or
  unexpected.
