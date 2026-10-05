# XRDSol

Diffusion model that solves atomic coordinates from PXRD for a known cell and composition; the cell stays fixed during sampling.
Upstream: [ai4mat-zhu/XRDSol](https://github.com/ai4mat-zhu/XRDSol) (MIT) ·
Paper: [Nature Communications 2026](https://www.nature.com/articles/s41467-026-70035-9)

## Quick start

```bash
./setup.sh                      # conda env "xrdsol", upstream at the pinned commit, checkpoint (148 MB, git LFS)
conda activate xrdsol
python run.py --pattern scan.xy --wavelength CuKa --composition LuOF --z 2 --spacegroup P4/nmm \
  --cell 3.849,3.849,5.31,90,90,90 --n-samples 25 --out results/
```

Supply the indexed cell for your sample. Outputs: `results/candidates/candidate_001.cif` onwards,
ranked by decreasing R_cos (cosine similarity between each candidate's simulated pattern and the
input pattern), and `results/results.json`, with `r_cos` in each candidate's details.

The checkpoint is upstream's `xrdsol/prop_models/mp20/epoch=189-step=5129.ckpt`;
[`setup.sh`](setup.sh) downloads it into `.weights/mp20/` and verifies SHA-256 checksums for the
checkpoint, configuration, and scalers.

| Option | Notes |
|--------|-------|
| `--pattern` | Measured profile: `.xy`, `.xye`, `.dat`, `.csv`, or pdCIF; mutually exclusive with `--peaks` |
| `--peaks` | Curated peak CSV with header `2theta,intensity`; requires `--wavelength` |
| `--wavelength`, `--x-unit` | Wavelength in Å or a name such as `CuKa`; required for 2θ column files, read from pdCIF. `--x-unit q` accepts Q in Å⁻¹ without a wavelength; default is `2theta` |
| `--composition`, `--z` | Required reduced formula and formula units per conventional cell |
| `--cell` | Required conventional `a,b,c,alpha,beta,gamma`, comma-separated, in Å and degrees |
| `--spacegroup` | Hermann–Mauguin symbol; required to convert cell and Z to the primitive cell unless `--cell-is-primitive` is supplied. R groups accept hexagonal or rhombohedral axes, detected from the cell |
| `--cell-is-primitive` | Declares that both `--cell` and `--z` already describe the primitive cell |
| `--n-samples`, `--seed` | Number of candidates (default 10) and random seed (default 0) |
| `--strip-ka2` | `auto` (default), `on`, or `off`; applies only to peak picking from `--pattern`, not `--peaks` |
| `--model-dir`, `--upstream` | Override `.weights/mp20/` and `.upstream/`; also settable through `XRDSOL_MODEL` and `XRDSOL_DIR` |
| `--out` | Required output directory |

At most 20 atoms per primitive cell (MP-20). Needs a CUDA GPU.

## How the input is prepared

Training uses primitive cells (`hparams.yaml: data.primitive`). Upstream's
`xrdsol/common/data_utils.py:process_one` calculates integrated peak intensities with
`XRDCalculator()`, whose default wavelength is averaged Cu Kα, **1.54184 Å**. The apparent
1.54056 Å argument to `get_voigt_xrd` is unused. Peaks are rendered as Voigt profiles with width
parameter `0.03/cos(θ)` on a 4500-point grid from 0 to below 90° 2θ in 0.02° steps, then normalized
to a maximum of 1.

[`run.py`](run.py) prepares the same representation:

1. For a raw profile, [`inference/_common/pxrd_io.py:pick_peaks`](../_common/pxrd_io.py) subtracts
   background, picks peaks, and estimates integrated intensities from peak height times width.
   Alternatively, `--peaks` supplies a curated peak list directly.
2. `condition_pattern` converts peak positions to 1.54184 Å, keeps peaks inside the model's
   angular window, and calls upstream's `get_voigt_xrd` with the training grid and width.
3. `primitive_cell` converts the supplied cell and Z using the space-group centering. For R
   groups, hexagonal axes are reduced and rhombohedral axes are already primitive. The atom list
   is built from the composition and primitive Z.
4. Upstream's `xrdsol/pl_modules/diffusion.py:sample` generates coordinates with that atom list
   and fixed cell. `run.py:r_cos` simulates each candidate using the training representation and
   ranks the candidates by cosine similarity, following the paper's protocol.

Peak picking from a raw profile is a heuristic adapter: upstream's experimental entry point,
`scripts/solution.py:SampleDataset`, takes curated peak lists. It also passes conventional cell
parameters and Z directly, which is inconsistent with primitive training cells for centered
lattices; this wrapper performs the conversion.

The [environment](environment.yml) follows upstream's Python 3.8, PyTorch Lightning 1.3.8,
torch-geometric 1.7.2, hydra 1.1.0, and pymatgen 2023.5.10. It uses torch 1.13.1 with CUDA 11.7
because upstream's torch 1.9 does not run on current GPUs and drivers; pyxtal 0.6.0 and
pyshtools 4.10.4 provide compatible Python 3.8 wheels. `run.py` explicitly registers upstream's
`scripts/` directory because this repository's own `scripts/` package would shadow it. Every
checkpoint weight is loaded (checked: no missing or unexpected keys); the wrapper checks the
checkpoint and model key sets and aborts if they differ.

## Verification (2026-10-04, RTX 4090, upstream commit `b2144df`)

**Upstream example.** [`reproduce_upstream.py`](reproduce_upstream.py) uses 100 random MP-20
test structures, with inputs built exactly as `process_one` and 25 samples per structure:

| Metric | Match rate |
|--------|------------|
| Top-1 by R_cos (paper protocol) | 0.770 |
| Any of 25 (upstream repository's `scripts/compute_metrics.py` metric) | 0.830 |
| Single sample | 0.480 |

The paper reports 82.3% for its top-1 protocol; the 0.830 any-of-25 result uses a looser metric
and should not be compared to that figure as if it were the same measure.

Pattern dependence during generation is weak. The decoder compresses the entire pattern with
`nn.Linear(4500, 1)` into one scalar per structure, appended to node features only before the
final coordinate layer (`xrdsol/pl_modules/cspnet.py`, lines 137 and 260–290). With shuffled
patterns on MP-20 (`reproduce_upstream.py --shuffle-xrd`), the match rate within 5 samples was
0.65, versus 0.63 with the correct patterns. These results indicate that generation barely depends
on the pattern; its main benefit is through R_cos ranking, from 0.480 for a single sample to
0.770 for the top-ranked candidate among 25.

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
[`inference/_benchmark/cases.json`](../_benchmark/cases.json)), 20 candidates per case, scored
with `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`. Fractions are computed over all returned
candidates and averaged across cases; no runs failed in either setting. Full report:
[`benchmark.json`](benchmark.json), settings `composition_z_cell_spacegroup` and
`composition_z_cell_spacegroup__mismatched_pattern`.

| Setting | Any match | Top-1 match | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|-------------|---------------------|-----------------------------------|-----------------|
| composition + Z + cell + space group | 11/12 | 8/12 | 0.608 | 0.312 | 20 |
| control: same inputs, another case's pattern | 10/12 | 7/12 | 0.592 | 0.321 | 20 |

The control is barely lower: any-match and top-1 each lose one case, and mean match fraction
changes from 0.608 to 0.592. The correct-space-group fraction is slightly higher in the control.
This does not show a clear benefit from the correct diffraction pattern on this benchmark.
Reference cells come from the solved structures, so these results are an upper bound on how well
users with a correct indexed cell would do. The space-group input selects the primitive cell and
atom count; it does not enforce the output symmetry.

Median per-case `runtime_s`: 13.3 s for the matched setting and 13.65 s for the control.

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol, high).
Fixed as a result: the paper's R_cos top-1 protocol in `run.py` and `reproduce_upstream.py`
(the earlier any-of-25 figure used the looser repository metric), explicit R-setting handling,
pymatgen pinned to upstream's 2023.5.10, Cu Kα2 auto-stripping restricted to data declared at
averaged Cu Kα in the shared loader, and manifest wording clarifying that Q input needs no
wavelength. Raw-profile peak picking remains a heuristic adapter and is documented as such.

**Status: reproduced.** The mismatched-pattern control is not clearly lower, so this does not
meet the repository's [`verified` criterion](../README.md#what-verified-means).

## Known limitations

- Requires a known composition, Z, and indexed cell, plus the space group for primitive-cell
  conversion unless the cell and Z are explicitly supplied as primitive. The cell is not refined.
- At most 20 atoms per primitive cell; requires a CUDA GPU.
- Raw-profile peak picking is heuristic. Overlapping or weak peaks and imperfect background
  subtraction can change the conditioning; curated integrated peak intensities are closer to
  upstream's experimental input.
- Wavelength conversion moves peak positions without correcting wavelength-dependent intensities.
- Cu Kα2 stripping is heuristic. `auto` only strips data declared at averaged Cu Kα; data declared
  as Kα1 are treated as monochromated or already stripped. Use `off` for already stripped data or
  `on` to force stripping for Cu profiles. Curated `--peaks` lists bypass this step and must already
  have the intended doublet treatment.
