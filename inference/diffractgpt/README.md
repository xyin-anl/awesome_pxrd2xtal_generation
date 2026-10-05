# DiffractGPT

Mistral-7B with a LoRA adapter that writes a crystal structure as text from a chemical formula and
a diffraction peak list.
Upstream: [atomgptlab/atomgpt](https://github.com/atomgptlab/atomgpt) (NIST public-domain terms) ·
Weights: [knc6/diffractgpt_mistral_chemical_formula](https://huggingface.co/knc6/diffractgpt_mistral_chemical_formula) ·
Paper: [J. Phys. Chem. Lett. 2025](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.4c03137)

## Quick start

```bash
./setup.sh                      # conda env "diffractgpt", adapter + 4-bit base (~4 GB) at pinned revisions
conda activate diffractgpt
python run.py --pattern scan.xy --wavelength CuKa --composition KCaCO3F --out results/
```

Outputs: `results/candidates/*.cif` and `results/results.json` (including the exact prompt).

| Option | Notes |
|--------|-------|
| `--pattern` / `--peaks` | Raw profile (peak-picked here) or your own `2theta,intensity` CSV (needs `--wavelength`) |
| `--composition` | Required; prompted as JARVIS's reduced formula, as in training. Z and space group are not inputs |
| `--peak-method` | `harness` (default): shared peak picker + the training prompt generator; `upstream`: atomgpt's `load_exp_file` peak selection |
| `--strip-ka2` | `auto` (merge resolved Cu Kα2 satellites for data declared at averaged Cu Kα, or for other Cu data whose profile shows the doublet), `on`, or `off` for monochromated Kα1 (harness method) |
| `--n-samples` | 1 (default) = upstream's greedy decoding; more = sampled structures at `--temperature` (0.7), our addition |

Needs an NVIDIA GPU (bitsandbytes 4-bit).

## How the input is prepared

The adapter on Hugging Face was re-uploaded on 2025-10-24, and the training set published with it
(`knc6/diffractgpt_jarvis_dft`) uses **peak-list prompts**, built by
`atomgpt/scripts/diffractgpt/dataset_atomgpt_spectra2.py:make_diffractgpt_prompt`:

```
The chemical formula is: BH6N.
The XRD pattern shows main peaks at: 24.9°(1.0), 27.1°(0.14), 29.6°(0.26), ....
Generate atomic structure description with lattice lengths, angles, coordinates and atom types.
```

The paper and the previous script in this directory used the older format, a binned intensity
vector, which the current weights were not trained on. `run.py` picks peaks from the measured
pattern (`inference/_common/pxrd_io.py`), converts them to averaged Cu Kα (JARVIS's simulator
default, 1.54184 Å), and applies the same steps as `make_diffractgpt_prompt`: Gaussian sticks
(σ 0.1°) on a 0.1° grid over 0-90°, `find_peaks(height=0.01, distance=1, prominence=0.05)`, and the 20
strongest peaks sorted by angle. The formula is JARVIS's `reduced_formula`, which reproduces all
3,799 formulas of the released test prompts; pymatgen's `reduced_formula` differs for 703 of them
(it groups polyanions, e.g. `Th3(SbAs)2` instead of `Th3Sb2As2`).

Upstream's own experimental loader (`load_exp_file`) selects peaks differently (raw heights,
`height=0.05`, `prominence=0.02`, at least 0.5° apart) and writes the prompt without the training
text's periods. `--peak-method upstream` uses that peak selection with the training template;
both methods are benchmarked below.

The model is loaded with transformers + PEFT on the 4-bit NF4 base, computing in bf16 where the
GPU supports it (fp16 otherwise) as upstream's loader does, and decoded greedily with the Alpaca
template from atomgpt's `TrainingPropConfig`, as upstream's `gen_atoms` does. As in upstream's
parser, a generation with any malformed atom row is rejected rather than truncated.

## Verification (2026-10-04, RTX 4090)

**Upstream example** (`reproduce_upstream.py`, 30 random records of the released test set,
greedy decoding on the authors' own prompts):

| | This wrapper | Upstream loader (atomgpt `FastLanguageModel` + `gen_atoms`) |
|-|--------------|------------------------------------------------------------|
| Parsed structures | 30/30 | 30/30 |
| StructureMatcher match | 0.20-0.30 (4-bit GPU kernels are not bit-reproducible) | 0.20 |
| Lattice MAE a, b, c | 1.11, 0.76, 1.10 Å | 1.14, 0.79, 1.13 Å |

The wrapper reproduces upstream's own inference. Both are well above the paper's lattice MAE
(0.17, 0.18, 0.27 Å), which was reported for the earlier binned-vector model; the current
weights and test split differ from the paper's, so the numbers are not directly comparable.

`run.py`'s prompt builder applied to pymatgen peaks of the answer structures writes the exact
formula line for 30/30 records and recovers 70% of the peaks in the authors' prompts within 0.15°; the rest differ because JARVIS's XRD simulator,
which made the training prompts, computes intensities differently from pymatgen.

**Benchmark.** 12 experimental pdCIFs from `exp_pxrd_data/pxrdnet` (see
`inference/_benchmark/cases.json`), 20 samples per run except for greedy decoding, scored with
`StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)` and fractions over all returned candidates
(invalid candidates count as misses), averaged across cases; no runs failed (full report:
[`benchmark.json`](benchmark.json)).

| Setting | Any match | Mean match fraction | Mean correct-space-group fraction | Mean candidates |
|---------|-----------|---------------------|-----------------------------------|-----------------|
| composition, sampled (harness peaks) | 3/12 | 0.125 | 0.03 | 20 |
| composition, sampled (upstream peaks) | 5/12 | 0.17 | 0.05 | 20 |
| composition, greedy (harness peaks) | 1/12 | 0.08 | 0.00 | 1 |
| control: composition, sampled (harness peaks), another case's pattern | 4/12 | 0.125 | 0.04 | 20 |

The sampled harness setting matches 3/12 cases (Mg2Si, Mg2Sn, and BaTiO3); its
mismatched-pattern control matches the same three plus a loose EuI2 match (fraction 0.10, RMS
0.476), and both have a mean match fraction of 0.125. The control is not
clearly lower: on this benchmark the pattern has little measurable effect on the output, so
matches mostly reflect the composition.

Upstream peaks give 5/12 any-match and a 0.17 mean match fraction, versus 3/12 and 0.125 for
harness peaks. The additional matches, Na2LiAlF6 and LaInO3, each occur in only 0.05 of the
candidates and are loose (best RMS 0.250 and 0.371). Mean best RMS over matched cases is
0.130 with upstream peaks versus 0.010 with harness peaks, over different sets of cases.
These small differences on 12 cases do not establish a reliable advantage for either picker.

Greedy decoding returns one candidate per case and matches only Mg2Si (1/12), itself a loose
match with RMS 0.467 and the wrong space group. Sampling increases the chance of finding a
match, but the low correct-space-group fractions remain a limitation.

Median runtime per case: 19.45 s for composition with sampled harness peaks (median of per-case `runtime_s`).

**Independent review.** Reviewed against the pinned upstream by Codex (gpt-5.6-sol, high).
Fixed as a result: JARVIS formula canonicalization (pymatgen's differed for 703 of 3,799 released
prompts), strict parsing that rejects any malformed atom row as upstream does, a `--strip-ka2`
option, an `--peak-method upstream` option porting `load_exp_file`, and bf16/fp16 compute matching
upstream's loader.

**Status: reproduced, not verified.** The wrapper reproduces upstream's own inference, but the
mismatched-pattern control is not clearly below the matched setting on this benchmark, which is
the repository's criterion for "verified" (`inference/README.md`).

## Known limitations

- Only the reduced formula is given; the model chooses Z and the cell.
- Peak picking is heuristic; see `inference/crystallm_pi/README.md` for its benchmark accuracy.
