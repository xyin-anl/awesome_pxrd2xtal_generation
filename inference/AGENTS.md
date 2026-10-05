# Playbook: adding a runnable inference script

Instructions for an agent (or person) wrapping a new model from `data/resources.json` under
`inference/<id>/`. Read `inference/README.md` first for the directory contract. Work on a branch
and open a pull request; a maintainer reviews and merges.

## 0. Decide whether it is feasible

Stop and open a draft PR or issue that explains why if any of these hold:

- No public code, or no public weights (training from scratch is out of scope).
- Weights sit behind a login, request form, or click-through license.
- The license forbids redistribution of usage instructions or the code is unlicensed (you may
  still proceed for unlicensed code, since nothing is vendored, but flag it).
- The model needs a platform we cannot test (Windows-only, proprietary solver, paid API).

Record the reason in `manifest.yaml` with `status.state: blocked` and `status.reason`.

## 1. Read upstream before writing code

- Find the authors' own inference entry point (script, notebook, or API class) and reuse it.
  Never reimplement the model, tokenizer, or preprocessing if upstream exposes it.
- Pin the upstream commit you tested. Clone it in `setup.sh`; never copy upstream code here.
- Write down exactly what the model is conditioned on and how the training data was produced:
  full profile or picked peaks, 2theta or Q, wavelength, angular/Q window, normalization,
  number of points, and how composition/space group enter the prompt. Put this in the model
  README. Most silent failures come from getting one of these wrong.

## 2. Build the environment

- `environment.yml` with exact pins for everything that matters (torch, numpy, pymatgen, the
  ML framework). Prefer the versions upstream used. Create it from scratch with `setup.sh`.
- Use `inference/_common/pxrd_io.py` for loading patterns, wavelength/Q conversion, and peak
  picking. Extend it instead of writing per-model loaders.
- Download weights in `setup.sh` from the official location and verify a sha256 checksum.

## 3. Reproduce the authors' own example first

Run upstream's documented example (their data, their settings) through your `run.py` and
compare with what the paper or README reports. Record the result in the model README. If it
does not reproduce, do not continue to the benchmark; investigate or mark the model blocked.

## 4. Run the shared benchmark

```bash
python inference/_benchmark/benchmark.py inference/<id> --python <env python> --n-samples 20 --control
```

Declare the input settings the model supports under `benchmark.settings` in the manifest.
The run must complete on every case, and the mismatched-pattern control must score clearly
below the matched setting. If it does not, the wrapper is probably not passing the pattern
through correctly. Run one model's benchmark at a time: GPU memory use varies a lot between
models (deCIFer can take ~19 GB), and out-of-memory failures look like model failures.

Commit a script that reruns the upstream-example check (see `uni3dar/reproduce_upstream.py`)
when it needs more than a single `run.py` command.

## 4b. Independent review

Have a second model review the wrapper against the pinned upstream checkout before marking it
verified (read-only, one pass, narrow brief: input representation, prompt format, upstream call,
weights, README claims). Verify every finding before changing code; test disputed scientific
choices with an extra benchmark setting rather than taking either side on trust. Rerun the
benchmark after fixes that change the model's input.

## 5. Document and open the PR

- `README.md` for the model: quick start, inputs, preprocessing, the upstream-example result,
  the benchmark table, and known limitations.
- Set `status` in the manifest (`verified`, date, hardware) only after steps 3-4 pass.
- Add the run.py/environment/setup links to the resource's `inference` field in
  `data/resources.json` and its id to `inference_order`, then run
  `python3 scripts/catalog.py render && python3 scripts/catalog.py check && python3 -m unittest discover -s tests`.
- The PR description lists: upstream commit, weights + checksum, the example reproduction,
  the benchmark summary (with control), and anything you were unsure about.

## Lessons from earlier wrappers

- **Match the training representation exactly**, not just roughly: Cu Kα1 vs averaged Cu Kα
  shifts peaks by ~0.1° at 90°; primitive vs conventional cell changes the atom list; full
  vs reduced formula breaks prompts. Check against the released training data when available
  (Uni-3DAR's peak positions were confirmed against pymatgen's averaged Cu Kα record by record).
- **Look for silently swallowed errors upstream.** Uni-3DAR builds structures inside a bare
  `except:`; a missing `ase` produced zero candidates without an error.
- **Check which metric the paper actually reports.** XRDSol's repository scores "any of 25 runs"
  while the paper ranks 25 candidates by pattern similarity and scores the top one; reproduce the
  paper's protocol, and report the repository's metric only as a secondary number.
- **Look at how the pattern enters the network.** XRDSol compresses the whole pattern to one scalar
  (`nn.Linear(4500, 1)`), which explains why its output barely changes with a wrong pattern.
- **Refinement tools are instrument-specific.** Ab-PXRD-Solver refines against a fixed Cu Kα
  instrument model, so other radiation must be rejected rather than converted.
- **Do not trust output files named "best"/"match" blindly.** Ab-PXRD-Solver rewrites its Match CIF
  per trial; the wrapper exports the state the pipeline actually selected.
- **Check the benchmark data too.** One "experimental" pdCIF turned out to be a two-phase
  sample, and two files carried a processed wavelength that differed from the source wavelength.

- **CrystaLLM-pi** conditions on the 20 strongest *picked peaks*. Upstream does not check this;
  passing a raw profile silently produces meaningless conditioning. Its `--search_zs` keeps only
  one structure per formula, so loop over Z explicitly when you want many candidates.
- **deCIFer** prompts with the *full-cell* formula as written in `_chemical_formula_sum`, with
  explicit 1s (`K1Ca1C1O3F1`, `Ce4O8`). The reduced formula (`CeO2`) gave 0/24 correct CeO2
  structures versus 22/25 with `Ce4O8`. The checkpoint pickles `__main__.TrainConfig` and does
  not load with torch >= 2.6 defaults; pin torch 2.5.1 instead of patching upstream.
- **Data provenance**: `exp_pxrd_data/decifer/HEO/crystalline_CeO2_BM31.xye` is 2theta at about
  0.2545 Angstrom, not Q, although the upstream example uses a Q-space copy. Check units by
  comparing peak positions with a known phase before trusting any file.
- Loose structure matching can hide symmetry errors (cubic candidates matched tetragonal
  BaTiO3). Always look at the space-group fraction alongside the match rate.
