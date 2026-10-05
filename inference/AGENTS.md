# Playbook: adding and maintaining runnable inference scripts

Instructions for an agent (or person) wrapping a new model from `data/resources.json` under
`inference/<id>/`, or changing an existing wrapper or the shared code. Read `inference/README.md`
first for the directory contract and the status definitions. Work on a branch; a maintainer
reviews, pushes, and merges. Never push, open a pull request, or merge without the maintainer's
go-ahead.

Every pass ends with a dated entry in `curation_protocal.md` (step 7), so the next agent can see
what was done and why.

## 0. Decide whether it is feasible

Stop, and record why for the maintainer, if any of these hold:

- No public code, or no public weights (training from scratch is out of scope).
- Weights sit behind a login, request form, or click-through license.
- The license forbids redistribution of usage instructions or the code is unlicensed (you may
  still proceed for unlicensed code, since nothing is vendored, but flag it).
- The model needs a platform we cannot test (Windows-only, proprietary solver, paid API).

Record the reason in `manifest.yaml` with `status.state: blocked` and `status.reason`, and in
`inference/TRIAGE.md`.

Scope is general inorganic materials. Models specialized to one material class (MOFs, a few
structure families) go under "Out of focus" in `TRIAGE.md`. When a model's own authors maintain
inference code, weights, and a demo, link those from the catalog entry and list the model under
"Referenced, not wrapped here" instead of duplicating them (AlphaDiffract).

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
The run must complete on every case (if it cannot for reasons inside upstream, see 6b and the
`limited` status), and the mismatched-pattern control must score clearly below the matched
setting. If it does not, the wrapper is probably not passing the pattern through correctly;
check that before concluding the model ignores the pattern. Run one model's benchmark at a time: GPU memory use varies a lot between
models (deCIFer can take ~19 GB), and out-of-memory failures look like model failures.

Commit a script that reruns the upstream-example check (see `uni3dar/reproduce_upstream.py`)
when it needs more than a single `run.py` command.

### Keep the machine alive

The test machine has one 24 GB GPU and 30 GB of RAM, shared with the agent session itself. A
kernel out-of-memory kill takes the session down with the job, so:

- Run every benchmark inside a memory cap, leaving several GB for the system:
  `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=2G <python> inference/_benchmark/benchmark.py ...`.
  A wrapper whose upstream spawns worker processes caps and cleans them up itself (see
  `ab_pxrd_solver/run.py`: a named scope, stopped after every run, with `Result=oom-kill` read
  before stopping).
- Run one GPU benchmark at a time, and do not start a CPU-heavy job next to it if the two caps
  add up to more than the RAM.
- Launch anything longer than a few minutes detached (`nohup setsid ... < /dev/null > log 2>&1 &`)
  and pass `--resume`, so an interrupted run continues where it stopped. `--resume` reuses a case
  only if its command and the wrapper and shared code are unchanged.
- Do not use `pkill -f`/`pgrep -f` with a pattern that also appears in your own command line; it
  matches and kills your own shell. Kill by PID.

### Choose the status

Statuses are defined in `inference/README.md#what-verified-means`:

- `verified`: setup from scratch, the authors' example, the full benchmark, a control that scores
  clearly lower, and a clean review.
- `reproduced`: the same, except the control is not clearly lower. "Clearly" means lower by more
  than the model's run-to-run variation; if in doubt, rerun the matched setting and compare.
- `limited`: setup, example, and review pass, but most benchmark runs cannot complete for reasons
  inside upstream or by design. List every failure and its cause in the README.
- `blocked`: step 0 failed. `untested`: work in progress.

## 4b. Independent review

Have a model other than the author review the wrapper against the pinned upstream checkout before
setting a status. Changes to shared code, or a branch with several wrappers, get two independent
reviewers from different model families, each with the same brief.

- **Brief.** Read-only (no edits, no setup, inference, or benchmarks); the repository path, the
  commits or files in scope, and where the pinned upstream checkouts are; priorities in order:
  scientifically wrong input or output (wavelength, Kα handling, peak picking, cell and Z, formula
  and space-group format, ranking), mismatches with upstream's own code (cite upstream file:line),
  benchmark scoring, crash paths and `setup.sh`, then documentation that contradicts the code or
  `benchmark.json`. Ask for severity, file:line, a concrete failure scenario, the evidence, a
  confidence, and whether the reviewer traced the code path or is inferring; ask it to list what
  it checked and found correct. Exclude style.
- **Verify every finding** by reading the code path or with a minimal test (a simulated pattern,
  a modified input file) before changing anything. Reviewers can be wrong in either direction:
  one claimed an unresolved Kα doublet is picked at its centroid; a simulation showed the maximum
  sits near Kα1, which in turn exposed a real error for data labelled averaged Cu Kα.
- Test disputed scientific choices with an extra benchmark setting or a simulation rather than
  taking either side on trust.
- Summarize the review in the model README ("Independent review": reviewer, what was fixed) and
  list each finding's outcome (fixed, rejected with evidence) in the commit message.
- Rerun the benchmark after fixes that change the model's input (step 6).

## 5. Document and prepare the PR

- `README.md` for the model: quick start, inputs, preprocessing, the upstream-example result,
  the benchmark table, and known limitations.
- Set `status` in the manifest (state chosen as in step 4, date, hardware) only after steps 3-4b.
- Add the run.py/environment/setup links to the resource's `inference` field in
  `data/resources.json` and its id to `inference_order`, then run
  `python3 scripts/catalog.py render && python3 scripts/catalog.py check && python3 -m unittest discover -s tests`.
- The PR description lists: upstream commit, weights + checksum, the example reproduction,
  the benchmark summary (with control), and anything you were unsure about.
- Commit one wrapper or one logical fix at a time. The message says why, with the evidence
  (numbers before and after, the test that showed it). Commit `benchmark.json` together with the
  README numbers taken from it, check that the report contains no local paths, and run the catalog
  checks above before every commit.

## 6. Changing shared code: rerun what it touches

A change to `inference/_common/`, `utils/parse_cifs.py`, or the benchmark harness can move every
wrapper that uses it.

1. Find the affected wrappers (`grep -l pick_peaks inference/*/run.py`, and so on) and check
   whether the change alters what they feed their models: compare the prepared inputs (peak
   lists, profiles) before and after on the benchmark files.
2. If inputs change, rerun those wrappers' benchmarks one at a time, moving the old run
   directories aside.
3. Compare the new and old reports case by case, not only the summaries. A change is real if it
   follows the cases whose input changed; if cases with unchanged input move as much, it is
   run-to-run noise (XRDSol's GPU sampling is not bit-reproducible). Say which in the commit.
4. Update every README number from the new report and recheck the README's per-case claims;
   revisit the status if the control comparison changed.
5. A metric may be added to an old report without rerunning only when the stored data fully
   determine it (top-1 from stored ranked candidate rows), and only if every existing summary
   field recomputes identically.

## 6b. Diagnosing a model that fails on the benchmark

Before spending hours on a model that mostly fails, establish whether the fault is in the
wrapper or upstream:

1. Inspect the exact file the model receives (the wrapper writes it to the output directory).
2. Score the reference structure against that input with upstream's own figure of merit. If it
   scores as high as on upstream's own example, the input is fine.
3. Check the benchmark data for the case: radiation and Kα2, extra phases, impurity peaks.
4. Find the phase where upstream stops (indexing, memory, a crash) from its own logs, and
   measure memory over time under a cap if runs are killed.
5. Validate any simulated control before trusting it: a simulated pattern that upstream's
   scoring does not match to its own source structure says nothing about the model.

Do not patch upstream. Document each failure class with counts and evidence in the README and
set the status accordingly.

## 7. Log the pass

Append a dated section to `curation_protocal.md` in the format of the existing inference entries
("## <Month D YYYY> inference ..."): a numbered entry naming who built and who reviewed, then
"Material changes" bullets covering wrappers added or replaced, status changes and why, fixes to
shared code, data problems found, and anything left open. Update the "Open Gaps" list when a gap
is closed.

## Lessons from earlier wrappers

- **Match the training representation exactly**, not just roughly: Cu Kα1 vs averaged Cu Kα
  shifts peaks by ~0.1° at 90°; primitive vs conventional cell changes the atom list; full
  vs reduced formula breaks prompts. Check against the released training data when available
  (Uni-3DAR's peak positions were confirmed against pymatgen's averaged Cu Kα record by record).
- **Peak maxima of a Cu Kα doublet sit at Kα1**, even when the doublet is unresolved: the
  stronger line dominates the maximum. Converting such maxima from Kα1 to averaged Cu Kα cut the
  median position error 2-5x for peaks 0.06-0.1° wide in simulations, compared with treating them
  as already averaged (`pxrd_io.pick_peaks`).
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
  Radiation labels are unreliable: `ks5409BTsup2` declares Cu Kα1 (1.54056 Å) yet shows Kα2
  shoulders, and several Cu files give no wavelength at all.
- **Test the scoring before blaming the search.** For a search-based solver, compute upstream's
  own figure of merit for the reference structure against your prepared input (Ab-PXRD-Solver:
  0.986, as high as its own example). That separates input-preparation errors from search failures.

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
