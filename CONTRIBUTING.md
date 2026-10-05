# Contributing

Thank you for helping keep the PXRD-to-crystal catalog accurate and useful.

## Add or update a resource

1. Edit `data/resources.json`, the canonical source of truth. Do not edit rows inside the generated README regions.
2. Confirm that the method, dataset, or scientific tool has a documented and evaluated pathway using a one-dimensional PXRD profile or a representation derived from one. A structure source qualifies only as supporting substrate for a named cataloged PXRD workflow; the ability to simulate PXRD from a generic structure collection is not sufficient by itself.
3. Set the narrowest accurate `scope_relation`:
   - `direct_pxrd` for workflows that accept, produce, or evaluate powder patterns;
   - `powder_derived` when the required user input is a peak list, d-I list, indexed reflections, extracted amplitudes, or another documented powder-derived representation;
   - `multimodal_pxrd` when the resource also covers another measurement technique beyond PXRD but PXRD remains independently usable;
   - `supporting` only for a structure source or generic tool directly used by a named cataloged PXRD workflow.
4. Choose the narrowest accurate category:
   - `core_solver` for full structure generation or solution;
   - `pipeline_module` for symmetry, lattice, retrieval, decomposition, or refinement tasks;
   - `dataset` for training or evaluation corpora;
   - `utility` for simulation, conversion, and workflow tools.
5. Link the primary paper or preprint and the official code, model, or data artifact when available.
6. Record required inputs precisely, including whether the workflow starts from a raw profile or a powder-derived representation and whether formula, composition, lattice, unit cell, space group, or a candidate database is supplied.
7. Attribute performance to the authors and describe the actual evaluation setting. Do not imply that metrics from different datasets or assumptions are directly comparable.
8. Set `verified_at` to the date on which the links and claims were checked. Update the catalog-level `last_updated` date when factual content changes.
9. If no public artifact is found, keep `artifact_links` empty and add a dated `artifact_note` instead of guessing.

Pair distribution function/total-scattering methods, single-crystal or electron-diffraction methods, literature-mining systems, and generic structure datasets with no named PXRD use are out of scope. Record their review in `curation_protocal.md` rather than adding them to the canonical resource list.

## Add or update an inference wrapper

Runnable wrappers live in `inference/<model>/` and follow [`inference/AGENTS.md`](inference/AGENTS.md); `inference/README.md` describes the directory contract and the status definitions.

1. Check feasibility first: public code and weights without a login, a usable license, and a platform that can be tested. Record blocked, out-of-focus, and externally maintained models in `inference/TRIAGE.md`.
2. Reuse upstream's own inference code at a pinned commit, pin and checksum the weights, and use `inference/_common/` for pattern loading and peak picking.
3. Reproduce the authors' own example, run `inference/_benchmark/benchmark.py` with `--control`, and have the wrapper reviewed independently before setting `verified`, `reproduced`, or `limited`.
4. If you change shared code in `inference/_common/`, `utils/parse_cifs.py`, or the benchmark harness, rerun the benchmarks of every wrapper it affects.
5. Add the wrapper's links to the resource's `inference` field and `inference_order` in `data/resources.json`, and log the pass in `curation_protocal.md`.

## Validate the change

Run:

```bash
python3 scripts/catalog.py render
python3 scripts/catalog.py check
python3 -m unittest discover -s tests -v
```

The catalog checker validates schema fields, dates, link roles, duplicate IDs and URLs, local inference paths, and generated README tables.

## Pull request checklist

- Explain why the resource is in scope and whether it is a full solver or a supporting module.
- Identify its `scope_relation` and distinguish raw-profile input from powder-derived input.
- Cite the primary source for each performance claim.
- Identify simulated, experimental, or mixed evaluation data.
- Note access or license restrictions for datasets and model artifacts.
- Keep website-only presentation text out of the canonical catalog; the Nodeology website enriches these records after synchronization.

Corrections to existing entries are as valuable as new additions. If a claim cannot be verified, open an issue with the uncertain text and the best available primary source.
