# Contributing

Thank you for helping keep the PXRD-to-crystal catalog accurate and useful.

## Add or update a resource

1. Edit `data/resources.json`, the canonical source of truth. Do not edit rows inside the generated README regions.
2. Choose the narrowest accurate category:
   - `core_solver` for full structure generation or solution;
   - `pipeline_module` for symmetry, lattice, retrieval, decomposition, or refinement tasks;
   - `dataset` for training or evaluation corpora;
   - `utility` for simulation, conversion, and workflow tools.
3. Link the primary paper or preprint and the official code, model, or data artifact when available.
4. Record required inputs precisely, including any supplied formula, composition, lattice, unit cell, space group, or candidate database.
5. Attribute performance to the authors and describe the actual evaluation setting. Do not imply that metrics from different datasets or assumptions are directly comparable.
6. Set `verified_at` to the date on which the links and claims were checked. Update the catalog-level `last_updated` date when factual content changes.
7. If no public artifact is found, keep `artifact_links` empty and add a dated `artifact_note` instead of guessing.

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
- Cite the primary source for each performance claim.
- Identify simulated, experimental, or mixed evaluation data.
- Note access or license restrictions for datasets and model artifacts.
- Keep website-only presentation text out of the canonical catalog; the Nodeology website enriches these records after synchronization.

Corrections to existing entries are as valuable as new additions. If a claim cannot be verified, open an issue with the uncertain text and the best available primary source.
