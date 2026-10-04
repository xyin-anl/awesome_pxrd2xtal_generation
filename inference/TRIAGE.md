# Inference feasibility triage

Status of every `core_solver` in `data/resources.json` (plus one supporting module) with respect to
a runnable wrapper under `inference/`. "Feasible" means public code and weights, a usable license,
and an inference entry point; it says nothing about quality. Checked 2026-10-04 by reading each
repository, its weights location, and its inference code.

## Wrapped and verified

| Model | Directory |
|-------|-----------|
| CrystaLLM-π | `crystallm_pi/` |
| deCIFer | `decifer/` |
| Uni-3DAR | `uni3dar/` |
| Crystalyze | `crystalyze/` |
| DiffractGPT | `diffractgpt/` |
| PXRDnet | `pxrdnet/` |

## Feasible, not yet wrapped

| Model | Weights | License | Inputs | Notes |
|-------|---------|---------|--------|-------|
| XRDSol | git LFS in the repo (`xrdsol/prop_models/mp20/epoch=189-step=5129.ckpt`, 148 MB; downloads) | MIT | composition × Z, **unit cell**, peak list (rendered as Voigt profiles at 1.54056 Å) | CDVAE/DiffCSP-era stack (torch 1.9, PyG 1.7.2, PL 1.3.8); the Crystalyze recipe (torch 1.13.1 + a DimeNet shim) should carry over. Needs the cell, so it completes structures for already-indexed patterns; the benchmark needs a cell input setting (reference cells are available). |
| CrySTARNet | Zenodo 17896706 (`epoch=699-step=246399.ckpt`, `best_xrd.pt`, `best_xrd_tem.pt`) | MIT code, CC-BY-4.0 data/weights | PXRD + composition (TEM optional) | Trained on six structure families (perovskites, spinels, …), predicts fractional occupancies; ~10 min per 10 candidates for a perovskite. Same old stack as XRDSol. The general inorganic benchmark is largely out of its training domain. |
| Ab-PXRD-Solver | bundled in the repo (peak CNN, space-group, Roost density models) | MIT | formula, optional space group | Search-and-refine pipeline (PyXtal sampling, MACE relaxation, GSAS-II refinement); CPU-heavy and slow; GSAS-II installation is the main setup risk. |
| XtalNet | Zenodo 13629658 (`XtalNet_ckpt.zip`, 472 MB) | GPL-3.0 code, CC-BY-4.0 data | PXRD + composition (MOFs) | MOF-specific (hMOF-100/400); needs a MOF test case instead of the inorganic benchmark. |
| Xrd2Mof | in the repo (`pretrained_model/`), extra databases on Zenodo | MIT | PXRD of a MOF | MOF-specific (assembles building blocks); needs a MOF test case. |
| OpenAlphaDiffract (module) | Hugging Face `linked-liszt/OpenAlphaDiffract` | BSD-3-Clause | PXRD profile | Predicts crystal system, space group, and lattice, not structures. Needs a symmetry/lattice scorer and a catalog schema change (inference entries are currently limited to `core_solver`). |

## Blocked

| Model | Reason |
|-------|--------|
| RealPXRD-Solver | Training and sampling code only; no released checkpoint. |
| PXRDGen | Code and weights only inside a CodeOcean capsule. |
| XRDiff | No public code or weights. |
| AGAPI-XRD | Agent/web service; repository has no license and no standalone model. |
| CrystalNet (deep-crystallography) | No license; no released weights found. |
| DeepStruc | Input is a pair distribution function for metal nanoparticles, not a PXRD pattern. |

## Suggested order

1. XRDSol: published model, clear license, and the environment recipe already exists.
2. Ab-PXRD-Solver: different method family (search + refinement), so it adds diversity.
3. CrySTARNet, XtalNet, and Xrd2Mof once suitable domain test cases (perovskite/spinel; MOF) are added.
4. OpenAlphaDiffract together with a symmetry/lattice benchmark.
