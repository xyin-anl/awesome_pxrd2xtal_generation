# Inference feasibility triage

Status of every `core_solver` in `data/resources.json` plus selected structure-producing pipeline
modules with respect to a runnable wrapper under `inference/`. "Feasible" means public code and
weights, a usable license, and an inference entry point; it says nothing about quality. Checked
2026-10-04 by reading each repository, its weights location, and its inference code; updated
2026-10-16 after the catalog refresh and a duplicate search across open/closed issues, pull
requests, and existing wrappers.

## Wrapped (status in each manifest: verified, reproduced, or limited)

| Model | Directory |
|-------|-----------|
| CrystaLLM-π | `crystallm_pi/` |
| deCIFer | `decifer/` |
| Uni-3DAR | `uni3dar/` |
| Crystalyze | `crystalyze/` |
| DiffractGPT | `diffractgpt/` |
| PXRDnet | `pxrdnet/` |
| XRDSol | `xrdsol/` |
| Ab-PXRD-Solver | `ab_pxrd_solver/` |

Scope: general inorganic materials. Models specialized to one material class are recorded under
"Out of focus".

## Referenced, not wrapped here

| Model | Where to run it |
|-------|-----------------|
| AlphaDiffract (crystal system, space group, lattice) | Maintained by its authors: [OpenAlphaDiffract](https://github.com/AdvancedPhotonSource/OpenAlphaDiffract) (training, simulation, and a FastAPI inference app), weights and [`example_inference.py`](https://huggingface.co/linked-liszt/OpenAlphaDiffract/blob/main/example_inference.py) on [Hugging Face](https://huggingface.co/linked-liszt/OpenAlphaDiffract), and a [live demo](https://huggingface.co/spaces/linked-liszt/OpenAlphaDiffract-UI). |
| XDecomposer (multiphase decomposition and phase retrieval) | Maintained by its authors: [paper-linked source, checkpoints, and MP500 database](https://github.com/Licht0812/XDecomposer), plus the Gan Jiang team's [notebook, FastAPI service, and local browser app](https://github.com/DeltaGanjiang/XDecomposer). |

## Out of focus

| Model | Reason |
|-------|--------|
| CrySTARNet | Trained on six structure families (perovskites, spinels, ...); weights on Zenodo 17896706. |
| XtalNet | MOF-specific (hMOF); weights on Zenodo 13629658. |
| Xrd2Mof | MOF-specific; weights in the repository. |

## Blocked

| Model | Reason |
|-------|--------|
| Bragglite | Source/configurations are public in [PXRD-Bench](https://github.com/dongyeop3813/PXRD-Bench), but pretrained Bragglite and two-phase checkpoints are explicitly TBA. |
| RealPXRD-Solver | Training and sampling code only; no released checkpoint. |
| PXRDGen | Code and weights only inside a CodeOcean capsule. |
| XRDiff | No public code or weights. |
| AGAPI-XRD | Agent/web service; its reproducibility repository (atomgptlab/agapixrd) has no license, needs an AGAPI key and external databases, and ships no standalone model. |
| CrystalNet (deep-crystallography) | No license; no released weights found. |
| Gan Jiang | Public component repositories exist, but no complete downloadable agent release was available on 2026-10-16; the official paid-credit site listed the download as coming soon. |
| PXtal | No public code, pretrained encoders, or prepared paired datasets were found on 2026-10-16. |
