# Inference feasibility triage

Status of every `core_solver` in `data/resources.json` (plus one supporting module) with respect to
a runnable wrapper under `inference/`. "Feasible" means public code and weights, a usable license,
and an inference entry point; it says nothing about quality. Checked 2026-10-04 by reading each
repository, its weights location, and its inference code.

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

## Out of focus

| Model | Reason |
|-------|--------|
| CrySTARNet | Trained on six structure families (perovskites, spinels, ...); weights on Zenodo 17896706. |
| XtalNet | MOF-specific (hMOF); weights on Zenodo 13629658. |
| Xrd2Mof | MOF-specific; weights in the repository. |

## Blocked

| Model | Reason |
|-------|--------|
| RealPXRD-Solver | Training and sampling code only; no released checkpoint. |
| PXRDGen | Code and weights only inside a CodeOcean capsule. |
| XRDiff | No public code or weights. |
| AGAPI-XRD | Agent/web service; its reproducibility repository (atomgptlab/agapixrd) has no license, needs an AGAPI key and external databases, and ships no standalone model. |
| CrystalNet (deep-crystallography) | No license; no released weights found. |
| DeepStruc | Input is a pair distribution function for metal nanoparticles, not a PXRD pattern. |
