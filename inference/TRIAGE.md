# Inference feasibility triage

Status of every `core_solver` in `data/resources.json` plus selected structure-producing pipeline
modules with respect to a runnable wrapper under `inference/`. "Feasible" means public code and
weights, a usable license, and an inference entry point; it says nothing about quality. Checked
2026-10-04 by reading each repository, its weights location, and its inference code; updated
2026-10-16 after the catalog refresh and a duplicate search across open/closed issues, pull
requests, and existing wrappers; updated 2026-10-23 after the Gan Jiang v0.1.0 release.
The blocked set and duplicate search were rechecked on 2026-10-30; no blocker changed. Updated
2026-11-06 after Bragglite released weights and issue #15 was opened for a single-phase wrapper;
updated 2026-11-13 after EXPO entered the catalog and Bragglite draft implementation PR #16
opened.

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
| Gan Jiang (agent-orchestrated identification and refinement) | Maintained by its authors as a [v0.1.0 desktop client](https://ganjiang.yangtzeailab.com/) with an [online manual](https://ganjiang.asia/booklet/index.html). The installers are public, but the client uses a closed, account- and credit-metered backend; complete agent source and model weights are not public, so an independent repository wrapper would neither reproduce nor replace the maintained service. |
| EXPO (classical structure solution and Rietveld refinement) | Maintained by its authors as the current [EXPO 2.3.13 application](https://ccorrado71.github.io/expo-docs/download.html), with Windows, Apple-silicon macOS, Ubuntu, source, and Docker distributions plus command-line, no-GUI, automatic-run, and MPI modes. The official application is the runnable interface, so a second repository wrapper would duplicate it. Noncommercial use is free; commercial use requires a paid license. The migrated [documentation](https://github.com/ccorrado71/expo-docs) is still incomplete. |

## Implementation in review

| Model | Status |
|-------|--------|
| Bragglite | [Issue #15](https://github.com/xyin-anl/awesome_pxrd2xtal_generation/issues/15) is implemented by draft [PR #16](https://github.com/xyin-anl/awesome_pxrd2xtal_generation/pull/16), which wraps the known-composition and element-set single-phase modes and reports a verified upstream reproduction, matched/control benchmark, and clean setup. The PR is not merged, so `main` still has no Bragglite wrapper; maintainer review is pending. |

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
| PXtal | No public code, pretrained encoders, or prepared paired datasets were found on 2026-10-16. |
