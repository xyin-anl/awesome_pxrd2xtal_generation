# Curation Protocol

This file records how the repository content was generated, checked, and updated. The repository intentionally uses the filename `curation_protocal.md` to match the existing historical file name.

## Historical curation record

1. **OpenAI Deep Research — Jun 8 2025**  
   Generated the initial report and README. Resources were highly relevant, but some gaps were identified. [Full conversation →](https://chatgpt.com/share/68461d44-2e8c-8005-a54f-e4fc7e3e462c)

2. **Anthropic Deep Research — Jun 8 2025**  
   Generated another version of the initial report and README, then refined the output to focus on PXRD-specific models. [Final output →](https://claude.ai/public/artifacts/c47e47fb-55e8-4329-bf47-f602c281517f)

3. **OpenAI O1 Pro — Jun 8 2025**  
   Merged both reports using advanced reasoning. Strategy documented in [this blog post](https://xiangyu-yin.com/content/post_deep_research.html). [Full conversation →](https://chatgpt.com/share/68462b6d-c924-8005-b27b-9315ee87796b)

4. **OpenAI O3 + Web Search — Jun 8 2025**  
   Checked and corrected information and links. Reorganized and simplified tables. [Full conversation →](https://chatgpt.com/share/68462cb8-f7b4-8005-96e9-8b3d255a144f)

5. **OpenAI Deep Research — Jun 8 2025**  
   Performed information verification and missing-information retrieval. [Full conversation →](https://chatgpt.com/share/68467093-dae0-8005-9afc-a0c40f0eb412)

6. **Manual Edit — Jun 2025**  
   Manual verification, correction, and reorganization of content.

## 2026 targeted update pass

7. **OpenAI Deep Research App — May 2026**  
   A new Deep Research pass was run against the public web and the GitHub repository `xyin-anl/awesome_pxrd2xtal_generation`. The pass confirmed that the existing README already covered many of the major 2023–2025 PXRD-to-structure generation models, but it did not surface many new datasets, metrics, or related modules.

8. **GPT-5.5 Pro targeted web pass — May 9 2026**  
   A follow-up search pass focused specifically on addable 2025–2026 information: new full-structure solvers, experimental datasets, benchmark corpora, auxiliary PXRD ML models, search-match systems, phase-decomposition models, refinement models, simulation tools, and evaluation metrics.

9. **Codex web verification pass — May 2026**
   Verified README claims, metrics, repository-local links, and external resource links using web search and source pages. Corrected stale or broken links and adjusted unsupported or outdated dataset sizes, benchmark descriptions, and author-reported metrics.

10. **Manual Edit — May 2026**
   Manual verification, correction, and reorganization of content.

## July 18 2026 competitive refresh

11. **Codex primary-source verification pass — Jul 18 2026**
    Compared this board with `Bin-Cao/awesome-xrd2crystal`, searched primary publication and repository sources for newer full-structure solvers, and added three verified 2026 resources: **Xrd2Mof**, **Ab-PXRD-Solver**, and **XRDiff**. The pass also added a task-oriented entry point, documented a contribution workflow, and preserved explicit caveats around author-reported metrics and missing artifacts.

The three additions were verified against their primary sources:

- **Xrd2Mof** — https://pubs.acs.org/doi/10.1021/jacs.5c16416 and https://github.com/PKUsam2023/Xrd2Mof
- **Ab-PXRD-Solver** — https://arxiv.org/abs/2605.24594 and https://github.com/MaterSim/Ab-PXRD-Solver
- **XRDiff** — https://arxiv.org/abs/2606.14003; public code/data were not found during this pass

The competitor comparison informed presentation and process improvements, but factual entries were independently verified from the original paper, code, and data sources rather than copied from another curated list.

You can see the [full conversation here →](https://chatgpt.com/share/69ff8242-0554-83ea-8c9d-595209e41636)

## July 27 2026 biweekly refresh

12. **Codex primary-source and artifact audit — Jul 27 2026**
    Searched for work released after the July 18 catalog cutoff, compared the current catalog with `Bin-Cao/awesome-xrd2crystal` for discovery, and independently checked candidate claims against publisher, arXiv, official repository, PyPI, and Hugging Face pages. The pass also tested every catalogued external paper, artifact, dataset, and utility link for availability.

Material changes:

- Added **CrystaLLM-π** as a core solver after its maintained repository received post-cutoff updates. The entry records its requirement for pre-picked PXRD peaks and composition/unit-cell stoichiometry rather than implying raw-pattern-only inference. Sources: https://arxiv.org/abs/2511.21299, https://github.com/C-Bone-UCL/CrystaLLM-pi, https://huggingface.co/c-bone/CrystaLLM-pi_Mattergen-XRD, and https://huggingface.co/c-bone/CrystaLLM-pi_Chili100K-XRD.
- Added the missing official **XtalNet** code repository and corrected the method and hMOF-100/hMOF-400 top-10 results. Sources: https://arxiv.org/abs/2401.03862 and https://github.com/dptech-corp/XtalNet.
- Replaced the stale **XRD-Rust** 4–6× summary with the paper's serial-SIMD and eight-thread benchmark results, and added the paper link. Sources: https://arxiv.org/abs/2602.11709, https://github.com/bracerino/xrd-rust, and https://pypi.org/project/xrd-rust/.
- Rechecked **XRDiff**, **XCCP**, and **PhaseDifformer**. Their paper and existing data links remained available, but no official public code/data release was found for XRDiff or PhaseDifformer and no official public code release was found for XCCP.

No post-cutoff arXiv submission or newly created GitHub repository meeting the catalog scope was found. **GraPhAI** and **CrystalX** were reviewed but excluded because their inputs are reflection-level or single-crystal diffraction data rather than PXRD patterns. The competitor board's stale `C-Bone-UCL/CrystaLLM-2.0` and `usnistgov/diffractgpt` links were not imported; current official resources were verified independently.

Representative discovery queries included:

- `site:arxiv.org/abs/2607 "powder X-ray diffraction" crystal structure`
- `site:arxiv.org/abs/2607 PXRD materials machine learning`
- `PXRD crystal created:>=2026-07-18`
- `"powder X-ray diffraction" crystal created:>=2026-07-18`
- `site:github.com "XRDiff" "Powder X-Ray Diffraction"`
- `site:github.com "PhaseDifformer"`
- `site:github.com "XCCP" powder XRD contrastive`

## August 10 2026 biweekly refresh

13. **Codex primary-source and artifact audit — Aug 10 2026**
    Updated local `main` to include the merged July 27 refresh, queried the official arXiv API and GitHub repository index, searched publisher/project pages, and checked `Bin-Cao/awesome-xrd2crystal` for discovery. The competing board had no commit after July 14, so it supplied no post-cutoff candidate. All 65 pre-existing external catalog URLs were rechecked; available links resolved, while publisher and Code Ocean anti-bot responses were treated as inconclusive rather than broken. Only entries whose claims or artifact availability were directly reverified received a new `verified_at` date.

Material changes:

- Added **AGAPI-XRD** as a core solver. It combines DiffractGPT, JARVIS-DFT/COD pattern matching, optional ALIGNN-FF relaxation, and automated GSAS-II/BGMN Rietveld refinement. The catalog distinguishes candidate/lattice-parameter return rates from structural correctness: the paper reports a candidate for 93.8% and valid lattice parameters for 79.7% of 276 RRUFF minerals, while only 229 of 1,000 Alexandria structures matched under `StructureMatcher` in the no-refinement run. Sources: https://arxiv.org/abs/2607.08890, https://github.com/crhysc/agapi_xrd_paper, and https://atomgpt.org/xrd. The paper-linked `atomgptlab/agapi_xrd_paper` URL returned 404; the available first-author repository is linked instead.
- Added **XMatcher** as a pipeline module after its official repository received a post-cutoff update. It is a local, evidence-oriented search-match system with bounded global-angle correction, one-to-one peak assignment, and AutoMix multiphase fitting. The catalog preserves the authors' warning that AutoMix outputs are diffraction-evidence contributions, not quantitative phase fractions. Sources: https://arxiv.org/abs/2607.17162, https://github.com/Asterbin/Asterbin-XMatcher, and https://doi.org/10.6084/m9.figshare.32812985.
- Added the **invariant lattice-bispectrum predictor** as a pipeline module. The preprint predicts an E(3)-invariant reciprocal-lattice descriptor from PXRD and inverts it to lattice vectors; on MP-20, the fixed-architecture comparison reports length MAPE decreasing from 11.18% to 2.44% and angle MAPE from 12.74% to 3.07%. Source: https://arxiv.org/abs/2607.21829. The paper says code/data will be released on publication, but its named GitHub URL returned 404 during this pass.
- Added **XRDStudio** as a utility after its August 2026 release. It is a single-file browser simulator and pattern-comparison tool with CIF import; the entry explicitly notes that it is not a refinement program and omits texture, absorption, and size/strain broadening. Source: https://github.com/shirishchandrakar/XRDStudio.
- Corrected **PhaseDifformer** by adding its official source repository and reverified the paper/repository match. Sources: https://www.nature.com/articles/s41524-026-02087-w and https://github.com/quantumbeam/PhaseDifformer.
- Rechecked **XRDiff** and **XCCP** paper/artifact availability. No official public code/data release was found for XRDiff, and no official public code release was found for XCCP as of Aug 10 2026.
- Corrected the README's stale displayed update date, which had remained at July 18 even though the canonical catalog had advanced to July 27.

Candidates reviewed but not added:

- **ED-CSP** (arXiv:2608.06448) uses electron-diffraction detector-plane spots rather than powder X-ray diffraction, so it is outside this catalog even though it benchmarks against PXRDGen.
- **XRD Fitting Toolkit** (`liuchzzyy/XRD-toolkit`) is a repository-only Windows-oriented GSAS-II wrapper with a hard-coded WC/W2C acceptance recipe and no independent validation; its general refinement capability overlaps the existing GSAS-II entry.
- **x_ray-diffraction-pattern-of-SnS2-crystal-structure** is a one-material analysis notebook rather than a reusable PXRD-to-structure resource.

Representative discovery queries included:

- official arXiv API queries for `all:"powder X-ray diffraction"` and `all:PXRD`, sorted by submission date;
- `site:arxiv.org/abs/2608 PXRD crystal structure`;
- `site:arxiv.org/abs/2608 "powder X-ray diffraction" machine learning crystal`;
- `PXRD crystal created:>=2026-07-27`;
- `"powder X-ray diffraction" crystal created:>=2026-07-27`;
- `XRD crystal structure created:>=2026-07-27`;
- exact GitHub searches for `XRDiff`, `PhaseDifformer`, `XCCP`, `PowderXRD_Project`, `XMatcher`, and `AGAPI-XRD`.

### Search strategy for the May 9 2026 pass

Representative search targets included:

- `PXRD crystal structure determination machine learning 2026`
- `powder X-ray diffraction structure generation diffusion model 2026`
- `XRDSol powder X-ray diffraction crystal structure GitHub`
- `RealPXRD-Solver experimental powder X-ray diffraction crystal structure determination`
- `SIMPOD benchmark PXRD machine learning dataset`
- `AlphaDiffract PXRD space group lattice prediction`
- `XQueryer intelligent PXRD structure identifier`
- `XCCP PXRD contrastive learning space group retrieval`
- `PhaseDifformer multi-phase XRD decomposition`
- `XRD-Rust XRDCalculator powder diffraction simulation`
- `machine learning Rietveld refinement powder X-ray diffraction 2026`

#### Main additions from the May 9 2026 pass

The README was updated with the following new or substantially expanded resources:

- **XRDSol** — equivariant diffusion solver for inorganic crystal structure determination from PXRD; added as a core PXRD-to-structure method.  
Paper: https://www.nature.com/articles/s41467-026-70035-9  
Code/data: https://github.com/ai4mat-zhu/XRDSol

- **RealPXRD-Solver** — generative solver for experimental PXRD, with lattice-conditioned and lattice-free modes and experiment-mimicking augmentation; added as a core PXRD-to-structure method.  
Paper: https://arxiv.org/abs/2603.00965  
Code: https://github.com/liqi-529/RealPXRD-Solver

- **SIMPOD** — COD-derived benchmark with 467,861 structures, simulated 1D PXRD, and radial 2D images; added as a major benchmark dataset.  
Paper: https://www.nature.com/articles/s41597-025-05534-3  
Code/data landing page: https://github.com/BCV-Uniandes/SIMPOD

- **AlphaDiffract** — deep model for crystal system, space group, and lattice parameter prediction from PXRD; added as a related module rather than a full structure generator.
Paper: https://arxiv.org/abs/2603.23367
Open model: https://huggingface.co/linked-liszt/OpenAlphaDiffract
Training code: https://github.com/AdvancedPhotonSource/OpenAlphaDiffract

- **XQueryer** — neural search-match / PXRD structure identification system with real-time diffractometer integration; added as a related retrieval/identification method and dataset source.  
Paper: https://academic.oup.com/nsr/article/12/12/nwaf421/8268901  
Code/data landing page: https://github.com/Bin-Cao/XQueryer

- **XCCP** — contrastive PXRD-candidate matching and space-group identification with elemental pre-screening; added as a related retrieval method and metric reference.  
Paper: https://www.nature.com/articles/s41524-026-02015-y  
Experimental dataset: https://huggingface.co/datasets/caobin/opxrd_hkust_expdata

- **PhaseDifformer** — generative model for single-observation multi-phase XRD decomposition; added as a related front-end for mixture analysis, not a full structure generator.  
Paper: https://www.nature.com/articles/s41524-026-02087-w

- **XRD-Rust** — Rust-accelerated PXRD simulator compatible with pymatgen-style workflows; added to simulation/refinement utilities.  
Paper: https://arxiv.org/abs/2602.11709  
Package: https://pypi.org/project/xrd-rust/

- **Machine-learning automated Rietveld refinement** — convolutional-network refinement workflow; added as a related refinement method and metric reference.  
Article: https://ora.ox.ac.uk/objects/uuid:f25209eb-c034-4ef5-9056-6cab8cf303d1

### Editorial changes from the May 9 2026 pass  

The README was reorganized to avoid conflating different levels of the PXRD pipeline:

- Core PXRD → full-structure generation / solution methods.
- Related modules: lattice/symmetry prediction, retrieval/search-match, phase decomposition, refinement, and simulation.
- Datasets and benchmarks.
- Metrics and evaluation recommendations.
- Simulation, refinement, and utility tools.

### Evaluation guidance added in the May 9 2026 pass  

- top-k structure recovery;
- crystallographic structure matching;
- RMSD / site-wise RMSD / MaxDist;
- lattice MAE and unit-cell errors;
- crystal-system and space-group accuracy;
- PXRD profile similarity metrics including cosine/Rcos, Pearson correlation, SSIM, Wasserstein distance, and peak-overlap scores;
- Rietveld reliability factors such as Rwp, Rp, Rexp, and χ²;
- retrieval top-k hit rate, MRR, confidence, and open-set rejection;
- experimental transfer benchmarks;
- robustness to noise, background, peak broadening, 2θ range, preferred orientation, impurity phases, and scanning step;
- throughput and computational cost;
- validity, uniqueness, novelty, and coverage for generative CIF outputs.

### Known uncertainty / follow-up items  

- Some experimental datasets depend on restricted databases such as ICDD/PDF or ICSD. The README flags restricted or partially restricted resources where relevant.
- For **XCCP**, public data were found, but public source code was not found during the May 9 2026 pass.
- For **PhaseDifformer**, code/data were not found during the May 9 2026 pass.
- For **RealPXRD-Solver**, GitHub code was found. The availability and stability of linked external data/model weights should be rechecked before adding inference scripts.
- Author-reported performance values are not directly comparable because models differ in required inputs, test sets, tolerance thresholds, post-refinement procedures, and experimental-vs-simulated settings.
- The README should keep the distinction between complete structure generation and auxiliary tasks explicit.

### Open Gaps and Suggested Contributions

- Add runnable inference wrappers for **XRDSol** and **RealPXRD-Solver**.
- Normalize input/output schemas across CIF-generating methods: CIF text, pymatgen `Structure`, lattice+fractional coordinates, and PXRD arrays.
- Add a small, open, experimental smoke-test set that can be used without ICDD/PDF/ICSD license restrictions.
- Add a common structure-matching script with configurable tolerances and clear reporting of lattice/composition assumptions.
- Add common PXRD profile metrics: Rcos/cosine, Pearson correlation, Wasserstein distance, peak F1, and post-refinement Rwp/Rp.
- Add robustness tests for 2θ range shifts, wavelength variation, background, noise, preferred orientation, impurity peaks, crystallite-size broadening, and scanning step.
- Track whether each method requires formula, elements, composition, lattice parameters, space group, or unit cell.
- Separate **retrieval/search-match** methods from **open-ended generation** methods in future benchmarks.

### Recommended future curation workflow 

1. Search for new papers and repositories using both broad and named queries.
2. Classify each candidate as core structure solver, related module, dataset, metric, or utility.
3. Verify whether code, weights, and data are public.
4. Record required inputs: PXRD only, composition, formula, lattice parameters, unit cell, space group, candidate database, or experimental metadata.
5. Record whether reported metrics are simulated, experimental, or mixed.
6. Prefer reproducible links to papers, repositories, model cards, data cards, CodeOcean/Zenodo artifacts, and Hugging Face pages.
7. Add inference wrappers only after confirming environment reproducibility and license compatibility.
8. Edit [`data/resources.json`](data/resources.json), which is the canonical source for resource tables; do not edit generated README rows directly.
9. Set `verified_at` on every reviewed resource, update the catalog-level `last_updated` date when content changes, and allowlist only intentionally shared external URLs in `shared_urls`.
10. Run `python3 scripts/catalog.py render`, then `python3 scripts/catalog.py check` and `python3 -m unittest discover -s tests -v` before opening a pull request.

The Nodeology resource board should consume this same catalog rather than maintain a second hand-edited copy. Website-specific presentation fields can remain in the website repository, but factual resource metadata should flow from this repository after review and merge.
