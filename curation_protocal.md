# Curation Protocol

This file records how the repository content was generated, checked, and updated. The repository intentionally uses the filename `curation_protocal.md` to match the existing historical file name.

Inference wrappers follow a separate procedure, [`inference/AGENTS.md`](inference/AGENTS.md) (feasibility, upstream fidelity, benchmark, independent review, reruns after shared-code changes, statuses, and machine safety); each inference pass is logged here like a catalog refresh.

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

## August 24 2026 biweekly refresh

14. **Codex primary-source and artifact audit — Aug 24 2026**
    Updated local `main`, searched the official arXiv API over the Aug 10–24 interval, queried the official GitHub repository index, searched publisher/project pages, and compared the full canonical README from `Bin-Cao/awesome-xrd2crystal`. The competing board had no commit after Aug 10 and did not list the candidates below. The only newly submitted/released paper-backed resource found after the catalog cutoff was **PowderLine**; a broader cross-reference audit also exposed six material omissions from earlier 2025–2026 releases.

Material changes:

- Added **PowderLine** as a utility. It expresses Rietveld or single-peak analyses as versioned declarative JSON recipes, executes them through refinement engines, and returns structured results. Its advertised Read the Docs site returned 404, so the catalog links the source repository, which includes local documentation, instead. Sources: https://arxiv.org/abs/2608.17009 and https://github.com/NSLS2/PowderLine.
- Added **CrySTARNet** as a core solver because its official repository and Zenodo record provide runnable code, model checkpoints, and data for PXRD/composition-conditioned structure generation with optional TEM conditioning. No paper or preprint was found, so its >85% top-10 claim is explicitly labeled as repository/Zenodo-reported rather than peer-reviewed. Sources: https://github.com/PKUsam2023/CrySTARNet and https://doi.org/10.5281/zenodo.17896706.
- Added **AIdex-R2** as a pipeline module for joint extinction-group and unit-cell indexing from low-angle reflections. The paper reports ~98.5% top-5 extinction-group accuracy, ~1.44% cell-parameter MAPE, and >90% indexing success under its strongest combined perturbation benchmark. Public code, weights, and benchmark data were not found. Source: https://pubs.acs.org/doi/10.1021/acs.jcim.6c01362.
- Added **MatDiffract** as a pipeline module for vector-retrieval phase identification, Rietveld refinement, and phase quantification. The public service was available, but source code and downloadable benchmark artifacts were not found. Sources: https://arxiv.org/abs/2607.20880 and https://matdiffract.nhepsdc.cn/.
- Added **RADAR-PD** as a pipeline module for X-ray and neutron phase identification using mismatch-tolerant neural screening, lattice nudging, and GSAS-II verification. Sources: https://arxiv.org/abs/2605.12478, https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6, and https://huggingface.co/spaces/Lalityadav07/phase_detection.
- Added **Dara** as a pipeline module for multiple-hypothesis phase identification using peak-matching-pruned tree search and BGMN refinement. The catalog notes that its open package still needs the external refinement engine and a user-supplied structure database. Sources: https://pubs.acs.org/doi/10.1021/acs.chemmater.5c02820 and https://github.com/CederGroupHub/dara.
- Added the **Automated Interpretation Framework (AIF)** as a pipeline module for probabilistic ranking and trust assessment of competing phase interpretations. Its code is public, but the repository declares no license and the workflow depends on Dara, an ICSD-backed database, and CBORG/OpenAI-compatible model access. Sources: https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.76450, https://github.com/hackingmaterials/AIF, and https://doi.org/10.5281/zenodo.21141588.
- Rechecked **XRDiff**, the **invariant lattice-bispectrum predictor**, **XMatcher**, and **XCCP**. Their existing paper/artifact links remained available; no public XRDiff code/data, lattice-bispectrum implementation, or XCCP code release was found.
- Rechecked all 74 unique pre-existing external URLs. No link returned 404 or a transport error. Publisher and Code Ocean 403 responses were treated as anti-bot/inconclusive rather than broken because the corresponding primary pages were independently discoverable.

Candidates reviewed but not added:

- **ED-CSP v2** remained outside scope because it predicts structures from electron-diffraction detector-plane spots rather than PXRD.
- **DiffractScout**, **XRDIO**, and the post-cutoff `xrd-autoencoder` tutorial were classified as utilities but excluded because they were repository-only, peripheral to PXRD-to-structure workflows, or lacked a release/paper and independent validation.
- The `socoolblue/Advanced_XRD_Analysis` repository was not attributed to AIdex-R2 because its README cites a different, unnamed work and provides no link to the AIdex-R2 paper.

Representative discovery queries included:

- official arXiv API queries for `all:"powder X-ray diffraction"`, `all:PXRD`, `all:"X-ray diffraction" AND all:"crystal structure"`, `all:diffraction AND all:"structure prediction"`, and `all:Rietveld`, with submitted-date bounds for Aug 10–24;
- official GitHub API repository searches for `PXRD`, `XRD`, `"powder X-ray diffraction"`, and `diffraction` with `created:>=2026-08-10`;
- exact searches for `PowderLine`, `CrySTARNet`, `AIdex-R2`, `MatDiffract`, `RADAR-PD`, `Dara`, `AIF`, `XRDiff`, `XCCP`, and the invariant lattice-bispectrum predictor;
- publisher searches on ACS, Wiley, Nature, and arXiv for 2026 PXRD indexing, phase identification, refinement, and structure generation.

## September 7 2026 biweekly refresh

15. **Codex primary-source and artifact audit — Sep 7 2026**
    Updated local `main`, searched the official arXiv API over the Aug 24–Sep 7 interval, queried the official GitHub repository index, searched publisher and project pages, and compared the canonical README from `Bin-Cao/awesome-xrd2crystal`. The competing board still had no commit after Jul 14 and supplied no post-cutoff candidate. The date-bounded arXiv search returned six records, of which only **AutoXRD** was a reusable in-scope PXRD resource.

Material changes:

- Added **AutoXRD** as a pipeline module for language-model-planned powder-diffraction analysis with executable FullProf/GSAS-II backends, deterministic physical checks, and preserved refinement trajectories. The paper reports 1,340 runs across ten models, with the mean score decreasing from 61.9/100 on 100 diagnostic tasks to 53.7/100 on 34 end-to-end workflows. Public source and tests are available, but the repository has no declared software license or tagged release; full refinement also requires model API access and separately configured crystallographic backends. Sources: https://arxiv.org/abs/2609.00070, https://github.com/Stephen-SMJ/AutoXRD, and https://stephen-smj.github.io/AutoXRD/.
- Added **XRDBench** as a dataset/benchmark. Its public snapshot contains 100 diagnostic tasks and 34 measured-pattern workflows across 11 task families, including 30 X-ray and four neutron cases, with downloadable XYE/PRM/CIF inputs. Because the published JSON also exposes evaluator oracles, those files must not be mounted in a blind solver workspace. Sources: https://arxiv.org/abs/2609.00070, https://github.com/Stephen-SMJ/XRDBench, and https://stephen-smj.github.io/XRDBench/.
- Corrected the existing machine-learning Rietveld entry to its released name, **RAPID**, and added the official GPL-3.0 repository. The current implementation is Windows-only and requires separate Python 2.7 and 3.11 environments plus AutoFP/FullProf. Sources: https://doi.org/10.1107/S1600576726001494 and https://github.com/DataForgeSci/RAPID.
- Replaced the opXRD web portal, which returned HTTP 500 during this audit, with the stable Zenodo dataset archive while retaining the paper record. Sources: https://doi.org/10.5281/zenodo.14279434 and https://publikationen.bibliothek.kit.edu/1000182521.
- Added PowderLine's newly advertised and working GitHub Pages documentation. Sources: https://github.com/NSLS2/PowderLine and https://nsls2.github.io/PowderLine/.
- Rechecked **XRDiff**, the **invariant lattice-bispectrum predictor**, **XCCP**, **AIdex-R2**, and **MatDiffract**. Their available primary links remained reachable, but no public XRDiff code/data, lattice-bispectrum implementation, XCCP code, AIdex-R2 code/weights/benchmark data, or downloadable MatDiffract implementation/benchmark artifacts were found.
- Rechecked all 90 unique catalogued external URLs. No link returned 404 or a transport error. Ten publisher/Code Ocean links returned HTTP 403 and were treated as anti-bot/inconclusive because their primary records remained independently discoverable; the opXRD HTTP 500 was resolved by linking its official archive.
- Validation passed: `python3 scripts/catalog.py render`; `python3 scripts/catalog.py check` (56 resources); `python3 -m unittest discover -s tests -v` (8 passed); and `git diff --check`.

Candidates reviewed but not added:

- **EXIT** was published in ACS Central Science after the cutoff, but it uses XRD alongside MOF identity to predict adsorption and decomposition properties rather than to infer crystal structure, so it is outside this catalog's task scope.
- **Artifact segmentation using the U-Net architecture for powder X-ray diffraction images** operates on two-dimensional detector images upstream of the catalog's one-dimensional PXRD-to-structure workflow and was therefore excluded.
- `ArchiteuthisDuxDux/pxrd-ml-crystal-structure` contains a detailed project report but no implementation or data. `moobeed/pxrdfc` and `lauraliborio/pxrd-plotter` were effectively empty, and `manuhergo/rietveld-refinement` was only a package stub.
- `cristian-galeazzi/XRD-Plotter` was classified as a utility but excluded because it post-processes GSAS-II CSV exports into publication figures rather than supporting structure inference, refinement, or benchmark evaluation.

Representative discovery queries included:

- an official arXiv API union query for `"powder X-ray diffraction"`, `PXRD`, `Rietveld`, and `diffraction AND "structure prediction"`, bounded to Aug 24–Sep 7;
- official GitHub API repository searches for `PXRD`, `"powder X-ray diffraction"`, `Rietveld`, and `XRD "crystal structure"` with `created:>=2026-08-24`;
- exact GitHub searches for `XRDiff`, `XCCP`, the invariant lattice-bispectrum predictor, `AIdex-R2`, `MatDiffract`, PowderLine, and the catalogued machine-learning Rietveld paper;
- publisher searches on ACS, Nature, Wiley, and IUCr for 2026 PXRD machine learning, indexing, refinement, and structure prediction.

## September 21 2026 biweekly refresh

16. **Codex primary-source and artifact audit - Sep 21 2026**
    Started from a clean checkout and fast-forwarded `main` to `d8f5a7d`, including the merged previous refresh. Searched from the canonical Sep 7 cutoff inclusively because it records a date, not a timestamp. This includes same-day publications that were not captured in the previous pass. The expanded official arXiv query returned 32 records; the competing `Bin-Cao/awesome-xrd2crystal` board still had no commit after Jul 14 and supplied no new candidate. Technical claims below were checked against the original papers and official artifacts, not the competing board.

Material additions and corrections:

- Added **GALAXI** as a `pipeline_module`, not a full structure generator. Its paper benchmark and downloadable 365-phase example are distinguished from the larger hosted library. The score counts profile-similarity groups rather than exact CIF identities. Sources: https://arxiv.org/abs/2609.06908, https://github.com/Szymanski-Group/galaxi, https://doi.org/10.6084/m9.figshare.33360183, and https://galaxi-xrd.com/.
- Added the **GALAXI experimental benchmark** as a `dataset`. The GitHub tree contains 128 XY files and 365 reference CIFs, whereas the paper describes 130 experimental test patterns. Full cohort completeness is therefore not claimed. Filenames expose phase labels and should not be passed unchanged to a blind solver. Source: https://github.com/Szymanski-Group/galaxi/tree/main/examples/pretrained_catalog.
- Added **ChatXRD** as a `pipeline_module` for extracted-peak symmetry/lattice prediction. Recorded simulated/curated-data and best-of-seven-split caveats rather than presenting its classifier accuracy as raw experimental or agent-level accuracy. The official CC-BY-4.0 data archive exists, but no separate code/checkpoint release was found. Sources: https://onlinelibrary.wiley.com/doi/full/10.1002/mgea.70095 and https://doi.org/10.5281/zenodo.19535340.
- Added **ERAF4XRD** as a `utility` for literature-to-PXRD figure/metadata curation, not a crystal solver or numeric pattern dataset. Its BSD-3-Clause source is available and requires model API access; the README warns that package-index installation is not yet available. The paper's benchmark DOI, `10.5281/zenodo.22683615`, returned 404 both at the DOI resolver and Zenodo API, so it was not added as an available dataset. Sources: https://arxiv.org/abs/2609.18583 and https://github.com/niaz60/ERAF4XRD.
- Added **xrdkit** as a `utility` after its Sep 12 initial package release and Sep 21 v0.2.0 release. Checked its package metadata, source readers, COD matching code, and GSAS-II workflow documentation. The entry records external refinement dependencies and starting-model requirements without adding a performance claim. Sources: https://github.com/amirkhesro/xrdkit and https://pypi.org/project/xrdkit/.
- Corrected **XMatcher** to its official Desktop v1.3.0 archive. Removed the GitHub link after repeated HTTP 404 responses from both the repository page and API; cached search results still displayed the former repository and were not treated as current availability. The Figshare record retains Windows/macOS downloads and a CC-BY-4.0 deposit license. Sources: https://arxiv.org/abs/2607.17162 and https://doi.org/10.6084/m9.figshare.32812985.
- Corrected **RRUFF** to its new official portal. The old `rruff.info` URL returned HTTP 200 but redirected to the unrelated Gale Crater database. Removed the unverified current PXRD subset count and noted the new portal's own under-construction warning. Source: https://www.rruff.net/.
- Reverified **Xrd2Mof** after its September installation updates, corrected its year to distinguish Dec 2025 online publication from the 2026 journal issue, and recorded the README's full-data-on-request/CSD access caveat. No new performance claim was inferred from the code update. Sources: https://pubs.acs.org/doi/10.1021/jacs.5c16416 and https://github.com/PKUsam2023/Xrd2Mof.
- Rechecked **XRDiff**, the **invariant lattice-bispectrum predictor**, **XCCP**, **AIdex-R2**, and **MatDiffract** against their paper records and current artifact searches. Their verified official release status did not change. Only these directly reviewed entries and the corrections above received new `verified_at` dates; transport-only checks did not refresh other entries.

Availability and validation boundaries:

- Checked all 96 unique pre-existing external catalog URLs: 82 returned HTTP 200, two returned HTTP 202, eleven returned HTTP 403, and XMatcher returned HTTP 404. A 200 response was not equated with correct content: RRUFF redirected incorrectly, OpenReview presented challenge redirects, and several Nature URLs carried cookie-challenge parameters. Publisher/Code Ocean restrictions remain inconclusive, not evidence that the papers or artifacts disappeared.
- Downloadable archive inventories were inspected through official APIs; large model/data archives were not downloaded, and published inference/refinement pipelines were not executed. Availability checks do not reproduce author-reported performance or establish that hosted inference works.
- Checked all 12 new/replacement external URLs. GALAXI's advertised service failed local HTTPS certificate-chain validation; the source and example-weight records were accessible. The service link is retained as an advertised resource with this explicit caveat, not asserted to be operational. ChatXRD's publisher returned 403 to curl but its full article was retrievable through the web tool; Figshare's landing page returned 202 and its API exposed the weight inventory.
- Validation passed: `python3 scripts/catalog.py render`; `python3 scripts/catalog.py check` (61 resources); `python3 -m unittest discover -s tests -v` (8 passed); and `git diff --check`.

Candidates reviewed but not added:

- **PXRD2Seq** is an intended `pipeline_module`, but its preliminary release currently contains dataset-construction utilities rather than the model, checkpoints, processed data, or exact evaluation manifests. Keep it on the watchlist for the announced model release; do not describe it as a runnable solver. Source: https://github.com/Slosmr/PXRD2Seq.
- `1213718318/MatDiffract` is a new `dataset` candidate containing raw-pattern files, but it has no README, attribution, or declared license, and neither the paper nor the checked official service pages linked it. It was not promoted to an official MatDiffract artifact solely because its name matches. Source: https://github.com/1213718318/MatDiffract.
- **PARA-X** (arXiv:2609.13619), **basis-adaptive texture tomography** (arXiv:2609.17299), and **cross-modal dislocation inference** (arXiv:2609.12713) target spatially resolved microstructure/orientation/strain rather than this catalog's one-dimensional powder-pattern-to-crystal task. **Neutron magnetic-interaction inference** (arXiv:2609.21970) targets a magnetic Hamiltonian, and **SynAgent** (arXiv:2609.18598) targets synthesis control rather than a reusable PXRD solver. These supporting-method candidates were excluded as out of scope.
- **BraggsView** is a plotting/comparison `utility`, not a phase-identification method; **XRD-MAT** is a `utility` with a limited ten-reference-lattice demonstration; **XRD Tools** is a desktop `utility` for overlays and TOPAS input generation. These were not added in this pass: the first two are peripheral to the catalog's inference focus, and numerical validation of the third's generated refinement inputs was not established. Sources: https://github.com/anjulnj/BraggsView, https://github.com/aryan3verma31/XRD-MAT, and https://github.com/EmilJaffal/xrd-tools-app.

Representative discovery queries included:

- official arXiv API query `(all:XRD OR all:PXRD OR all:"powder diffraction" OR all:"X-ray diffraction" OR all:Rietveld) AND submittedDate:[202609070000 TO 202609212359]`, plus a narrower crystal-structure query;
- official GitHub repository searches for `PXRD created:>=2026-09-07` and `(PXRD OR XRD OR diffraction) created:>=2026-09-07`, including both result pages;
- exact paper/artifact searches for GALAXI, ChatXRD, ERAF4XRD, PXRD2Seq, XRDiff, XCCP, AIdex-R2, MatDiffract, and the invariant lattice-bispectrum predictor;
- September 2026 PXRD, indexing, refinement, and machine-learning searches restricted to arXiv, ACS, Nature, Wiley, and IUCr, followed by primary-source cross-reference checks.

## October 4 2026 inference verification pass

17. **Claude Code inference build and Codex review - Oct 4 2026**
    Introduced a common inference layout (`inference/README.md`, `inference/AGENTS.md`): pinned upstream commits and weights, a shared `run.py` interface, a shared PXRD loader/peak picker, and a 12-pattern experimental benchmark with a mismatched-pattern control. Each wrapper was checked against the authors' own example, benchmarked on an RTX 4090, and reviewed against the pinned upstream code by Codex (gpt-5.6-sol and gpt-6-astra); every review finding was verified before it was fixed or rejected.

Material changes:

- Added a verified **CrystaLLM-π** wrapper. The released XRD checkpoints condition on picked peaks; the wrapper picks peaks and converts them to the averaged Cu Kα wavelength used in training.
- Replaced the **deCIFer** script. The previous script prompted with the reduced formula (`CeO2`), which gave 0/24 correct CeO2 structures versus 22/25 with the full-cell formula (`Ce4O8`) that the model was trained on. The previous checkpoint link served a different file; the wrapper now downloads the published `decifer_v1_ckpt.pt` and verifies its checksum.
- Replaced the **Uni3DAR** local and Modal scripts. They passed a raw profile without wavelength conversion, conditioned on distinct elements instead of the primitive-cell atom list, and skipped upstream's composition filtering and ranking. The new wrapper reproduces the paper's MP-20 PXRD-guided result (0.750 top-1 on 40 test records vs 75.08% reported).
- Fixed `utils/parse_cifs.py`: processed wavelengths take precedence, standard uncertainties are parsed, observed columns are preferred over calculated ones, net intensities are not background-subtracted, and neutron data are rejected.
- Recorded that `exp_pxrd_data/decifer/HEO/crystalline_CeO2_BM31.xye` is 2θ at 0.2545 Å (not Q, as upstream's README states), and excluded a two-phase pdCIF (`av5088sup4`) from the benchmark.
- Replaced the **Crystalyze**, **DiffractGPT**, and **PXRDnet** scripts with wrappers in the same layout. Crystalyze now converts wavelengths and runs on current GPUs (torch 1.13.1 with a documented torch-geometric shim); 5 of the paper's 8 example compounds give raw matches before refinement. DiffractGPT's weights were re-uploaded in October 2025 and now expect peak-list prompts with JARVIS-style formulas, which the old script did not produce; the wrapper reproduces upstream's own inference but its output barely depends on the pattern on our benchmark, so it is marked `reproduced` rather than `verified`. PXRDnet now calls upstream's experimental pipeline directly, with primitive-cell atom counts as upstream uses.
- Added `inference/TRIAGE.md` with the feasibility status of every other catalogued solver (XRDSol, Ab-PXRD-Solver, CrySTARNet, XtalNet, Xrd2Mof, and OpenAlphaDiffract feasible; RealPXRD-Solver, PXRDGen, XRDiff, AGAPI-XRD, CrystalNet, and DeepStruc blocked or out of scope).
- Codex (gpt-6-astra) drafted the benchmark sections of the Crystalyze, DiffractGPT, and PXRDnet READMEs from the benchmark reports; every number was checked against the JSON before committing.

## October 5 2026 biweekly refresh

18. **Codex primary-source and artifact audit - Oct 5 2026**
    Started from a clean checkout and fast-forwarded `main` to `07c087f`, including the merged Sep 21 refresh. Searched inclusively from the catalog's Sep 21 date because the cutoff has no timestamp. The official arXiv searches returned five broadly matched records; two became catalog entries and one withdrawn benchmark remains on the watchlist. The competing `Bin-Cao/awesome-xrd2crystal` board still had no commit after Jul 14 and supplied no new candidate. Technical claims were checked against papers, official repositories, publisher metadata, and official archives rather than copied from discovery indexes.

Material additions and corrections:

- Added **PhiGen** as a `pipeline_module` for generative crystallographic phasing from reflection amplitudes. The powder result is zeolite-specific: the preprint reports framework-map recovery for 84.2% of 1,854 held-out simulated 3 A datasets versus 1.0% for Superflip, plus two experimental phase-seeding demonstrations. No public code, data, or model was found. Source: https://arxiv.org/abs/2609.28987.
- Added **HyPhID** as a `pipeline_module` for crystal-system/space-group classification and ranked phase retrieval. The apparent external test is not raw RRUFF measurement: it comprises 750 augmented simulated patterns generated from 50 RRUFF CIFs. The paper-linked repository supplies code and notebooks but omits large data, has no LICENSE file despite an LGPL-3.0 README claim, and links a Zenodo record that was not public during the audit. Sources: https://arxiv.org/abs/2609.31888 and https://github.com/lab-mids/xrd_classification.
- Added **RealBind** as a `dataset`. It contains 3,288 structures, 1,468 measured PXRD queries, and leakage-controlled prototype/composition splits. The ChemRxiv abstract and release both expose the simulation-to-measurement gap: one contrastive baseline falls from 78.5% Recall@10 on simulated queries to 5.3% on measured queries, while model-free masked correlation reaches 82.8% on the 492-candidate test library. The 1.1.3 Zenodo release includes code, manifests, per-query protocols, expected results, data arrays, and checkpoints; opXRD/RRUFF-derived measured arrays must be rebuilt under their source terms. The bundled 104-query smoke test passed locally. Sources: https://doi.org/10.26434/chemrxiv.15009668/v1 and https://doi.org/10.5281/zenodo.23142866.
- Added the **ERAF4XRD benchmark** as a `dataset` after its formerly unavailable DOI became public on Sep 25. The CC-BY-4.0 archive covers 273 literature documents, extracted figure crops, staged/final JSON records, and analysis outputs. It is a literature-extraction benchmark, not a numeric PXRD-profile corpus. Source: https://doi.org/10.5281/zenodo.22683615.
- Added **PyXplore / PyWPEM** as a `utility`. Its Sep 18 package release was just before the prior cutoff but was missed then. The MIT package supports whole-pattern effect modelling, refinement, simulation, and decomposition; refinement still requires an initial structure/CIF, and author-reported paper benchmarks were not reproduced. Sources: https://arxiv.org/abs/2602.16372, https://github.com/Bin-Cao/PyWPEM, and https://pypi.org/project/PyXplore/.
- Added the Sep 29 MIT implementation for the **invariant lattice-bispectrum predictor**. It includes training/evaluation and inversion code, tests, and exact split IDs, but not datasets, checkpoints, or saved inversion results. Source: https://github.com/atomicarchitects/bispectrum-xrd-ml.
- Corrected **AGAPI-XRD** to the now-live official AtomGPTLab reproducibility repository; it has no declared license and still needs an AGAPI key plus external databases/refinement tools. Added the published journal records for **RADAR-PD** and **XRD-Rust**, and added the peer-reviewed experimental evaluation of **deCIFer**, which recovers Si and CeO2 in four-case testing but exposes limitations for lower-symmetry Fe2O3 and nanocrystalline CeO2. Sources: https://github.com/atomgptlab/agapixrd, https://doi.org/10.1063/5.0345499, https://doi.org/10.1107/S1600576726005273, and https://doi.org/10.1039/d6dd00088f.
- Rechecked **XRDiff**, **Xrd2Mof**, **XMatcher**, **XCCP**, **AIdex-R2**, and **MatDiffract** against their primary records and current artifact searches. Their release status did not materially change. Reworded the **GALAXI** service note as audit-network inconclusiveness rather than evidence of a server certificate defect.

Availability and review boundaries:

- Checked all 119 unique external catalog URLs after editing: 99 returned HTTP 200, three returned HTTP 202, sixteen publisher/DOI endpoints returned HTTP 403, and the GALAXI service produced one TLS connection failure. The 403 responses were treated as anti-bot/inconclusive and cross-checked through primary metadata APIs where relevant. No catalog URL returned 404.
- Large data/model archives and full inference/refinement pipelines were not downloaded or run. RealBind's 2.5 MB release archive and bundled miniature smoke-test data were inspected; the multi-gigabyte arrays and checkpoints were not downloaded. Author-reported performance is not presented as independently reproduced.
- An independent Claude Opus 5.5 review at high effort received the search brief without Codex's candidate list. It independently found RealBind, PhiGen, the lattice-bispectrum code, the ERAF4XRD release, and the AGAPI-XRD/RADAR-PD/XRD-Rust/deCIFer corrections; it also agreed that X2SBench should not be listed while withdrawn. Its more conservative recommendation to omit HyPhID was not followed because a paper-linked public implementation exists, but the licensing, synthetic-evaluation, and unavailable-data limits are explicit. Its recommendations to hold repository-only Phasentic and the unvalidated Afruz workbench were accepted.

Candidates reviewed but not added:

- **X2SBench** (`arXiv:2609.29751`) was withdrawn on Sep 29 because all coauthors' public-posting consent had not been confirmed. Its 591-record prerelease remains live without a declared license, and the repository omits the simulated set and model/inference runners. Revisit only after a consent-cleared repost and licensing update. Sources: https://arxiv.org/abs/2609.29751 and https://github.com/CleverPhysician/X2SBench.
- **Phasentic** is an in-scope `pipeline_module` candidate whose aggregate receipts report 74/200 strict and 99/200 compound-family matches; those totals and Wilson intervals were independently recomputed. It remains a three-day-old repository without a paper or tagged release, does not publish per-case predictions, and needs an external POW_COD database and raw scans. Source: https://github.com/qaemu/phasentic.
- **Afruz PXRD Analyzer** is an in-scope `utility` candidate with a Windows release and synthetic tests, but its own audit states that those fixtures are not experimental scientific certification. It overlaps established refinement tools and was kept off the catalog pending independent experimental validation. Source: https://doi.org/10.5281/zenodo.23119993.
- **PXRD2Seq** remains a preliminary data-construction repository without the announced model, checkpoints, processed data, or exact evaluation manifests. PeakMatch remains a preview with its full database and Rietveld support unfinished. Format converters, plotting-only tools, single-crystal/electron-diffraction methods, and repository stubs were excluded as peripheral, out of scope, or not yet reusable.

Representative discovery queries included:

- official arXiv API queries for `powder diffraction`, `Rietveld`, `phase identification`, and diffraction-conditioned structure determination, bounded to Sep 21-Oct 5;
- official GitHub repository searches for `PXRD`, `powder X-ray diffraction`, `Rietveld`, `XRD crystal structure`, and `phase identification` with post-cutoff creation/update filters;
- exact artifact searches for XRDiff, XCCP, AIdex-R2, MatDiffract, XMatcher, PXRD2Seq, the invariant lattice-bispectrum predictor, PhiGen, HyPhID, X2SBench, ERAF4XRD, and PyXplore;
- 2026 PXRD, indexing, refinement, and structure-prediction searches on arXiv, ACS, Nature, Wiley, IUCr, AIP, RSC, ChemRxiv, Zenodo, PyPI, and Hugging Face, followed by primary-source checks.

Validation passed: `python3 scripts/catalog.py render`; `python3 scripts/catalog.py check` (66 resources); `python3 -m unittest discover -s tests -v` (8 passed); and `git diff --check`.

## October 5 2026 scope correction

19. **Codex full-catalog PXRD scope audit - Oct 5 2026**
    Classified all 66 post-refresh entries by their actual diffraction input rather than their title or possible downstream use, using their cataloged primary records plus targeted source checks for boundary cases. Existing `verified_at` dates were retained when the underlying metadata and links were not reopened. Two Claude Opus 5.5 review sessions with follow-ups checked the classification and corrective diff; their MP-20, CHILI, PhiGen, AutoXRD, opXRD-HKUST, and schema findings were adopted, and the final review reported no blocking issue. High effort was requested and the model was confirmed as `claude-opus-5-5`, but the CLI does not report whether the requested effort was honored. The temporary invocation records were `/tmp/pxrd-scope-audit-claude.json`, `/tmp/pxrd-scope-correction-review-brief.md`, `/tmp/pxrd-scope-correction-review.json`, `/tmp/pxrd-scope-correction-rereview-brief.md`, `/tmp/pxrd-scope-correction-rereview.json`, `/tmp/pxrd-scope-correction-final-review-brief.md`, `/tmp/pxrd-scope-correction-final-review.json`, `/tmp/pxrd-scope-correction-approval-brief.md`, and `/tmp/pxrd-scope-correction-approval-review.json`; each review exited successfully. These files document the delegation, not the scientific evidence, which comes from the primary sources cited below.

Material scope corrections:

- Retained **PhiGen** as a powder-derived pipeline module, not a raw-profile PXRD solver. Its general model consumes indexed structure-factor amplitudes plus a known unit cell and space group. The powder-specific evidence comes from a separately trained 3 A zeolite branch; its simulated benchmark models reflection overlap, while only the two experimental demonstrations first extract amplitudes by Le Bail fitting. The preprint's broader 210-space-group results are not powder results. Source: https://arxiv.org/abs/2609.28987.
- Retained **HyPhID** as direct PXRD because its released workflow consumes one-dimensional simulated powder patterns. Its RRUFF evaluation remains explicitly labeled as simulation from RRUFF CIFs rather than measured-profile testing. Sources: https://arxiv.org/abs/2609.31888 and https://github.com/lab-mids/xrd_classification.
- Removed **DeepStruc** and **diffpy/PDF tools** from the canonical PXRD list. They operate on atomic pair distribution functions from total scattering, an adjacent inverse problem rather than conventional Bragg PXRD. Sources: https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00086e, https://github.com/EmilSkaaning/DeepStruc, and https://www.diffpy.org/.
- Removed **ERAF4XRD** and its benchmark from the canonical PXRD list. They extract figures and metadata from publications and do not provide numerical diffraction-pattern-to-structure tasks. Sources: https://arxiv.org/abs/2609.18583, https://github.com/niaz60/ERAF4XRD, and https://doi.org/10.5281/zenodo.22683615.
- Removed **Perov-5** and **Carbon-24** because the linked artifacts are generic structure-generation benchmarks and no retained PXRD workflow was verified to use them for a powder task. Sources reviewed: https://figshare.com/articles/dataset/Perov5/22705189 and https://huggingface.co/datasets/albertvillanova/carbon_24.
- Removed the repository-local **parse_cifs.py** entry from the external-resource catalog; the tool remains documented beside the runnable inference scripts.
- Retained **MP-20-PXRD** as a direct simulated-PXRD benchmark after confirming that PXRDnet releases precomputed unbroadened patterns in pickled PyTorch tensors. The files carry `.csv` names but are not structure-only CSV records. PXRDGen instead starts from the same MP-20 structures and calculates a different GSAS-II pattern representation. Sources: https://github.com/gabeguo/cdvae_xrd/tree/main/data/mp_20, https://arxiv.org/abs/2406.10796, and https://arxiv.org/abs/2409.04727.
- Reclassified **JARVIS-DFT/JARVIS-XRD** and **COD** as supporting substrate rather than standalone PXRD datasets. JARVIS-DFT provides structures alongside a separate simulation module, while COD is primarily a structure database with optional deposited diffraction content. Sources: https://pages.nist.gov/jarvis/ and https://www.crystallography.net/cod/.
- Reclassified **CHILI-3K/100K** as supporting substrate. Its own Debye-equation XRD channel describes finite-cluster total scattering, but deCIFer uses periodic CHILI-100K CIFs as an out-of-distribution structure set and calculates Bragg PXRD from them. Sources: https://github.com/UlrikFriisJensen/CHILI and https://arxiv.org/abs/2502.02189.
- Replaced the generic **pymatgen XRDCalculator** and GSAS-II links with their direct official documentation/repository pages. Corrected the **opXRD-HKUST and literature** row to the data card's 1,277-pattern composition. The XCCP paper explicitly links that collection plus a separate 22-MPEA CIF repository in its data-availability statement, so both are attached to XCCP while the absence of public source code remains explicit. Sources: https://pymatgen.org/pymatgen.analysis.diffraction.html, https://github.com/AdvancedPhotonSource/GSAS-II, https://huggingface.co/datasets/caobin/opxrd_hkust_expdata, https://github.com/George-JieXIONG/Materials-Dataset/tree/main/XRD-Files, and https://www.nature.com/articles/s41524-026-02015-y.

The corrected canonical catalog contains 59 resources: 44 direct PXRD, four powder-derived, eight multimodal with PXRD, and three supporting substrates. Pair distribution function/total-scattering, single-crystal, electron-diffraction, literature-mining, and generic structure-only resources are now explicitly outside the catalog scope. The schema is version 2 because `scope_relation` is now required, and README rendering now keeps the displayed update date synchronized with `data/resources.json:last_updated`.

Validation passed: `python3 scripts/catalog.py render`; `python3 scripts/catalog.py check` (59 resources); `python3 -m unittest discover -s tests -v` (11 passed); and `git diff --check`.

## October 5 2026 inference follow-up

20. **Claude Code inference wrappers and branch review - Oct 5 2026**
    Extended the inference layout to two more solvers, narrowed the triage to general inorganic materials, and had the whole branch reviewed independently by Codex (gpt-6-astra) and Claude (Fable 5.1); every finding was verified before it was fixed or rejected.

Material changes:

- Added an **XRDSol** wrapper (`reproduced`): the paper's top-1-by-R_cos protocol is reproduced on MP-20, but the model compresses the pattern to one scalar and the mismatched-pattern control is not lower on the experimental benchmark.
- Added an **Ab-PXRD-Solver** wrapper (`limited`, a new status): upstream's example reproduces exactly, but on experimental data most runs exceed memory, fail to index, time out, or crash inside upstream; non-Cu data are rejected because refinement uses a fixed Cu Kα instrument model.
- **AlphaDiffract** is referenced to its authors' own resources (OpenAlphaDiffract code, Hugging Face weights and example script, live demo) instead of being wrapped here; MOF- and structure-family-specific models (XtalNet, Xrd2Mof, CrySTARNet) are recorded as out of focus.
- Cu Kα2 handling now looks at the data: wavelength labels proved unreliable (four benchmark files declared as Kα1 contain the doublet). Stripped peaks are expressed at averaged Cu Kα, the merge window is limited to half the doublet splitting, and the peak-based benchmarks were rerun.
- Benchmark harness: resumed runs are checked against the command and wrapper code, partial (`--only`) runs must go to a separate report, and error entries no longer contain local paths. pdCIF input no longer needs a solved structure, and `_pd_proc_wavelength`-only files are accepted.

## October 16 2026 weekly refresh

21. **Codex primary-source and artifact audit - Oct 16 2026**
    Started from a clean checkout on an up-to-date `main` and searched inclusively from the catalog's Oct 5 cutoff because `last_updated` records a date rather than a timestamp. The official arXiv query returned 23 broad PXRD/diffraction candidates. Publisher searches, official GitHub repository searches, and exact artifact searches were checked against original papers, repositories, release pages, and model/data pages. The competing `Bin-Cao/awesome-xrd2crystal` board had no commit after Jul 14 and supplied no post-cutoff candidate.

Material additions:

- Added **Bragglite** as a `core_solver` and **PXRD-Bench** as its dataset/benchmark. For Bragglite trained on PXRD-Bench, 100 experimental RRUFF patterns with known composition yield a mean 44.2% top-1 match over five seeds, 24.6 percentage points above its strongest retrained prior baseline under strict unrefined structure matching. Source and single-phase corpus-building machinery are public, but pretrained Bragglite/two-phase checkpoints and the experimental/two-phase evaluation inputs are not distributed. The two-phase benchmark is MP-20-derived rather than part of the 4.26-million-structure corpus. The dataset's own contributions are CC BY-NC 4.0, and AFLOW imposes additional bulk-retrieval, repackaging, and nonacademic-use restrictions. Sources: https://arxiv.org/abs/2610.16480 and https://github.com/dongyeop3813/PXRD-Bench.
- Added **AriadneX** as a `pipeline_module` for known-phase assignment and unknown-phase anomaly recovery. MIT source, an approximately 820 MB checkpoint, and 56 measured example patterns are public. Reproducing the examples also requires user-supplied PCD or ICSD structures and a Gurobi license. Sources: https://doi.org/10.26434/chemrxiv.15010300/v1 and https://github.com/lrcfmd/AriadneX.
- Added **XDecomposer** as a `pipeline_module` after its Oct 6 preprint revision and inference-app release. The entry records its current two-to-four-phase design range and that its five-fold RRUFF tests use software-composed mixtures of measured single-phase patterns. The paper calls that benchmark zero-shot while the repository documents a five-fold fine-tuning command, so the exact evaluation path remains ambiguous. Both repositories are MIT-licensed; the paper-linked project publishes a checkpoint and MP500 database, while the Gan Jiang project adds a notebook, FastAPI service, and local browser interface. Mixing coefficients are not treated as quantitative mass fractions because components are max-normalized without scale-factor calibration. Sources: https://arxiv.org/abs/2605.05866, https://github.com/Licht0812/XDecomposer, and https://github.com/DeltaGanjiang/XDecomposer.
- Added **PhaseMatcher** as a `pipeline_module`. Its reported exact-set results use author-constructed one-to-four-phase mixtures, including mixtures generated from RRUFF single-phase patterns rather than raw multiphase observations. All splits contain the same phase identities with different perturbations. No public code, checkpoints, or PhaseMix-135K download were found. Source: https://arxiv.org/abs/2610.05844.
- Added **Gan Jiang** as a `pipeline_module` and **DeltaXRDbench** as its associated dataset. The three headline results are closed-library identification, not open-ended structure solution. The candidate library is drawn from the same sources as the tests (1,159 RRUFF structures versus 1,164 test pairs; 848 combined opXRD/private structures versus 880 opXRD test pairs), but the paper does not document ID-level exclusion or how many of the 848 are opXRD. Public benchmark and component repositories are available, but no complete agent download was public on Oct 16. DeltaXRDbench's 62,044 records include 50,000 generated linear mixtures, exposes answers beside pattern pointers in its manifests, and declares no project or dataset license. Sources: https://arxiv.org/abs/2610.07862, https://ganjiang.asia/, and https://github.com/DeltaGanjiang/deltaxrdbench.
- Added **PXtal** as a `pipeline_module` for simulated PXRD-to-crystal retrieval and transferable representation learning. The catalog distinguishes its simulated benchmark results from experimental validation and notes that the million-pair experiments used 16 GPUs across four nodes. No public code, pretrained encoders, or prepared paired datasets were found. Source: https://arxiv.org/abs/2610.10653.
- Added **kaisetsu** as a `utility`. It is a Japanese-first Python and Windows application for Rietveld/Le Bail/Pawley refinement, pair-distribution-function fitting, and maximum-entropy analysis. The LICENSE file and package metadata specify MIT, although one stale README section says the license is undecided. The repository publishes alpha releases and comparisons with GSAS-II and diffpy.pdffit2, but no paper or independent experimental certification; v0.1.13 fixed five result-affecting defects. Source: https://github.com/kinjo-tohoku/kaisetsu.

Availability audit and unchanged blockers:

- Rechecked 124 catalog URL references representing 114 unique URLs before editing: 105 checks returned HTTP 200, two returned HTTP 202, sixteen publisher endpoints returned HTTP 403, and the GALAXI service produced one TLS failure. The 403 responses were treated as anti-bot/inconclusive rather than broken; no URL returned 404.
- Rechecked **RealPXRD-Solver**, **AGAPI-XRD**, **CrystalNet**, **PXRD2Seq**, **HyPhID**, **XRDiff**, **X2SBench**, and the invariant lattice-bispectrum predictor. Their previously recorded artifact blockers did not change. Existing entries kept their prior `verified_at` dates because the audit did not reopen their full technical claims.
- Large model and dataset archives were not downloaded, and author-reported performance was not presented as independently reproduced.

Candidates reviewed but not added:

- `qiaoqinyu/pxrd-space-group-predictor` is a molecular-crystal space-group classifier without a software license, released checkpoints/data, or a paper; its repository describes only a synthetic smoke test.
- `milesdougan/crystal_clear` is a one-line repository stub without a paper, license, or release.
- `Dong-li-y/XRD-Rietveld-Skill-MCP` is an unlicensed bundle with unresolved upstream licenses, absent scripts, and only Windows-local validation.
- `owenxyzhang-seseTJ/pxrd-chemistry-refine` is a specialized MOF-oriented skill with synthetic checks but no paper or public experimental validation artifact.
- `jmigartua/diff-ref-struct-site` is only a rendered site for a private macOS Rietveld application; no source, license, paper, or inspectable validation artifact is public.

Representative discovery queries included:

- official arXiv API queries for `powder diffraction`, `PXRD`, `Rietveld`, `phase identification`, and diffraction-conditioned structure determination, bounded inclusively to Oct 5-16;
- official GitHub searches for `PXRD`, `powder X-ray diffraction`, `Rietveld`, `XRD crystal structure`, and `phase identification` with post-cutoff creation/update filters;
- exact searches for PhaseMatcher, Gan Jiang, PXtal, PXRD-Bench/Bragglite, AriadneX, XDecomposer, XRDiff, RealPXRD-Solver, AGAPI-XRD, CrystalNet, PXRD2Seq, HyPhID, X2SBench, and the invariant lattice-bispectrum predictor;
- 2026 PXRD, phase-decomposition, indexing, refinement, and structure-prediction searches on arXiv, IUCr, ACS, Nature, Wiley, and official project pages.

Two independent Claude Opus 5.5 passes checked the diff and primary sources without receiving Codex's conclusions. They confirmed the classifications and headline numbers, found no material literature omission, and identified method, closed-library, licensing, availability, release-link, and blind-evaluation caveats that were corrected above. The reviewers did not download or run the large model/data artifacts; Gan Jiang's candidate-library relationship is reported from aggregate counts rather than an ID-level comparison, and XDecomposer's experimental fine-tuning path remains ambiguous between the paper and repository.

Validation passed: `python3 scripts/catalog.py render`; `python3 scripts/catalog.py check` (68 resources); `python3 -m unittest discover -s tests -v` (11 passed); and `git diff --check`.

## October 16 2026 inference triage

22. **Post-catalog wrapper feasibility and duplicate audit - Oct 16 2026**
    After opening draft catalog PR #11, searched open and closed issues, implementation pull requests, and repository-local wrappers by catalog ID, model name/aliases, and upstream repository. No prior issue or wrapper was found for the new candidates; the only search result was the pending catalog PR itself. No inference issue was opened because no candidate both needed a wrapper and met the feasibility requirements.

- **Bragglite** is blocked because the public repository explicitly marks its pretrained Bragglite and two-phase checkpoints TBA.
- **XDecomposer** is referenced rather than wrapped because its authors already maintain source, checkpoints, an MP500 candidate database, a notebook, a FastAPI service, and a local browser app.
- **Gan Jiang** is blocked because no complete downloadable agent release was public on Oct 16; the official paid-credit site listed the download as coming soon.
- **PXtal** is blocked because no public code, pretrained encoders, or prepared paired datasets were found.
- **PhaseMatcher** was excluded because it identifies/decomposes a set of known phases rather than producing a crystal structure, and its code, checkpoints, and PhaseMix-135K data are not public.
- **AriadneX** was excluded because it assigns known phases and recovers anomaly patterns rather than producing a crystal structure; its authors already publish the runnable workflow, checkpoint, and examples.
- **PXRD-Bench**, **DeltaXRDbench**, and **kaisetsu** are a dataset, dataset, and utility respectively, so they do not qualify for solver-wrapper issues.
- Previously blocked **RealPXRD-Solver**, **AGAPI-XRD**, **CrystalNet**, and **XRDiff** still lacked the artifacts needed to change their status.

## October 23 2026 weekly refresh

23. **Codex primary-source and artifact audit - Oct 23 2026**
    Started from a clean checkout, fast-forwarded `main` to the merged Oct 16 catalog, and searched inclusively from Oct 16 because the catalog cutoff is a date rather than a timestamp. Technical claims were checked against arXiv or publisher pages, official repositories and releases, official data/model records, and the `Bin-Cao/awesome-xrd2crystal` board. That board's latest commit remained Jul 14 and supplied no new candidate.

Material additions and correction:

- Added **diffPXRD** as a `utility`. The Oct 20 arXiv paper introduces a BSD-3-Clause PyTorch engine for differentiable constant-wavelength PXRD and reports more than sevenfold batched-fitting speedup plus a 68.4% reduction in lattice-parameter variability for one low-count operando example. Public source includes synthetic CPU examples and reproduction scripts, but not the raw experimental inputs or large reference arrays; the separate data deposit named in the documentation was not publicly linked when checked. Sources: https://arxiv.org/abs/2610.28564 and https://github.com/AdvancedPhotonSource/DiffPXRD.
- Added **HEXRD / HEXRDGUI** as a multimodal `utility`. The Oct 18 paper covers powder, Laue, and high-energy diffraction microscopy workflows, including whole-pattern fitting. HEXRD 0.10.7, HEXRDGUI 1.2.6, and the 97 MB validation deposit are public under BSD/CC BY terms. Sources: https://arxiv.org/abs/2610.24989, https://github.com/HEXRD/hexrd/releases/tag/0.10.7, https://github.com/HEXRD/hexrdgui/releases/tag/1.2.6, and https://doi.org/10.5281/zenodo.23440009.
- Added **topas-editor / TOPilot** as a multimodal `utility`. The Oct 16 paper demonstrates AI-assisted TOPAS workflows from indexing through structure solution and refinement, phase identification, and total-scattering analysis. The MIT `topas-inp-writer` v1.2.0 skill and editor extension are public, but operation requires separately licensed TOPAS and a configured language-model provider. Sources: https://arxiv.org/abs/2610.23512, https://github.com/johnsoevans/topas-inp-writer/releases/tag/v1.2.0, and https://marketplace.visualstudio.com/items?itemName=johnsoevans.topas-editor.
- Added **QualX** as a `pipeline_module` for database-backed qualitative phase identification after the v1.0.8 release. Source and cross-platform packages are public, and open COD-derived databases are available; PDF-2 remains separately licensed. The repository's `LICENSE` and GitHub detector say GPL-3.0 while the README says LGPL-3.0, so the catalog records the conflict rather than selecting one. Sources: https://github.com/ccorrado71/qualx3, https://github.com/ccorrado71/qualx3/releases/tag/v1.0.8, and https://doi.org/10.1107/S1600576715002319.
- Added **XRD Phase Finder** as a `utility` after its v1.6.6 release. It provides source plus Windows and Apple-silicon macOS packages for interactive database search-match and residual inspection. The official tutorial marks its scores illustrative; no paper or independent validation benchmark was found. Sources: https://github.com/ABKuznetsov/XRD_Analysis_Toolkit and https://github.com/ABKuznetsov/XRD_Analysis_Toolkit/releases/tag/v1.6.6.
- Added the **XRD-CLIP organic retrieval benchmark** as a `dataset`. The Oct 18 Zenodo record contains measured-pattern truth sets, reference-code lists, per-pattern ranks, verification scripts, and an open COD replica. CSD structures, CSD-derived patterns, and the CSD-trained checkpoint are excluded by the CSD license. The associated article was submitted but had no public publisher or preprint page, so the catalog treats the deposit as a dataset rather than promoting XRD-CLIP as a method entry. Sources: https://doi.org/10.5281/zenodo.23446290 and https://github.com/fronkt/xrd-clip.
- Reverified **Gan Jiang** after its advertised launch date. The official release service now publishes v0.1.0 Apple-silicon macOS and x86-64 Windows installers, and the manual documents phase identification, decomposition, refinement, simulation, and visualization. The complete agent source and model weights remain closed; use requires an account, hosted credits, and separate model-token charges. Sources: https://ganjiang.asia/, https://ganjiang.yangtzeailab.com/, and https://ganjiang.asia/booklet/index.html.

Availability audit and unchanged blockers:

- Rechecked all 135 previously cataloged unique URLs: 115 returned HTTP 200, two returned 202, sixteen publisher/DOI endpoints returned anti-bot HTTP 403, the GALAXI site failed local TLS verification, and the Gan Jiang landing page failed `curl` transport but remained readable in an independent browser fetch. No URL returned 404. New links were checked directly through their official APIs or pages.
- Exact repository and release checks found no new usable checkpoint or license for **Bragglite**, **RealPXRD-Solver**, **AGAPI-XRD**, or **CrystalNet**, and no official code/weights repository for **XRDiff** or **PXtal**. Their catalog claims were not reopened, so their `verified_at` dates were intentionally left unchanged.
- Large installers, model files, and dataset archives were not downloaded or executed. Performance statements remain attributed to their authors; no paper result is presented as independently reproduced.

Candidates reviewed but not added:

- `qiaoqinyu/pxrd-space-group-predictor` released one approximately 19 MB molecular-crystal checkpoint on Oct 21, changing its artifact status, but the repository still has no software/model license or public paper and requires separate ChemBERTa, BOMLIP, and MACE components for its full candidate pipeline.
- `gabegraves/phase_eval` is a new Apache-2.0 expert-review workbench with an associated Hugging Face run dataset, but it has no tagged release or citation record and documents major task-building and search components as incomplete.
- `anuarmincache/rie-agent` is an unreleased, paperless BiFeO3-focused GSAS-II agent whose benchmarking is explicitly in progress.
- `fronkt/pxrd-diff` currently exposes only a Phase 0 data pipeline, not the proposed generative model, and has no license.
- `drthyang/materia-platform` and `yue-here/rietx` were active during the interval, but their latest releases predated the Oct 16 cutoff; neither supplied a new publication or release requiring a catalog change this week.

Representative discovery queries included the official arXiv API for `powder diffraction`, `PXRD`, `Rietveld`, `phase identification`, and diffraction-conditioned structure determination over Oct 16-23; GitHub repository and release searches for those terms plus exact candidate names; Zenodo records linked by official projects; and exact blocker searches for every unwrapped core solver. The arXiv interval yielded six broad records: the three added tools above and three material-specific or non-method studies outside catalog scope.

An independent Claude Opus review was requested without receiving Codex's conclusions, but the installed Claude session could not authenticate because its OAuth session had expired. No substitute was represented as cross-family review.

## October 23 2026 inference triage

24. **Post-catalog wrapper feasibility and duplicate audit - Oct 23 2026**
    After opening draft catalog PR #12, searched open and closed issues, pull requests, and repository-local wrappers for `Gan Jiang`, `GanJiang`, `gan-jiang`, both official domains, and the `DeltaGanjiang` organization. The only matches were catalog PRs #11 and #12; no inference issue or wrapper exists.

- No new `core_solver` was added. **QualX** is database-backed phase identification rather than crystal-structure production, and the other new entries are a dataset or utilities, so they do not qualify for inference-wrapper issues.
- **Gan Jiang** was reconsidered because its Oct 21 installers changed the prior availability blocker. It is now listed as author-maintained rather than blocked: the official desktop client and manual are public, but inference remains a closed, account- and credit-metered service without public complete agent source or model weights. An independent wrapper would not reproduce the maintained service, so no `needs-inference` issue was opened.
- **topas-editor / TOPilot** can orchestrate structure-solution workflows but is an author-maintained utility around separately licensed TOPAS, not a cataloged core solver or structure-producing pipeline module. No wrapper issue was opened.
- The new **XRD-CLIP** artifact is a retrieval benchmark and does not produce structures. **diffPXRD**, **HEXRD/HEXRDGUI**, and **XRD Phase Finder** are analysis utilities. None qualifies for an inference wrapper.
- The newly released `qiaoqinyu/pxrd-space-group-predictor` checkpoint did not trigger an issue because the project remains outside the catalog pending a public paper and a software/model license.

## October 30 2026 weekly refresh

25. **Codex primary-source and artifact audit - Oct 30 2026**
    Started from a clean checkout, pruned stale remote branches, and fast-forwarded `main` to the merged Oct 23 catalog. Searches were inclusive of Oct 23 because the catalog records a date rather than a timestamp. Technical claims were checked against official arXiv, publisher, repository, release, data, model, and thesis pages. The competing `Bin-Cao/awesome-xrd2crystal` board still had no commit after Jul 14 and supplied no new candidate.

Material additions and corrections:

- Added **JEPA for PXRD** as a `pipeline_module` from the Oct 23 arXiv submission. The paper evaluates masked joint-embedding predictive pretraining on simulated MP-20 patterns, then freezes the encoder for space-group and lattice probes. Its depth-8 result is reported with the supervised comparisons that limit the claim: the advantage is parameter efficiency, not best absolute accuracy. No public source, weights, or prepared pattern data were found. Source: https://arxiv.org/abs/2610.33310.
- Added the **Mineral-mixture XRPD phase-picking data** as a `dataset`. The CC BY 4.0 deposit contains 37 profiles with 3,001 intensities from 5 to 65 degrees 2theta at 0.02-degree spacing and a workbook of mineral-group weights. A direct workbook check found 4-15 nonzero groups per deposited profile (median 9), with an amorphous component in 8 of 37. Radiation, wavelength, instrument, reference structures, split, and evaluation protocol are absent. A 2025 BYU thesis appears to describe the same 37-profile cohort, but the deposit does not link it, so the relationship remains unconfirmed. Sources: https://doi.org/10.5281/zenodo.23692859 and https://scholarsarchive.byu.edu/etd/11389/.
- Added **diff-ref-struct Web** as a `utility`. The Apache-2.0 Rust/WebAssembly application performs browser-local visualization and analysis of powder-diffraction series. It was first released on Oct 27 by one author, the public repository is a generated mirror of a private upstream repository, and no independent validation benchmark is published. Sources: https://github.com/jmigartua/diff-ref-struct-web, https://github.com/jmigartua/diff-ref-struct-web/releases/tag/v1.1.0, and https://doi.org/10.5281/zenodo.23683238.
- Corrected **GALAXI** and its experimental benchmark after the authors changed the public ground truth. Ten pristine patterns formerly named Li4Ti7O16 were renamed to Li4Ti5O12, and a Li4Ti5O12 CIF and model were added directly to the repository. The live example now has 366 references/models, while the paper and Figshare snapshot use 365. The authors report unchanged group-level results because the two phases fall in the same profile-similarity group, but exact-phase labels and filenames differ from the published snapshot. Source: https://github.com/Szymanski-Group/galaxi.
- Reverified **XRD Phase Finder** at v1.6.7. Its current Windows, macOS, and Linux amd64 packages add manual peak lists, background-aware detection, reversible angle calibration, and measured-range-limited search/reference display. Source: https://github.com/ABKuznetsov/XRD_Analysis_Toolkit/releases/tag/v1.6.7.

Availability audit and unchanged blockers:

- Rechecked 156 unique catalog URLs before editing: 130 returned HTTP 200, two returned 202, eighteen publisher or data endpoints returned anti-bot HTTP 403, five timed out or failed TLS, and the TOPAS Marketplace URL returned 404 to a scripted HEAD request but loaded correctly in a browser. No catalog link was confirmed dead. The GALAXI service and Gan Jiang pages were also checked in an independent browser; service availability is not inferred from a failed audit-network request.
- Exact artifact searches found no newly usable checkpoint for **Bragglite** or **RealPXRD-Solver**, no standalone model for **AGAPI-XRD**, no released weights for **CrystalNet**, and no official public code or weights for **XRDiff** or **PXtal**. Their catalog claims were not fully reopened, so their `verified_at` dates were not changed.
- Large model archives and installers were not downloaded or executed. The small mineral workbook was parsed directly; reported model performance remains author-reported rather than independently reproduced.

Candidates reviewed but not added:

- **PhaseEval** remains excluded. Post-cutoff commits changed its interface, not the prior evidence gap: there is still no tagged release, citation, expert-labeled evaluation set, public candidate library, or public reinforcement-learning wrapper.
- `spotato-1/DSH-XRD` v0.9.1 is an author-maintained Windows DeepSeek Harness plug-in that orchestrates FullProf, lmfit, and GSAS-II, but its release explicitly provides no sample data or test set and no paper or independent scientific validation was found.
- **XRD-AutoAnalyzer** is MIT-licensed source with a submitted but unavailable paper; its tests use synthetic data only, and the README retains a placeholder repository URL. **XRD Phase Analyzer** and **XRD Mixture Generator** are preliminary applications without publication-backed validation.
- **PXRD2Seq** still publishes preliminary data construction only; the model, code, and checkpoints remain planned. `qiaoqinyu/pxrd-space-group-predictor` still has no software/model license or public paper. `Kishan-No1/pxrd-physics-augmentation` offers code only on request and uses a private 20-pattern dataset.
- **DGGP** was excluded because its generative search is not primarily PXRD-conditioned; the deposited beta-S8 neutron example is an application rather than the method's defining input. Teaching applications and material-specific analysis scripts were also outside scope.

Representative discovery queries included official arXiv API searches for `powder diffraction`, `PXRD`, `Rietveld`, `phase identification`, and diffraction-conditioned structure determination over Oct 23-30; GitHub repository, commit, and release searches for those terms plus exact candidate and blocker names; Hugging Face model and dataset searches for PXRD and powder diffraction; Crossref and Zenodo searches for new methods, software, and datasets; and the competing board's commit history.

An independent Claude Opus 5.5 review received the diff and primary-source set without Codex's conclusions. It rejected the proposed PhaseEval addition, identified the exact GALAXI ground-truth consequence, and tightened the JEPA comparison, dataset limitations, and diff-ref-struct maturity caveats. Those findings were verified against the artifacts and incorporated. The requested high reasoning effort could not be confirmed from the command output; the model identity was confirmed.

Validation passed: `python3 scripts/catalog.py render`; `python3 scripts/catalog.py check` (77 resources); `python3 -m unittest discover -s tests -v` (11 passed); and `git diff --check`.

## October 30 2026 inference triage

26. **Post-catalog wrapper feasibility and duplicate audit - Oct 30 2026**
    After opening draft catalog PR #13, searched all open and closed issues, all implementation pull requests, and repository-local wrappers by catalog ID, model name and aliases, paper identifier, and upstream repository. The repository has no issues of either state. Merged PR #9 contains the eight existing wrappers; no duplicate wrapper or implementation pull request was found for the candidates below. No `needs-inference` issue was opened because this refresh added or materially changed no eligible structure-producing model.

- **JEPA for PXRD** predicts space group and lattice parameters from a frozen representation but does not produce a crystal structure. It also has no public code or weights. **GALAXI** identifies phases from a supplied reference library and performs refinement; the ground-truth correction did not turn it into a structure-producing module.
- The new mineral-mixture record is a dataset, and **diff-ref-struct Web** and **XRD Phase Finder** are utilities. None qualifies for an inference-wrapper issue.
- **Bragglite** still marks its pretrained single- and two-phase checkpoints TBA and has no release assets. **RealPXRD-Solver** still has no release or checkpoint file. **AGAPI-XRD** still has no repository license or standalone model. **CrystalNet** still has no project license or released weights; its only repository license file belongs to the bundled PointNet dependency.
- Exact GitHub repository and code searches still found no official public implementation or weights for **XRDiff** or **PXtal**. The **PXRDGen** Code Ocean capsule remained inaccessible to the audit client, and no new public checkpoint location was found. None of these blockers changed, so no existing issue needed reconsideration.

## May 9 2026 detailed record

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

### Main additions from the May 9 2026 pass

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

- Add a runnable inference wrapper for **RealPXRD-Solver** once a checkpoint is released (see `inference/TRIAGE.md`).
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
7. Add or change inference wrappers only by following [`inference/AGENTS.md`](inference/AGENTS.md), and log the pass here.
8. Edit [`data/resources.json`](data/resources.json), which is the canonical source for resource tables; do not edit generated README rows directly.
9. Set `verified_at` on every reviewed resource, update the catalog-level `last_updated` date when content changes, and allowlist only intentionally shared external URLs in `shared_urls`.
10. Run `python3 scripts/catalog.py render`, then `python3 scripts/catalog.py check` and `python3 -m unittest discover -s tests -v` before opening a pull request.

The Nodeology resource board should consume this same catalog rather than maintain a second hand-edited copy. Website-specific presentation fields can remain in the website repository, but factual resource metadata should flow from this repository after review and merge.
