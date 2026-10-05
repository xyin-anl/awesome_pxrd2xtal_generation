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

## October 5 2026 biweekly refresh

17. **Codex primary-source and artifact audit - Oct 5 2026**
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

18. **Codex full-catalog PXRD scope audit - Oct 5 2026**
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
