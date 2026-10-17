# Research positioning and revision of the original abstract

The supplied `aimeloick_abstract_AI_Biomedecine.pdf` was read in full. Its title is *A Concept Backpropagation Approach to Audit Graph Neural Network Representations in Breast Cancer Histopathology*. No reviewer reports were supplied. This is a technical audit of the text and current evidence, not an explanation of why it was rejected.

## What is preserved

The original scientific motivation remains: determine whether a concept detected in a GNN representation has evidence of relevance to the model's computation, beyond an association in a probe. GCExplainer, clinician-facing histology concept explanations and concept backpropagation are relevant foundations. They now appear in the manuscript's introduction, comparison table and bibliography.

## Claims that cannot be carried forward unchanged

| Original statement | Evidence check | Revision |
|---|---|---|
| Cell-level histopathology graphs evaluated on the cited Duke collection of 922 cases | The cited TCIA collection provides breast MRI, not the H&E cell graphs described in the abstract. An additional imaging/histology linkage would require evidence not supplied here. | Use the actual Graph-BRACS/GrapHist release and BRACS source metadata. |
| 82.3% and 79.1% accuracy, with F1 values 0.84 and 0.78 | No matching experiment artifacts are present in this workspace. | Exclude these figures; report the verified seven-class seed-42 experiment. |
| HoVer-Net segmentation and feature extraction performed as part of this work | Current experiments consume precomputed released graphs. | Attribute graph provenance to GrapHist; do not claim this project reran segmentation. |
| Cell autoencoder, probe-guided latent perturbation, reconstruction and feature re-extraction | This implemented pipeline is not present in the current code. | Identify concept backpropagation as related motivation. Describe the actual descriptor-edit protocol. |
| Layer-specific encoding of mitosis and stromal invasion | Current targets are area, circularity, intensity, texture and mean edge distance; they are not independent mitosis/invasion annotations. | Remove the unsupported clinical concept and layer-specific findings. |
| Perturbations prevent large prediction shifts from imperceptible changes | No robustness bound or adversarial experiment supports that guarantee. | Do not claim adversarial robustness or perceptual plausibility. |
| Stronger evidence than GCExplainer plus LENs | There is no shared-protocol head-to-head comparison in current results. | Position the audit as complementary; do not claim superiority. |

The Duke collection is documented by [TCIA](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/). The mismatch does not establish that no other Duke histology data could exist; it establishes that the cited collection does not substantiate the pipeline described in that abstract.

## Candidate contribution and its current evidential level

**Research question:** for input-derived concepts in histopathology cell graphs, what evidence of task-specific reliance survives untrained-network, input-redundancy and realized-dose controls?

**Current contribution:** a patient-disjoint empirical audit combining these controls, with measured examples of (1) strong recoverability without task training, (2) different responses to isolated and multi-descriptor edits, and (3) severe requested/realized-dose mismatch for some features. Source assignments, model evaluation and per-graph results are verified. This is a contextualized empirical contribution, not a new GNN or causal explanation algorithm.

The general observation that decodability differs from reliance is not novel. Parameter-dependence sanity checks are established by Adebayo et al. CAV entanglement and sensitivity to explanatory design, including medical images, are studied by Nicolson et al. The manuscript states those antecedents explicitly rather than claiming an unoccupied general gap.

## Closest work and what a stronger claim would require

| Prior work | Established contribution | Current relationship |
|---|---|---|
| Magister et al., GCExplainer (2021) | Global graph-concept discovery | We audit predefined descriptors. A discovery superiority claim would require an actual comparison. |
| Farace di Villaforesta et al. (2023) | Histology graph concepts and logic-based explanations | This is a close application reference. Our question concerns the evidence for descriptor reliance. |
| Hammersborg and Strumke (2023) | Probe-guided concept backpropagation and visualization | The initial motivation is retained, but its reconstruction procedure is not claimed as implemented. |
| Kim et al., TCAV (2018) | Concept-direction class sensitivity | The completed graph-level CAV/TCAV run evaluates five pooled-embedding targets; adapted patient-block sign-flip reference assumptions remain exploratory and explicit. |
| Nicolson et al. (TMLR 2025) | CAV entanglement, layer inconsistency and spatial dependency | This already studies relevant pitfalls. Cell-graph evidence and dose diagnostics narrow our empirical scope; they do not create a new general principle. |
| Shin et al. (ICML 2023) | Sensitivity of concept-bottleneck intervention procedures | Relevant motivation for specifying edits, with a different model class from our post-hoc GINE. |
| Belrose et al., LEACE (NeurIPS 2023) | Guaranteed linear concept erasure | A possible future representation-level comparison, not a method executed in the current results. |

## Priorities before a demanding full-paper submission

1. **Test the core empirical claim across independent trained models.** Use the same source partitions and predeclared metrics; report effect variability rather than replacing a poor seed with a better one.
2. **Resolve specificity under controlled model dependence.** The exact oracle checks software. The learned synthetic stress failures require a benchmark separating label-level independence, model reliance, nuisance correlation and audit false positives. A noise variable is not automatically an unused feature of a trained model.
3. **Compare edits under comparable realized changes.** Match target displacement and perturbation magnitude where feasible. Determine whether conclusions survive control of the number and type of edited dimensions. Correlation propagation alone does not certify a biological intervention.
4. **Make one stronger methodological or comparative claim only if demonstrated.** A new specificity criterion or admissible-edit procedure would need an operational definition, implementation, appropriate known-ground-truth tests and comparison to relevant alternatives. A collection of known explainers is not sufficient novelty on its own.
5. **Keep the clinical scope accurate.** Pathologist concept validation and held-apart external data are needed for clinical/generalization claims. They are not fabricated requirements already satisfied by the current classifier scores.

An early, cited empirical manuscript can be discussed with an experienced supervisor now. A top-venue acceptance claim is not justified by wording or citations alone. The selected venue, reviewer expectations and the additional evidence above determine the eventual submission scope.

## Deliverables

The manuscript, bibliography, abstract, figures and detailed tables are generated locally under `Latex/`. That folder is excluded from Git by request; the executed experimental notebook and this research-positioning note are the public companion materials.
