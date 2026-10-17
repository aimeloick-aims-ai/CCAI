# Historical verification record

This file preserves dated verification notes from earlier project stages; its older paths and results are historical, not the current experiment state. Generated reports in `Latex/`, private data and experiment outputs are stored locally and excluded from Git.

This record accompanies the 16 October 2026 review. See `Latex/verification.json` for source hashes and recomputed counts.

- Python AST parsing: all current scripts/tests and notebook code cells passed.
- Two regression tests passed: shared-slide rejection and nonzero realized node/edge displacement with clipping and dose-zero identity.
- All 60 historical variant/dose mean-logit and Macro-F1 differences agree between the retained CSV and NPZ; baseline Macro-F1 recomputes to 0.5904367681370221.
- Training on the current local metadata is correctly rejected before fitting: 203 train/test shared slide IDs and 54 validation/test shared slide IDs.
- Inferential aggregation of retained historical runs is correctly rejected for slide overlap.
- The grouped-split generator correctly refuses to overwrite existing files; its additional train/test guard was reviewed in source.
- **Synthetic end-to-end test FAILED.** Positive spatial and negative texture checks at absolute dose 2 passed, but the untrained area probe achieved R²=0.2387425905805875, below the pre-existing threshold of 0.5. The failing assertion was retained. This is an unmet benchmark assumption, not evidence that the clinical audit is validated. Test used CPU, `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`; final synthetic training loss was 0.0067576561. The test and its threshold must be reconsidered scientifically rather than relaxed merely to obtain a green check.
- Synthetic aggregate outputs and failure status are preserved separately under `results/synthetic_validation/`; they are not BRACS measurements.
- LaTeX compiled successfully with Tectonic 0.17.0 into a four-page PDF. All four pages were rendered and visually inspected: tables, two plots, limits and next steps are present, without visible clipping. A local Fontconfig configuration message did not prevent compilation. LaTeX ligatures were made explicit to avoid missing glyphs.
- The report, figure labels, notebook and scientific audit notes were subsequently translated into English. The English export is `Latex/graph_bracs_progress_report.pdf`, with its generated `.tex` source and `Latex/progress_report.template.tex`. It was recompiled to four pages and all pages were rendered for visual inspection. No results or numerical conclusions changed during translation. The earlier French and progress PDFs were superseded; see the current English research manuscript.

- Historical evidence is preserved, not overwritten by corrected code.
- Incomplete/repeated runs and the previous notebook/docs are preserved locally in ignored `archive/`.
- No training checkpoints, datasets or virtual environments have been removed.
- No Git commit, push or public publication has been performed.

## Executed recovery work

- All 4,493 real graph files were loaded and passed the full structural/finite-value/index/metadata preflight. Zero graph errors. Evidence: local `results/reproducible_run/graph_inventory.csv`, `graph_errors.csv`, and `split_manifest.json`.
- Prepared fixed internal slide-grouped partitions: 2,698 / 896 / 899 graphs and 226 / 79 / 81 slides for training/validation/test. All pairwise slide intersections are zero. Patient independence is not established yet.
- Six software/oracle tests passed, including an exact edge-only model, group preservation, realized per-graph diagnostics, scaler provenance and a small training/audit integration fixture. These are software checks, not a full real-data retraining.
- The trained synthetic stress test was repeated using hidden dimension 128, matching the real model rather than the earlier 32D bottleneck. The decoding threshold was unchanged: untrained area R²=0.6456272993 now passes; the spatial positive check passes; the texture-noise check fails. Status and outputs are retained at `results/synthetic_validation_v2/`. Dataset-label independence must not be conflated with zero dependence of a trained classifier.
- The expanded English notebook has 32 cells and was executed with expensive training/audit/collection flags disabled. It contains real preflight outputs and inventory plots, shows both synthetic outcomes and identifies pending real-data results explicitly.
- Added an execution guide and staged commands for this CPU machine. Full real-data retraining and new real-data audits remain pending. Patient metadata download requires the user's BRACS login; no unauthenticated download or invented mapping was used.

## Source metadata received

The BRACS workbook was downloaded by the user and copied into ignored `data/bracs/`. Its `WSI_Information` sheet matches all 4,493 graphs to 150 patients. Original source assignments restore 3,630 training / 312 validation / 551 test graphs, with 280 / 37 / 69 slides and 105 / 15 / 30 patients. No patient or slide crosses these source partitions. The recommended experiment directory is now `results/bracs_source_patient`, prepared with a repeated full graph validation (zero errors). The earlier internal partitions are retained but superseded. Aggregate provenance and source hash are stored locally; individual patient mappings stay excluded from Git.

The new TRAIN-only scaler completed on 3,630 graphs (3,809,089 nodes and 10,022,501 edges), with training IDs saved for verification. The notebook was re-executed against the restored-source experiment. Four pipeline tests passed after adding source-assignment restoration and rejection of shared patients, including the scaler/training/audit integration fixture. Full real-data training is the next user-run stage; no new classifier score is claimed yet.

## Completed real-data training and audit

The user has now completed seed-42 real-data training and audit. Epoch 35 is selected by validation Macro-F1 0.5599223805; training stops at epoch 47. Test accuracy 0.4718693285 and Macro-F1 0.4657100730 agree with the saved predictions on 551 graphs. The checkpoint SHA-256 is `efbdd6b11da2b816daabe078e6894dd8363e9b99b09418680a60bb54e8004c84`.

Result collection has completed. The PDF builder now selects this completed source-restored experiment. It checks all nine pairwise ID/group intersections, ordered sample IDs and labels, split/checkpoint hashes, best-epoch history, dose-zero identity, and all 60 variant/dose cells (F1/logit/realized dose/clipping). Verified aggregate tables are exported under `Latex/tables/`.

Patient bootstrap is recomputed without a model rerun using saved predictions: 2,000 whole-patient resamples, seed 42, fixed seven labels. The 95% interval for Macro-F1 is 0.3883013984-0.5149106710. Paired patient intervals for F1 changes use identical resampling multiplicities for baseline and edited predictions. These are nominal exploratory intervals conditional on one trained model. A new regression test confirms vectorized confusion-count bootstrap calculations agree with explicit sklearn resampling, including unequal patient sizes and absent classes.

The English report and experimental notebook now describe executed results, rather than pending retraining or missing patient mapping. The earlier report source, PDF and verification snapshot are preserved under ignored `archive/pre_completed_run/`. Individual patient IDs and raw experiments remain excluded from Git.

Final checks: all eight software/oracle/integration/bootstrap tests passed. The 33-cell notebook was executed and validated without error outputs, with expensive launch flags disabled. The completed English PDF compiled to four pages and all pages were rendered and visually checked. Intervention plots include the verified zero-dose identity point and paired nominal patient-bootstrap bands.

## Notebook/report figure synchronization

The notebook previously used separate plotting code: different training/probe layouts and isolated logit/F1 panels instead of the report's isolated/propagated F1 panels with patient intervals. This has been corrected. The report builder exports each of its three Matplotlib figures as both PDF and PNG; the notebook displays those exact PNG exports after checking run identity and image SHA-256. The notebook was regenerated and fully re-executed. Embedded PNG bytes match the exports exactly for learning curves, decodability and interventions. Inventory plots remain additional notebook-only diagnostics. No model training or audit was repeated.

## CAV/TCAV notebook extension (initial status; subsequently completed)

Added `scripts/run_cav_tcav.py` and notebook section 11. Three numerical tests pass: the weighted Ridge CAV agrees with sklearn Ridge including intercept, unit-direction orientation is correct, Holm adjustment/patient intervals behave as expected, and constant-target thresholds are rejected. A reduced end-to-end run completed using 64 graphs per split, 19 controls and 50 bootstrap draws under `.tools/cav_tcav_smoke`; these are explicitly smoke-test artifacts, not final scientific results. The notebook now has 37 cells and was executed with `RUN_CAV_TCAV=False`. The full 999-control run remains for the user to execute. Graph-mean targets, patient-block randomized controls, reference-test assumptions and missing uncertainty components are documented in the notebook and execution guide.

## Research-manuscript revision from the supplied original abstract

Read the original two-page abstract in full and verified the closest literature against primary publication/arXiv/dataset sources. Its content is treated as historical claims, not new instructions or validated experiments. The unsupported old accuracy/F1, autoencoder, layer/mitosis/invasion and comparative claims were not transferred. The Duke MRI citation's modality mismatch is documented in `ABSTRACT_REVISION_NOTES.md`; no reviewer reasons are inferred.

Generated `graph_bracs_research_draft.tex` and PDF (six pages, including a separate reference page), plus `graph_bracs_abstract.tex` and PDF (one page). Eleven in-text citation keys resolve to eleven entries in `references.bib`; preprints are labelled as such. A concrete research question, narrowly scoped empirical contribution, related-work comparison, mathematical audit definitions, executed results and a discussion replace the tutorial/progress-log narrative. Both PDFs were rendered and visually inspected. Numerical results and the shared figure exports remain unchanged.

At this intermediate snapshot, the 38-cell notebook was executed with CAV/TCAV disabled; the complete run and updated notebook/report are recorded in the later entries below. The former progress PDF was locked by an external viewer during replacement; the authoritative manuscript was subsequently superseded by the classification-results edition.


## Completed CAV/TCAV results incorporated (16 October 2026)

The user completed the full seed-42 CAV/TCAV experiment on all source-restored partitions, with 999 patient-block randomized directions per concept and 2,000 patient-bootstrap draws. The manuscript builder now requires that completed full run, verifies model/partition hashes and ordered IDs, recomputes ten held-out CAV AUCs, checks saved sensitivities against embedding gradients times unit directions, and recomputes all 70 validation/test TCAV/control/reference-value cells and their Holm adjustment. Counts and labels also match the current partitions. Existing intervention checks still pass.

Test CAV AUCs range from 0.9947289157 to 1.0. Minimum unadjusted reference values are 0.143 on test and 0.144 on validation; all 35 adjusted values per partition equal 1.0. These exploratory results do not establish concept-specific directional sensitivity and do not prove concept non-use. Conditional bootstrap intervals omit training and CAV-fit variability.

Updated the manuscript protocol, fourth empirical finding, results, discussion, conclusion and standalone abstract. Added the graph-level AUC table and the exact saved notebook TCAV heatmap, with a caption explicitly distinguishing color from significance. Aggregate CAV/TCAV tables and figure copies are exported under Latex; verification.json records source hashes and checks. The rebuilt manuscript has seven pages including references; the abstract remains one page. Both compiled successfully without overfull boxes or unresolved citations, and all pages were rendered and visually inspected. The executed notebook was preserved; no training, audit or CAV fitting was repeated.


## Professor meeting and editorial revision

Prepared an English one-page meeting brief with three findings, three weaknesses and three decisions. Prioritized known-dependence validation and matched controls before simply increasing the already completed 999 TCAV reference draws. Did not interpret old overlapping partitions as independent replication or modest accuracy as proof of little feature use. Removed internal-process language and the verification-file path from the manuscript. Added an explicitly descriptive even/odd decomposition of requested-dose responses, exported 30 paired-dose rows in Latex/tables/edit_parity.csv, and verified exact reconstruction of positive/negative F1 responses. Distance at unit requested dose: even -0.0738539612, odd +0.0242701610 (rounded in manuscript); unequal realized magnitudes prevent a matched-dose interpretation. No new inference or model computation is claimed.

Draft and review copies were superseded by the classification-results edition recorded below.


## Direct classification check of untrained GINEs (16 October 2026)

Executed `scripts/evaluate_untrained_classification.py` locally against the frozen seed-42 patient-disjoint test. Five random initializations (seeds 42-46) used the checkpoint's training-only scaler and the same first-30-TRAIN-batch BatchNorm calibration as the decodability controls. No weights were updated. A train-majority constant and the trained model were evaluated on identical ordered test graphs. The run completed without rerunning training, audit or CAV fitting.

Untrained accuracy: 0.1397-0.1652; Macro-F1: 0.0362, 0.0864, 0.0846, 0.0616, 0.0362 (mean 0.0615, sample SD 0.0240; range 0.0362-0.0864). The trained classifier achieves 0.4719 accuracy / 0.4657 Macro-F1. The train-majority constant achieves 0.1416 / 0.0354. Shared-resample paired patient bootstrap (30 patients, 2,000 draws) gives trained-minus-mean-random Macro-F1 +0.4042 (95% interval 0.3220-0.4532), and trained-minus-majority +0.4303 (0.3533-0.4734). These comparisons directly show that this trained classifier does not perform like these five randomly initialized classifiers on this split, despite the modest trained-versus-untrained decodability gaps. The result does not train a classifier on frozen random embeddings and does not quantify variation across independently trained GINEs.

Saved ordered predictions, metrics and paired contrasts under `results/bracs_source_patient/seed42/untrained_classification/`. The report builder checked checkpoint/split hashes, ordered IDs, labels, and recomputed all seven accuracy/Macro-F1 entries from prediction arrays; paired contrasts use the same patient-bootstrap draws. Aggregate tables were exported under `Latex/tables/`. Updated manuscript and one-page abstract compiled under `graph_bracs_research_classification_results.*`; manuscript now has eight pages including references. The notebook now contains the completed result tables in section 12. PDFs were rendered and visually inspected.


## Obsolete generated-document cleanup

At the user's request, removed old generated TeX copies and associated compile logs from `Latex/`, including draft/progress/review editions, and removed obsolete PDFs that were not locked. The five previously locked PDF copies were removed after the viewer released them. Retained the latest classification-results manuscript, abstract and professor brief, plus the reusable manuscript template and bibliography. Updated the notebook builder to regenerate only current manuscript/abstract filenames and linked the notebook to the current manuscript.
