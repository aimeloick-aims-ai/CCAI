# Scientific audit - completed source-restored run

## Completed evidence

The recovered BRACS workbook completely maps the 4,493 available graph IDs to 150 patients. Its source assignments yield 3,630 training / 312 validation / 551 test graphs, from 105 / 15 / 30 patients and 280 / 37 / 69 slides. All pairwise sample, slide and patient intersections are zero. All graph files passed structural and finite-value checks.

A new seed-42 GINE is trained and audited. Validation Macro-F1 is 0.5599223805 at epoch 35; early stopping occurs at epoch 47. Test accuracy is 0.4718693285 and Macro-F1 is 0.4657100730. Patient-cluster bootstrap (2,000 draws, seed 42) gives a 95% interval of 0.3883013984-0.5149106710. This interval is conditional on the trained model and does not measure variation across training seeds.

All 60 variant/dose cells were checked against saved per-graph arrays for Macro-F1 deltas, logit means, realized doses and clipping. Ordered sample IDs and labels, zero-dose identity, partition hashes and the checkpoint hash also agree. Detailed verification tables and the source-hash record are generated locally under the Git-ignored `Latex/` directory.

## Findings

Trained R-squared values are 0.959 (area), 0.953 (circularity), 0.978 (intensity), 0.903 (texture) and 0.957 (mean edge distance). Untrained representations and other raw features already explain substantial target information. Intensity is almost reconstructable from other descriptors (R-squared 0.999); mean distance is better decoded by the untrained baseline (0.989). High decodability alone does not establish learned diagnostic use.

Isolated edge-distance edits reduce Macro-F1 in both directions at requested unit doses. Correlation-propagated cellular edits can also have substantial effects. These variants change different collections of descriptors and cannot be equated biologically. Texture at requested dose -1 clips about 95% of node values on average across graphs, producing a mean actual dose of only -0.359. Requested and actual dose must be distinguished.

## Resolved issues

The patient mapping, source partitions, train-only scaler, real-data retraining and selected real-data audit are now executed. The corrected code checks scaler/training IDs and stores ordered evaluation IDs plus per-graph displacement/clipping. A tensor-view overwrite caused invalid historical zero-dose-displacement logs; historical arrays are retained, not silently rewritten. Repeated checkpoint folders are not treated as independent training seeds.

## Remaining scientific limits

Only one new trained seed is available. Historical overlapping-split scores are not direct comparators. Source-restored test data were previously explored and are not an untouched external confirmation. Probes mix node and graph levels, use input-derived targets and do not save target/prediction pairs needed to calculate new patient-level probe intervals. Random controls are not fully matched to propagated multi-feature edits or edge edits. Nominal patient-bootstrap bands are exploratory rather than simultaneous across all tested cells. Descriptor sensitivity is not biological causality.

The learned synthetic stress benchmark remains unresolved. Its earlier 32D model failed area decoding; the 128D model passes that unchanged threshold but fails the expectation that trained predictions ignore label-independent texture noise. An exact edge-only oracle separately passes and guarantees zero model dependence on unused node descriptors. It validates computation rather than the biological audit or learned-model negative control.

## Historical work and next steps

Earlier profiling, CAV/TCAV, erasure, joint and geometric explorations are described in the supplied prior report. Missing Phase 1/geometric outputs are not treated as verified findings. Ten permuted TCAV directions cannot reach a 5% threshold under add-one counting. Joint interventions were skipped in the new focused run and no new joint result is claimed.

The next priority for stronger stability claims is independent training replication on the same frozen source partitions. Further control design, synthetic diagnosis and biological assessment address different questions; additional execution alone cannot establish concept specificity. A second architecture or external dataset is not needed merely to present this completed preliminary progress report honestly.
