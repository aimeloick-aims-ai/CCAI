# Executable recovery workflow

Use the project root in PowerShell. The notebook and scientific deliverables are in English.
The following stages preserve old results and never overwrite an existing checkpoint.

## Latest status: seed 42 and random-classification comparison complete

Training, audit and result collection have now completed for `results/bracs_source_patient/seed42`. Do not rerun that seed's training. The executed notebook and local manuscript report validation Macro-F1 0.5599, test Macro-F1 0.4657, with a patient-bootstrap interval of 0.388-0.515. The builder uses the completed source-restored experiment and the local narrative/bibliography under `Latex/`, which is excluded from Git. A direct classification comparison for random-initialized GINEs is in notebook section 12 and was incorporated into the local manuscript and abstract.

The stage commands below remain the reproducibility procedure. For further replication, choose another training seed (41, 43, 44 or 45) while keeping the same frozen partitions. A single seed supports an explicitly preliminary report; do not schedule four more CPU runs automatically just to present the work already completed.

## New optional stage: CAV / TCAV

Notebook section 11 adds graph-level CAV/TCAV on the existing frozen checkpoint. Change `RUN_CAV_TCAV = False` to `True` in that section, execute the launch cell, then execute the two results cells below it. The default run fits 999 patient-block randomized label-flip controls, uses 2,000 patient bootstrap draws, and saves results under `results/bracs_source_patient/seed42/cav_tcav/`. It does not retrain the GINE. Cached graph embeddings/gradients can be reused if an interrupted run is resumed with identical inputs/code/settings; an identical completed run is reused automatically.

Equivalent terminal command:

```powershell
.venv/Scripts/python.exe scripts/run_cav_tcav.py --work-dir results/bracs_source_patient --seed 42
```

All targets are graph means at the pooled graph representation, unlike historical cellular node-level CAVs. TRAIN q20/q80 and a fixed patient-weighted RidgeClassifier define each CAV. Validation/test only evaluate. Reported AUC describes high/low separability; TCAV describes directional true-class-logit sensitivity. Reference p-values and their Holm adjustment remain exploratory, conditional on patient-block sign-exchangeability assumptions. The method does not guarantee biological specificity, and its patient intervals do not include CAV-fit or model-training variability. The PDF currently reports the prior completed audit; CAV/TCAV should only be added after this full stage has completed.

## Current recommended experiment: restored source partitions

The downloaded `data/bracs/BRACS.xlsx` has now been inspected. Its `WSI_Information` sheet maps all 4,493 available graphs to 150 patients and source assignments (`WSI Filename`, `Patient Id`, `Set`). Restored partitions contain 3,630 / 312 / 551 graphs and 105 / 15 / 30 patients for training/validation/test, with no shared slides or patients. All graph files passed validation again. Provenance is summarized in `BRACS_METADATA_VERIFICATION.json`.

Use **`results/bracs_source_patient`** for new experiments; it supersedes the random internal partition proposals below. These are the source BRACS assignments on available graphs, not a newly sampled partition. Prior exploratory exposure to these data still prevents treating this as an untouched external test.

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage scaler --work-dir results/bracs_source_patient
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage train --work-dir results/bracs_source_patient --seed 42
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage audit --work-dir results/bracs_source_patient --seed 42
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage collect --work-dir results/bracs_source_patient
```

Run one stage at a time and stop on any error. The scaler stage verifies and reuses an existing matching scaler. Training may take a long time on CPU. The remaining sections document the earlier internal-split workflow and metadata-import procedure; substitute the recommended work directory for subsequent training/audit commands. In the notebook, select `WORK = ROOT / 'results/bracs_source_patient'`.

## Already executed locally

- All 4,493 graph files passed structural, finite-value, index and metadata checks.
- A fixed internal slide-grouped split was prepared in `results/reproducible_run`: 2,698 training graphs / 896 validation / 899 test; 226 / 79 / 81 slides. All three slide intersections are zero and all seven classes occur in each partition.
- The notebook was expanded to 32 experimental cells and executed with long-running flags disabled. Data checks and inventory plots have real saved outputs; new-model sections explicitly show pending results.
- Six software/oracle regression tests pass. A small integration fixture exercises scaler fitting, new split sizes, training and audit, including ordered output IDs.
- The trained synthetic stress test has not passed every assumption. The 32D variant failed untrained area decoding. Matching the real 128D architecture passes that threshold (R² 0.646) but fails the assumption that a trained model ignores texture noise. These failures are retained. The exact edge-only oracle passes and demonstrates zero response to unused node features by construction.

No full real-data retraining or new-model real-data audit has been completed yet.

## 1. Obtain source patient metadata before final long runs

The [official BRACS download page](https://www.bracs.icar.cnr.it/download/) requires an account/login. Obtain the small **BRACS.xlsx summary file**, described in the [official dataset details](https://www.bracs.icar.cnr.it/details/), and place it at `data/bracs/BRACS.xlsx`. There is no need to download the full slide/image collection for this mapping step.

The official documentation describes the summary as containing slide IDs, patient IDs and original partitions. The Graph-BRACS release contains a different supplied split; do not label a newly generated split as the official BRACS test.

```powershell
.venv/Scripts/python.exe scripts/map_bracs_patients.py --input data/bracs/BRACS.xlsx --inspect
```

This prints sheet names and columns without guessing identifiers. Once those names are known, run the mapping command using the actual values (replace the three placeholders):

```powershell
.venv/Scripts/python.exe scripts/map_bracs_patients.py --input data/bracs/BRACS.xlsx --sheet "ACTUAL_SHEET" --key-column "ACTUAL_SLIDE_COLUMN" --patient-column "ACTUAL_PATIENT_COLUMN"
```

The script requires a complete match for all graphs and rejects ambiguous assignments. If it fails, investigate the identifiers before training. If the source table maps individual ROI names instead of slides, use `--key-level sample` and the appropriate sample-name column. Do not substitute invented patient IDs or silently drop unmatched graphs.

## 2. Freeze patient-grouped partitions in a NEW directory

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage prepare --work-dir results/reproducible_patient --patient-mapping processed/graph-bracs/patient_mapping.csv
```

This scans the graph files again, records hashes, and creates an internal patient-grouped study. It does not change the old slide-grouped or dataset-supplied partitions. The split uses fixed folds and is not selected using model performance. If a class cannot be represented in every partition under the grouping, the generator fails for methodological review.

If the source mapping is genuinely unavailable, the already-prepared slide-grouped study can be used for a clearly limited exploratory run: replace `results/reproducible_patient` with `results/reproducible_run` in every command below. Patient independence must then remain unverified. Prefer resolving the source mapping before spending time on multiple seeds.

## 3. Fit normalization on TRAIN only

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage scaler --work-dir results/reproducible_patient
```

Wait for completion. The scaler stores its training IDs, and training rejects a mismatch. Do not reuse the historical scaler after changing the split.

## 4. Train ONE model first

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage train --work-dir results/reproducible_patient --seed 42
```

This machine currently has CPU-only PyTorch. Training uses two threads, up to 80 epochs and early stopping after 12 epochs without validation improvement. No fixed runtime is promised. Keep the terminal open; progress is flushed each epoch and saved to `seed42/train.log`. Successful completion produces `model.pt`, `model.history.csv` and `model.metrics.json`. A short `--epochs 2` run is only a runtime/installation smoke test and must use a separate work directory, not the final experiment.

If a stage fails, stop and inspect its log. The workflow refuses to overwrite failed attempts as well as completed outputs. Do not delete partial evidence blindly; select a new work directory when restarting a corrected protocol.

## 5. Audit that model

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage audit --work-dir results/reproducible_patient --seed 42
```

This is also computationally expensive: five untrained probe baselines, 12 random-feature controls, descriptor interventions and 2,000 slide bootstrap replicates. Joint interventions are omitted in this focused run. The holdout is explicitly named `test`, not `official_test`.

The runner records the checkpoint hash, training/evaluation IDs, status and per-graph arrays. Realized displacement and clipping are computed per graph. Adjusted extreme bootstrap intervals and random-feature contrasts remain exploratory; they do not prove concept specificity or biological causality. This limitation is a property of the controls, not something additional execution alone can remove.

## 6. Generate results from completed runs

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage collect --work-dir results/reproducible_patient
```

Read `results/reproducible_patient/summary/executed_results.md`, its CSV tables and plots. No result is synthesized for an absent or incomplete audit. The collector rejects overlapping slides, repeated checkpoint hashes/seeds, incomplete audits and changed split IDs.

In the notebook, set `WORK = ROOT / 'results/reproducible_patient'`, keep the launch flags false and rerun the cells to inspect the saved learning curves, scores, decodability, interventions and diagnostics. For terminal training, notebook launch flags need never be enabled.

## 7. Only then repeat independent seeds

After seed 42 completes, use the same frozen experiment directory for the remaining seeds:

```powershell
foreach ($trainingSeed in 41, 43, 44, 45) {
    .venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage train --work-dir results/reproducible_patient --seed $trainingSeed
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $trainingSeed" }
    .venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage audit --work-dir results/reproducible_patient --seed $trainingSeed
    if ($LASTEXITCODE -ne 0) { throw "Audit failed for seed $trainingSeed" }
}
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage collect --work-dir results/reproducible_patient
```

One trained seed is enough for an explicitly preliminary observation; stability claims require independent replications. On CPU, inspect seed 42's runtime before launching four additional runs.

## Recheck software or rerun synthetic stress tests

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
```

This runs the six software/oracle tests, including the small integration fixture. It does not rerun the expensive trained stress benchmark. To investigate that benchmark in a new output folder:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
.venv/Scripts/python.exe tests/test_v3_synthetic.py --hidden-dim 128 --out-dir results/synthetic_stress_new
```

A failed learned-model stress test remains a failed check; do not lower its threshold simply to produce a green result. Its label-level and classifier-level hypotheses must be separated when interpreting the result.

## Updating the professor-facing PDF

The English PDF now uses the completed source-restored seed-42 training and audit, including verified per-graph arrays and patient-bootstrap intervals. The five-random-GINE classification comparison and paired patient-bootstrap contrasts have now completed and are included in the classification-results PDF. `summary/executed_results.md` and aggregate CSVs describe all completed collected seeds. The reviewed PDF narrative is tied to the selected seed-42 snapshot; review it before changing checkpoints or claiming multi-seed stability. Do not present the source-restored holdout as a new external cohort.


## Direct classification performance of random GINE controls

The concept audit already compares trained and random GINE representations for descriptor decoding. It had not measured the random GINEs' direct seven-class predictions. Notebook section 12 now addresses that question on the frozen patient-disjoint test set. Set `RUN_UNTRAINED_CLASSIFICATION = True` and run that cell, then the result cell. This launches `scripts/evaluate_untrained_classification.py`; it makes no model-fitting updates.

It evaluates the five same initialization seeds used by the decodability control, with the checkpoint's TRAIN-only feature scaler and the same TRAIN-only BatchNorm calibration (first 30 ordered training batches). It reports each random network's own random output-head argmax, the training-majority constant and the trained checkpoint, with accuracy, macro-F1 and 2,000 patient-bootstrap intervals. It validates ordered IDs, split hashes and group separation. It refuses to overwrite a nonempty output directory. Results are saved under `results/bracs_source_patient/seed42/untrained_classification/`.

Interpret carefully: the five random networks measure initialization variation of the fixed random classifier; bootstrap intervals condition on each classifier. This does not test a learned classifier on frozen random embeddings, does not measure training-seed variability for the trained model, and cannot alone establish that a low-performing trained model learned no useful representation. Keep the paper comparison pending until these outputs have been inspected.


## Run the random-network classification comparison

Open `notebooks/graph_bracs_experiments_and_results.ipynb` with the project `.venv` kernel. Run notebook section 12's launch cell after setting `RUN_UNTRAINED_CLASSIFICATION = True`, then run the result cell immediately below it. This section does not require rerunning training, audit, CAV/TCAV, or earlier notebook cells if the kernel still contains the setup variables; after restarting the kernel, run the initial setup and partition/checkpoint cells first. Expect a few minutes on CPU. The script prints progress and writes the held-out predictions and aggregate table once. Do not rerun the launch cell after completion unless you preserve/remove its nonempty output directory first.

The resulting `classification_metrics.csv` and `manifest.json` are the input for the next report revision. Until these exist and agree with the saved checkpoint and partition provenance, the manuscript does not claim how well the random networks classify.

The direct random-classifier results were executed locally on CPU. Five random GINE Macro-F1 scores: 0.0389, 0.0864, 0.0846, 0.0616, 0.0362 (mean 0.0615, sample SD 0.0240); test accuracy range 0.1397-0.1652. The trained GINE is 0.4657 Macro-F1 / 0.4719 accuracy; the train-majority constant is 0.0354 / 0.1416. Paired patient bootstrap: trained minus mean of five random classifiers +0.4042 (95% interval 0.3220-0.4532), trained minus majority +0.4303 (0.3533-0.4734). These comparisons condition on fixed models and 30 test patients; the five random models do not replace independent trained-seed replication. The output and paired-comparison scripts are `scripts/evaluate_untrained_classification.py` and `scripts/summarize_untrained_classification.py`.
