# Graph-BRACS concept audit

Research code for separating concept decodability from prediction sensitivity in a GINE classifier of histology graphs.

## Current executed result

The original BRACS workbook now maps all 4,493 available graphs to 150 patients. Source assignments restore 3,630 training, 312 validation and 551 test graphs, with zero shared patients, slides or sample IDs.

One new training run (seed 42) and its selected audit are complete. The selected model has validation Macro-F1 **0.5599** (epoch 35), test accuracy **0.4719**, and test Macro-F1 **0.4657**. A 2,000-draw patient-cluster bootstrap gives a 95% interval of **0.388-0.515**, conditional on this trained model. Independent-training variability is not yet estimated.

- The manuscript, abstract, figures and detailed result tables are generated locally under `Latex/` and excluded from version control.
- [Original-abstract review and contribution positioning](docs/ABSTRACT_REVISION_NOTES.md)
- [Experimental notebook: 41 cells, including completed CAV/TCAV outputs and research references](notebooks/graph_bracs_experiments_and_results.ipynb)
- [Execution guide](docs/EXECUTION_GUIDE.md), [scientific audit](docs/SCIENTIFIC_AUDIT.md), and [verification history](docs/VERIFICATION.md)

The direct classification comparison is complete: five random GINEs scored Macro-F1 0.036-0.086 (mean 0.0615), versus 0.4657 for the trained model and 0.0354 for the train-majority baseline. The paired trained-minus-random difference is 0.4042 (95% patient-bootstrap interval 0.3220-0.4532). This confirms that the trained model predicts labels far better than its random-initialized counterparts, despite small trained-versus-untrained gaps in concept decodability. The comparison is conditional on one trained model and five random initializations.

The full CAV/TCAV extension is also complete: graph-level test CAV AUCs are 0.9947-1.0000, but none of the 35 test class/concept scores is distinguished from 999 patient-block randomized controls (minimum unadjusted reference p=0.143; all Holm-adjusted values=1.0). The manuscript includes the AUC table and the same TCAV heatmap displayed in the notebook. These reference comparisons remain exploratory.

Decodability is high, but untrained networks and other raw descriptors already recover much of the information. Descriptor edits affect predictions, with substantial clipping for some negative doses. These results do not establish concept specificity or biological causality. Prior exploratory exposure to the data prevents describing the restored test as an untouched external cohort.

## Environment

Create/activate a Python environment and install `requirements.txt`. The observed local versions are recorded in `docs/environment_observed.txt`; dependency ranges are not a portable lockfile. Select the project's Python kernel for the notebook.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Data and provenance

Datasets, patient mappings, checkpoints and raw experiment arrays are not redistributed. Obtain Graph-BRACS and the BRACS summary workbook under their applicable terms. Local expected inputs are `data/graph-bracs/metadata.csv`, trusted PyG graph files under `data/graph-bracs/data/`, and `data/bracs/BRACS.xlsx`. The workbook sheet and columns were inspected and matched, not guessed. See `docs/BRACS_METADATA_VERIFICATION.json` for aggregate mapping provenance.

`results/bracs_source_patient/` is the current local experiment. Earlier internal split proposals are preserved but superseded. Historical checkpoint scores of 0.6171 and 0.5904 used different, overlapping partitions and are not direct comparators for the new result. Historical realized-dose logs are invalid; the corrected run saves per-graph diagnostics and ordered IDs.

## Rebuild the completed report

```powershell
.venv/Scripts/python.exe scripts/build_completed_report.py
tectonic --outdir Latex Latex/graph_bracs_research_classification_results.tex
tectonic --outdir Latex Latex/graph_bracs_abstract_classification_results.tex
```

The builder reads the **completed source-restored seed-42 run**. It verifies labels, ordered IDs, partition/checkpoint hashes, all 60 variant/dose summary cells and zero-dose identity, then computes patient-bootstrap intervals from saved arrays. It also requires the completed full CAV/TCAV run and independently verifies held-out AUCs, gradient-direction products, all 70 validation/test directional/control/reference-value cells and Holm adjustment. It does not retrain the GNN. It generates the manuscript, tables, figures and verification record locally under the Git-ignored `Latex/` folder. Rebuilding requires local data and completed result files; these are not distributed. The notebook is an execution companion, not the paper. The current contribution is an early empirical audit, not a validated new causal method or a top-venue acceptance claim.

## Software checks and synthetic stress test

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
```

Tests cover split/group separation, scaler provenance, per-graph edit diagnostics, an exact edge-only oracle, a small scaler/training/audit integration fixture and patient-bootstrap equivalence to explicit resampling. The separately executed trained synthetic stress test is not fully validated: the historical 32D model failed area decoding; a 128D repeat passes that threshold but fails the texture-noise expectation. Dataset-label independence does not guarantee zero dependence of a finite trained classifier. Those failures remain recorded.

## Continue experiments

Seed 42 is complete; do not rerun its training command. Use a new independent seed with the same frozen source partitions for stability analyses:

```powershell
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage train --work-dir results/bracs_source_patient --seed 41
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage audit --work-dir results/bracs_source_patient --seed 41
.venv/Scripts/python.exe scripts/run_reproducible_experiment.py --stage collect --work-dir results/bracs_source_patient
```

Run stages individually and stop on errors. Training/audits can be slow on this CPU. Outputs refuse overwriting. The focused audit skips joint interventions; the current report claims no new joint-effect results. The notebook's training/audit/collection flags default to false and can display existing results without launching computations.

## Publication scope

Local environments, raw data, patient mappings, checkpoints, generated reports and experiment outputs are ignored by Git. The repository contains no software license, so code reuse permissions are not specified. Verify third-party and dataset terms before redistributing data or derived artifacts.
