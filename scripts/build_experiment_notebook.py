"""Create an English experimental notebook with safe, explicitly selected stages."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "id": f"cell-{len(cells):03}",
                "source": textwrap.dedent(source).strip().splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    md = lambda text: add("markdown", text)
    code = lambda text: add("code", text)
    md("""
    # Graph-BRACS: Reproducible Concept Audit

    **Purpose:** inspect data, verify group separation, run a train-only scaler, train independent GINEs,
    and examine descriptor sensitivity alongside decodability baselines.

    This notebook is the experimental companion to the early research manuscript.
    Historical results, executed checks, and pending new training runs are kept distinct.
    Figures and tables below come from files or current calculations, never invented results.

    **Research question:** for input-derived cell-graph concepts, what evidence of task-specific reliance
    survives untrained-network, input-redundancy and realized-dose controls?
    The companion manuscript and bibliography are generated locally under `Latex/`,
    which is excluded from the public repository at the author's request. This notebook
    contains the executable experimental companion and selected references.
    The general distinction between decodability and reliance is established prior work, not a novelty claim.
    """)
    md("""
    ## 1. Environment and experiment selection

    Select this project's `.venv` Python kernel. Long training and audit stages are opt-in below.
    Start with one seed, inspect its execution, then repeat the same frozen protocol across seeds.
    Source-restored partitions and exploratory internal partitions are identified by their manifests.
    Prior exploratory exposure means the restored test is not an untouched external evaluation.
    """)
    code("""
    from pathlib import Path
    import json, subprocess, sys, os, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from IPython.display import display, Markdown, Image

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/run_audit_v3.py').exists())
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'scripts'))
    WORK = ROOT / 'results/bracs_source_patient'
    SEED = 42
    RUN_TRAINING = False
    RUN_AUDIT = False
    RUN_COLLECTION = False
    print('Python executable:', Path(sys.executable).name)
    print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
    print('Experiment:', WORK.relative_to(ROOT), '| training seed:', SEED)

    def display_report_figure(name):
        verification = json.loads((ROOT / 'Latex/verification.json').read_text())
        expected_run = (WORK / f'seed{SEED}/audit').relative_to(ROOT).as_posix()
        actual_run = verification['source_run'].replace(chr(92), '/')
        if actual_run != expected_run or verification['seed'] != SEED:
            raise ValueError('Report figures belong to a different run. Rebuild the report for the selected experiment.')
        path = ROOT / 'Latex/figures' / f'{name}.png'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != verification['figures'][name]['png_sha256']:
            raise ValueError(f'Figure changed since report verification: {name}')
        display(Image(data=path.read_bytes()))
    """)
    md("""
    ## 2. Original metadata and slide identifiers

    The Graph-BRACS card describes 4,493 graphs with a supplied 3,594/899 split.
    This is not automatically the original BRACS patient-disjoint partition.
    BRACS documentation describes the filename as encoding the parent slide.
    Sources: [Graph-BRACS](https://huggingface.co/datasets/ogutsevda/graph-bracs),
    [BRACS data organization](https://www.bracs.icar.cnr.it/details/).
    """)
    code("""
    from concept_audit_v3.stats import wsi_ids
    metadata = pd.read_csv('data/graph-bracs/metadata.csv')
    metadata['slide_id'] = wsi_ids(metadata.sample_id)
    display(pd.crosstab(metadata.label, metadata.split))
    print('Graphs:', len(metadata), '| slide identifiers:', metadata.slide_id.nunique())
    assert not metadata.sample_id.duplicated().any()
    """)
    code("""
    source_slides = {s: set(d.slide_id) for s, d in metadata.groupby('split')}
    shared = source_slides['train'] & source_slides['test']
    print('Shared slide identifiers in supplied train/test:', len(shared))
    print('Test ROIs on a supplied training slide:', metadata.query("split == 'test'").slide_id.isin(shared).sum())
    """)
    md("""
    ## 3. Full graph validation and frozen evaluation partitions

    `prepare_reproducible_run.py` checks graph dimensions, finite descriptors, edge endpoints,
    raw nonnegative distances, and graph/metadata IDs and labels. It can restore source BRACS assignments
    or choose fixed internal group-stratified folds, without using classifier performance to select a split.
    Original source split labels are retained in `source_split`; no data files are overwritten.
    """)
    code("""
    manifest_file = WORK / 'split_manifest.json'
    split_manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else None
    if split_manifest:
        display(pd.DataFrame(split_manifest['counts']).T)
        print('Graph files checked:', split_manifest['graph_files_checked'])
        print('Errors:', split_manifest['graph_errors'], '| Ready:', split_manifest['ready_for_training'])
        print('Grouping:', split_manifest['grouping'])
    else:
        print('Pending: python scripts/run_reproducible_experiment.py --stage prepare')
    """)
    code("""
    from concept_audit_v3.integrity import validate_splits
    if split_manifest:
        partitions = {s: pd.read_csv(WORK / f'{s}_ids.csv') for s in ['train', 'validation', 'test']}
        display(pd.Series(validate_splits(partitions), name='Shared slide IDs'))
        display(pd.read_csv(WORK / 'class_counts.csv').set_index('label'))
        assert split_manifest['ready_for_training']
    """)
    code("""
    inventory_file = WORK / 'graph_inventory.csv'
    if inventory_file.exists():
        inventory = pd.read_csv(inventory_file)
        display(inventory[['nodes','edges','distance_min','distance_max']].describe().round(3))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout='constrained')
        for ax, col in zip(axes, ['nodes', 'edges']):
            ax.hist(inventory[col], bins=40)
            ax.set(xlabel=f'{col.capitalize()} per graph', ylabel='Graph count')
        plt.show()
    """)
    md("""
    ## 4. Patient mapping and restored BRACS assignments

    The downloaded `BRACS.xlsx` has now matched every available graph to its patient. The recommended
    `results/bracs_source_patient` experiment restores source assignments: 3,630 training, 312 validation,
    and 551 test graphs, from 105, 15 and 30 patients respectively, without shared patients or slides.
    The source sheet is `WSI_Information`; columns are `WSI Filename`, `Patient Id` and `Set`.
    The following documents how to inspect the source again if needed.
    Obtain the small `BRACS.xlsx` summary through the authenticated [BRACS download page](https://www.bracs.icar.cnr.it/download/).
    You do not need to download all images for this step. Inspect its columns first:

    ```powershell
    .venv/Scripts/python.exe scripts/map_bracs_patients.py --input data/bracs/BRACS.xlsx --inspect
    ```

    Then select the actual sheet, slide-ID column and patient-ID column explicitly; see `docs/EXECUTION_GUIDE.md`.
    Both earlier internal proposals remain preserved; use the restored-source experiment for new training.
    """)
    code("""
    mapping_path = ROOT / 'processed/graph-bracs/patient_mapping.csv'
    print('Patient mapping available:', mapping_path.exists())
    if split_manifest:
        print('Patient independence verified:', split_manifest['patient_independence_verified'])
    print('Source mapping provenance: docs/BRACS_METADATA_VERIFICATION.json')
    """)
    md("""
    ## 5. Executed controls: software correctness versus learned-model behavior

    The exact oracle classifier depends only on edge distances, so unused node descriptors must have
    exactly zero effect. Regression tests also check group separation, training/scaler provenance,
    dose-zero identity, clipping, and diagnostics under graphs of different sizes.

    Separately, the trained synthetic benchmark is a stress test. Independence from the **data labels**
    does not guarantee that a finite trained model ignores a feature. Its failures must remain visible.
    """)
    code("""
    for relative in ['results/synthetic_validation/status.json', 'results/synthetic_validation_v2/status.json']:
        path = ROOT / relative
        if path.exists():
            display(Markdown(f'**{relative}**'))
            display(pd.Series(json.loads(path.read_text())))
    """)
    md("""
    ## 6. Train-only normalization and staged execution

    The scaler stores its training IDs. Training refuses a scaler fitted on different IDs.
    The architecture is the same GINE as before; checkpoint parameter names remain compatible.
    Model selection uses validation Macro-F1, and the chosen checkpoint is evaluated on the explicit holdout.
    The default CPU configuration uses two threads. Terminal commands are listed in the execution guide.
    """)
    code("""
    def launch_stage(stage, execute=False):
        command = [sys.executable, 'scripts/run_reproducible_experiment.py', '--stage', stage,
                   '--work-dir', str(WORK.relative_to(ROOT)), '--seed', str(SEED)]
        print(subprocess.list2cmdline(command))
        if execute:
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print('Not launched. Enable the corresponding flag or run this command in a terminal.')

    scaler_path = WORK / 'scaler.npz'
    if scaler_path.exists():
        with np.load(scaler_path) as scaler:
            print('Training IDs stored:', len(scaler['train_ids']))
            assert set(scaler['train_ids']) == set(partitions['train'].sample_id)
    else:
        launch_stage('scaler')
    """)
    code("""
    checkpoint_path = WORK / f'seed{SEED}/model.pt'
    if checkpoint_path.exists():
        print('Existing checkpoint:', checkpoint_path.relative_to(ROOT))
    else:
        launch_stage('train', execute=RUN_TRAINING)
    """)
    md("""
    ### Learning curves and classifier scores

    These cells display outputs only after training completes. An absent result stays pending.
    Training and validation curves diagnose optimization; they are not independent test estimates.
    Report plots below display the PNG exports of the exact figures embedded in the PDF.
    Their source run and image hashes are checked before display; no separate notebook plotting style is used.
    """)
    code("""
    history_path = checkpoint_path.with_suffix('.history.csv')
    if history_path.exists():
        history = pd.read_csv(history_path)
        display_report_figure('learning_curve')
    else:
        print('Pending: no completed training history for the new protocol.')
    """)
    code("""
    metrics_path = checkpoint_path.with_suffix('.metrics.json')
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text())
        display(pd.DataFrame({'validation': model_metrics['validation'], 'internal_holdout': model_metrics['test']}))
    else:
        print('Pending: no classifier metrics for this new training seed.')
    """)
    md("""
    ## 7. Probing baselines and controlled descriptor interventions

    The runner fits Ridge probes on TRAIN only and compares trained representations with five untrained
    initializations and other raw features. Node targets and the graph distance target have different units.
    Interventions save ordered sample IDs, per-graph realized doses, clipping and predictions.

    The first execution skips joint interventions to keep the study focused. Random-feature contrasts
    remain descriptive: controls for propagated edits and edge edits are not fully matched.
    Bonferroni tail intervals with few bootstrap draws are not treated as validated clinical evidence.
    """)
    code("""
    audit_dir = WORK / f'seed{SEED}/audit'
    audit_manifest_path = audit_dir / 'manifest.json'
    if audit_manifest_path.exists():
        audit_manifest = json.loads(audit_manifest_path.read_text())
        print('Audit status:', audit_manifest.get('status', 'historical unknown'))
        print('Training provenance checked:', audit_manifest.get('training_ids_verified'))
        display(pd.Series(audit_manifest['leakage']))
    else:
        launch_stage('audit', execute=RUN_AUDIT)
    """)
    code("""
    probe_file = audit_dir / 'decodability_baselines.csv'
    if probe_file.exists():
        probes = pd.read_csv(probe_file)
        display(probes.round(4))
        display_report_figure('decodability')
    else:
        print('Pending: new-model decodability baselines.')
    """)
    code("""
    intervention_file = audit_dir / 'interventions_test.csv'
    if intervention_file.exists():
        interventions = pd.read_csv(intervention_file)
        display(interventions.loc[interventions.alpha.abs()==1,
            ['concept','alpha','isolated_dlogit','isolated_dF1','isolated_actual_change','isolated_clip']].round(4))
        display_report_figure('interventions')
        print('Left: isolated edits. Right: correlation-propagated edits.')
        print('Both panels: Macro-F1 changes with paired nominal 95% patient-bootstrap intervals.')
    else:
        print('Pending: new-model descriptor intervention responses.')
    """)
    md("""
    ### Verify actual doses and ordered evaluation IDs

    A nonzero requested dose can be clipped. Interpret its actual change and clipping alongside
    prediction changes. The odd/even response decomposition alone does not identify distribution shift.
    """)
    code("""
    arrays_path = audit_dir / 'interventions_test.npz'
    if arrays_path.exists():
        with np.load(arrays_path) as arrays:
            assert np.array_equal(arrays['sample_ids'],partitions['test'].sample_id.to_numpy(dtype=str))
            zero = int(np.flatnonzero(arrays['alphas']==0)[0])
            for key in arrays.files:
                if key.endswith('__dlogit'):
                    assert np.allclose(arrays[key][zero],0)
            print('Ordered IDs and dose-zero logits verified.')
    else:
        print('Pending: new per-graph output arrays.')
    """)
    md("""
    ## 8. Independent seeds and result collection

    One full seed-42 training and audit has now completed. For stability claims, repeat with seeds 41, 43, 44 and 45 using
    the **same** frozen partitions and hyperparameters. Rerunning seed 42 is not a new training seed.
    Keep slide bootstrap uncertainty conditional on a model separate from variation across trained models.
    On this CPU, start with one seed and inspect its runtime before scheduling four more.
    """)
    code("""
    completed = []
    for path in sorted(WORK.glob('seed*/audit/manifest.json')):
        item = json.loads(path.read_text())
        completed.append({'seed':item.get('seed'),'status':item.get('status'),
                          'checkpoint_hash':item['sha256'][:12]})
    display(pd.DataFrame(completed,columns=['seed','status','checkpoint_hash']))
    if completed:
        launch_stage('collect',execute=RUN_COLLECTION)
    else:
        print('Pending: complete at least one new-model audit before collecting results.')
    """)
    code("""
    executed_report = WORK / 'summary/executed_results.md'
    if executed_report.exists():
        display(Markdown(executed_report.read_text()))
        display(pd.read_csv(WORK / 'summary/classifier_scores.csv'))
    else:
        print('No final result summary generated for the new protocol yet.')
    """)
    md("""
    ## 9. Historical reference, kept separate

    The previous grouped-protocol model is retained for provenance. Its split had slide overlap,
    and its realized-dose logs contain a corrected-since tensor-view bug. These results are not
    silently merged with the newly partitioned experiment.
    """)
    code("""
    historical = ROOT / 'results/audit_v3_grouped_seed42_complete'
    if historical.exists():
        record = json.loads((historical / 'manifest.json').read_text())
        display(pd.Series(record['leakage'],name='Historical slide overlap'))
        display(pd.read_csv(historical / 'decodability_baselines.csv').round(4))
    """)
    md("""
    ## 10. Verified patient-cluster uncertainty and report

    The English PDF now uses the completed source-restored seed-42 experiment. The report builder
    checks ordered labels/IDs, split and checkpoint hashes, all 60 variant/dose cells and dose-zero identity.
    Intervals are recomputed from saved predictions by resampling whole patients (2,000 draws).
    This requires no GNN rerun. They condition on one model and are not inter-training uncertainty.

    Descriptor sensitivity remains distinct from concept specificity or biological causality.
    Historical runs and failed stress tests remain explicit, without replacing the completed new results.
    """)
    code("""
    verification = json.loads((ROOT / 'Latex/verification.json').read_text())
    if verification.get('source_run', '').replace('\\\\', '/').endswith('bracs_source_patient/seed42/audit'):
        print('Recomputed test Macro-F1:', verification['test_macro_f1_recomputed'])
        print('Patient-cluster 95% interval:', verification['patient_bootstrap_ci95'])
        print('Test patients:', verification['n_test_patients'])
        print(verification['checks'])
        verified_edits = pd.read_csv(ROOT / 'Latex/tables/verified_interventions.csv')
        display(verified_edits.loc[verified_edits.alpha.abs()==1].round(4))
    """)
    md("""
    ## 11. CAV / TCAV on the completed checkpoint (run explicitly)

    This stage freezes the existing GINE; it does not retrain it. **All five concepts are graph means**
    and CAVs live in the pooled graph embedding. This gives a common unit, but is a different
    protocol from the historical node-level cellular CAVs.

    TRAIN q20/q80 define low/high examples. A fixed patient-weighted RidgeClassifier fits the CAV,
    oriented toward high concept values. Validation and test only evaluate it. AUC measures
    high/low separation, not diagnostic usefulness. TCAV is the fraction of true-class graphs with
    a positive directional derivative of that class logit at the graph embedding.

    The runner fits 999 randomized CAV controls by flipping low/high labels together within each
    TRAIN patient. It computes patient-bootstrap intervals and exploratory two-sided control-reference
    p-values with Holm correction over the 35 concept/class cells, separately for validation and test.
    These reference tests assume patient-block label-sign exchangeability; this assumption is not
    guaranteed by the data. Do not equate small reference p-values with concept specificity or causality.
    These are adapted controls, not a claim to reproduce the original TCAV significance procedure.
    [TCAV method: Kim et al. (2018)](https://proceedings.mlr.press/v80/kim18d.html).

    Set `RUN_CAV_TCAV = True` in the next cell and execute it. Representations/gradients are cached,
    so control fits do not rerun the GNN hundreds of times. A matching completed run is reused.
    """)
    code("""
    RUN_CAV_TCAV = False  # Change to True to launch this new stage.
    CAV_CONTROLS = 999
    CAV_BOOTSTRAPS = 2000
    CAV_TCAV_DIR = WORK / f'seed{SEED}/cav_tcav'

    if RUN_CAV_TCAV:
        command = [sys.executable, 'scripts/run_cav_tcav.py', '--work-dir', str(WORK.relative_to(ROOT)),
                   '--seed', str(SEED), '--n-controls', str(CAV_CONTROLS), '--n-boot', str(CAV_BOOTSTRAPS)]
        print(subprocess.list2cmdline(command), flush=True)
        environment = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'PYTHONUNBUFFERED':'1'}
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True)
        if process.wait():
            raise RuntimeError('CAV/TCAV failed. Keep its outputs and inspect the message before continuing.')
    else:
        print('New CAV/TCAV stage not launched. Set RUN_CAV_TCAV = True above to execute it.')
    """)
    code("""
    cav_manifest_path = CAV_TCAV_DIR / 'manifest.json'
    if cav_manifest_path.exists():
        cav_manifest = json.loads(cav_manifest_path.read_text())
        print('CAV/TCAV status:',cav_manifest['status'])
        if cav_manifest['status']=='completed':
            if cav_manifest['smoke_test']:
                raise ValueError('A smoke test is not a final CAV/TCAV result.')
            display(pd.read_csv(CAV_TCAV_DIR / 'cav_scores.csv').style.format(
                {'auc':'{:.3f}','q20':'{:.3e}','q80':'{:.3e}'},na_rep='not estimable'))
            display(Image(filename=str(CAV_TCAV_DIR / 'cav_auc.png')))
        else:
            print('Incomplete run; do not treat partial outputs as results.')
    else:
        print('Pending: no new-checkpoint CAV results yet.')
    """)
    code("""
    if cav_manifest_path.exists() and json.loads(cav_manifest_path.read_text())['status']=='completed':
        tcav = pd.read_csv(CAV_TCAV_DIR / 'tcav_scores.csv')
        columns = ['concept','class','n_graphs','n_patients','tcav','tcav_ci95_low','tcav_ci95_high',
                   'control_tcav_mean','tcav_minus_controls','control_reference_p','control_reference_p_holm','status']
        display(tcav.loc[tcav.split=='test',columns].round(4))
        display(Image(filename=str(CAV_TCAV_DIR / 'tcav.png')))
        print('Intervals condition on this trained model and fitted CAV; they do not measure CAV-fit or training variability.')
        print('Reference p-values are exploratory; no automatic biological/significance conclusion is generated.')
        print('Saved outputs:', CAV_TCAV_DIR.relative_to(ROOT))
    else:
        print('Pending: execute the CAV/TCAV stage before reporting these results.')
    """)
    md("""
    ## 12. Do untrained GINEs classify the histology labels? (run explicitly)

    The decodability audit asks whether input concepts can be recovered from random representations.
    This additional analysis directly evaluates the random-initialized GINEs' seven-class predictions
    on the same patient-disjoint test set. It uses the same five initialization seeds as the audit,
    the checkpoint's TRAIN-only scaler, and TRAIN-only BatchNorm calibration; weights stay random and frozen.
    It also compares a TRAIN-majority constant and the existing trained model. Patient-bootstrap intervals
    are conditional on each fixed predictor and do not measure variation across training runs.

    This evaluates each random network's own randomly initialized output head. It is distinct from
    training a new classifier on frozen random embeddings, which would test representation quality under
    a learned readout. No test labels are used for fitting or selection.
    """)
    code("""
    RUN_UNTRAINED_CLASSIFICATION = False  # Set True, then run this cell and the result cell below.
    UNTRAINED_CLASSIFICATION_DIR = WORK / f'seed{SEED}/untrained_classification'
    if RUN_UNTRAINED_CLASSIFICATION:
        command = [sys.executable, 'scripts/evaluate_untrained_classification.py',
                   '--work-dir', str(WORK.relative_to(ROOT)), '--seed', str(SEED),
                   '--n-models', '5', '--n-boot', '2000']
        print(subprocess.list2cmdline(command), flush=True)
        environment = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'PYTHONUNBUFFERED':'1'}
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True)
        if process.wait():
            raise RuntimeError('Untrained-classification evaluation failed. Inspect the error before continuing.')
    else:
        if (UNTRAINED_CLASSIFICATION_DIR / 'manifest.json').exists():
            print('Completed classification results already exist; they are displayed below. No rerun was launched.')
        else:
            print('Not launched. Set RUN_UNTRAINED_CLASSIFICATION = True to evaluate the five random GINEs.')
    """)
    code("""
    classification_result = UNTRAINED_CLASSIFICATION_DIR / 'classification_metrics.csv'
    if classification_result.exists():
        untrained_classification = pd.read_csv(classification_result)
        display(untrained_classification.round(4))
        random_rows = untrained_classification.query("model.str.startswith('untrained_')",engine='python')
        paired_file = UNTRAINED_CLASSIFICATION_DIR / 'paired_comparisons.csv'
        if paired_file.exists():
            display(pd.read_csv(paired_file).round(4))
        else:
            print('Paired comparison not summarized yet; run scripts/summarize_untrained_classification.py.')
        print('Random-model macro-F1 mean:', round(random_rows.macro_f1.mean(),4))
        print('Random-model macro-F1 range:', [round(random_rows.macro_f1.min(),4), round(random_rows.macro_f1.max(),4)])
        print('Bootstrap intervals condition on each fixed model; the five-model range describes initialization variation.')
        print('This does not evaluate a trained classifier on frozen random embeddings.')
    else:
        print('Pending: run the preceding cell to compare random-network classification with the trained model.')
    """)
    md("""
    ## 13. Research positioning and references

    The proposed contribution is an empirical audit of input-derived concepts in histopathology cell graphs,
    combining untrained and redundancy baselines with edit-definition and realized-dose diagnostics under
    patient-disjoint evaluation. It is not a claim of a new causal explanation method, a completed
    concept-backpropagation implementation, or superiority over methods not rerun here.

    Closely related work:

    - Magister et al., **GCExplainer** (2021): global concept discovery in GNNs. [Paper](https://arxiv.org/abs/2107.11889).
    - Farace di Villaforesta et al., **Digital Histopathology with Graph Neural Networks: Concepts and Explanations for Clinicians** (2023): graph concepts and logic-based histology explanations. [Paper](https://arxiv.org/abs/2312.02225).
    - Hammersborg and Strumke, **Concept Backpropagation** (2023): probe-guided input perturbations. The current fixed-descriptor edits are a different procedure. [Paper](https://arxiv.org/abs/2307.12601).
    - Kim et al., **TCAV** (ICML 2018): directional class-score sensitivity along CAVs. [Paper](https://proceedings.mlr.press/v80/kim18d.html).
    - Nicolson et al., **Explaining Explainability** (TMLR 2025): CAV entanglement, inconsistency and spatial dependency, including medical images. This is prior evidence of pitfalls, not a gap we claim to discover. [Paper](https://arxiv.org/abs/2404.03713).
    - Alain and Bengio, **Linear Classifier Probes** (2016): accessible information in intermediate representations. [Paper](https://arxiv.org/abs/1610.01644).
    - Adebayo et al., **Sanity Checks for Saliency Maps** (NeurIPS 2018): parameter/data dependence checks motivate our random-network comparison. [Paper](https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).
    - Brancati et al., **BRACS** (Database 2022), and Ogut et al., **GrapHist** (2026 preprint): image/source metadata and released graph provenance. [BRACS](https://doi.org/10.1093/database/baac093), [GrapHist](https://arxiv.org/abs/2603.00143).

    See `docs/ABSTRACT_REVISION_NOTES.md` for the audit of the original abstract.
    The generated manuscript bibliography is kept locally under the Git-ignored `Latex/` folder.
    Full CAV/TCAV results are included when section 11 has completed.
    """)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/graph_bracs_experiments_and_results.ipynb"
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"Wrote {len(cells)} experimental notebook cells: {path}")


if __name__ == "__main__":
    main()
