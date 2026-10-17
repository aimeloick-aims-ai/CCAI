"""Verify the completed source-restored experiment and build the English report.

No GNN is rerun. Tables and patient-cluster intervals use saved per-graph arrays.
"""
import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ADH", "DCIS", "FEA", "IC", "N", "PB", "UDH"]
CONCEPTS = ["cell_size", "cell_shape", "cell_intensity", "cell_texture", "spatial_distance"]
NAMES = ["Area", "Circularity", "Intensity", "Texture", "Edge distance"]


def verify_cav_tcav(directory, frames, checkpoint_hash, figures, tables):
    """Recompute separation and all directional/control scores from saved arrays."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest.get("smoke_test") or any(manifest["leakage"].values()):
        raise ValueError("A completed, patient-disjoint full CAV/TCAV run is required")
    if manifest["identity"]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("CAV/TCAV checkpoint differs from the reported classifier")
    if manifest["n_controls"] != 999:
        raise ValueError("Reviewed CAV/TCAV narrative requires 999 controls")
    cav = pd.read_csv(directory / "cav_scores.csv")
    tcav = pd.read_csv(directory / "tcav_scores.csv")
    if len(cav) != 10 or len(tcav) != 70:
        raise ValueError("Incomplete CAV/TCAV tables")
    for split, frame in frames.items():
        if manifest["sample_ids"][split] != frame.sample_id.tolist():
            raise ValueError(f"CAV/TCAV ordered IDs differ: {split}")
        digest = hashlib.sha256((directory.parent.parent / f"{split}_ids.csv").read_bytes()).hexdigest()
        if digest != manifest["identity"]["split_hashes"][split]:
            raise ValueError(f"CAV/TCAV partition hash differs: {split}")
    with np.load(directory / "directions_and_sensitivities.npz") as directions:
        for split in ("validation", "test"):
            frame = frames[split]
            with np.load(directory / f"{split}_representations.npz") as data:
                expected_y = frame.label.map({label:i for i,label in enumerate(LABELS)}).to_numpy()
                np.testing.assert_array_equal(data["y"], expected_y)
                for ci, concept in enumerate(CONCEPTS):
                    vector = directions[concept]
                    np.testing.assert_allclose(np.linalg.norm(vector, axis=0), 1, atol=1e-6)
                    sensitivity = directions[f"{concept}__{split}__sensitivity"]
                    np.testing.assert_allclose(data["gradients"] @ vector, sensitivity, atol=1e-8)
                    row = cav.query("split == @split and concept == @concept").iloc[0]
                    target = data["targets"][:, ci]
                    keep = (target <= row.q20) | (target >= row.q80)
                    auc = roc_auc_score(target[keep] >= row.q80, data["embeddings"][keep] @ vector[:, 0])
                    np.testing.assert_allclose(auc, row.auc, atol=1e-10)
                    positive = sensitivity > 0
                    for k, label in enumerate(LABELS):
                        record = tcav.query("split == @split and concept == @concept and `class` == @label").iloc[0]
                        select = expected_y == k
                        scores = positive[select].mean(axis=0)
                        p = (1 + np.sum(np.abs(scores[1:]-.5) >= abs(scores[0]-.5)-1e-12)) / 1000
                        np.testing.assert_allclose([record.tcav, record.control_tcav_mean, record.tcav_minus_controls, record.control_reference_p],
                                                   [scores[0], scores[1:].mean(), scores[0]-scores[1:].mean(), p], atol=1e-10)
                        assert record.n_graphs == select.sum()
                        assert record.n_patients == frame.loc[select, "patient_id"].nunique()
            subset = tcav.query("split == @split")
            pvalues = subset.control_reference_p.to_numpy()
            order = np.argsort(pvalues)
            adjusted = np.empty(len(order))
            adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
            np.testing.assert_allclose(adjusted, subset.control_reference_p_holm, atol=1e-10)
    for name in ("tcav", "cav_auc"):
        for extension in ("pdf", "png"):
            shutil.copyfile(directory / f"{name}.{extension}", figures / f"{name}.{extension}")
    cav.to_csv(tables / "cav_scores.csv", index=False)
    tcav.to_csv(tables / "tcav_scores.csv", index=False)
    evidence = {"status":"completed", "n_controls":999, "n_verified_directional_cells":70,
                "test_auc_range": [float(cav.query("split == 'test'").auc.min()), float(cav.query("split == 'test'").auc.max())],
                "minimum_reference_p_by_split":tcav.groupby("split").control_reference_p.min().to_dict(),
                "minimum_holm_p_by_split":tcav.groupby("split").control_reference_p_holm.min().to_dict(),
                "checks":"Checkpoint and partition hashes, ordered IDs, labels, direction norms, gradient products, AUCs, all 70 TCAV/control/reference-p cells, counts and Holm adjustment verified against saved arrays.",
                "inference":"Exploratory patient-block sign-flip reference; no concept-specific sensitivity established."}
    return cav, tcav, evidence


def bootstrap_f1(y, pred, group_index, weights, num_classes=7):
    """Macro-F1 with whole-cluster resampling, using fixed class labels."""
    counts = np.zeros((weights.shape[1], num_classes, num_classes), dtype=float)
    np.add.at(counts, (group_index, np.asarray(y, int), np.asarray(pred, int)), 1)
    confusion = np.einsum("bg,gij->bij", weights, counts)
    denominator = confusion.sum(1) + confusion.sum(2)
    numerator = 2 * np.diagonal(confusion, axis1=1, axis2=2)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default="results/bracs_source_patient")
    ap.add_argument("--seed", type=int, default=42, choices=[42], help="Reviewed narrative covers the seed-42 snapshot")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.n_boot < 100:
        ap.error("Use at least 100 bootstrap resamples")
    work = ROOT / args.work_dir
    run = work / f"seed{args.seed}/audit"
    output = ROOT / "Latex"
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(exist_ok=True, parents=True)
    tables.mkdir(exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    split_manifest = json.loads((work / "split_manifest.json").read_text())
    metrics = json.loads((work / f"seed{args.seed}/model.metrics.json").read_text())
    frames = {s: pd.read_csv(work / f"{s}_ids.csv", dtype={"patient_id": str})
              for s in ("train", "validation", "test")}
    if manifest.get("status") != "completed" or not manifest.get("training_ids_verified") or any(manifest["leakage"].values()):
        raise ValueError("Completed audit with verified training IDs and group separation required")
    if manifest.get("smoke_test") or manifest.get("descriptive_only"):
        raise ValueError("A smoke test or unverified historical run cannot supply this report")
    if not split_manifest.get("patient_independence_verified"):
        raise ValueError("Patient mapping must be verified before patient-cluster reporting")
    overlaps = {}
    for a,b in itertools.combinations(frames,2):
        for field in ("sample_id", "slide_id", "patient_id"):
            value = len(set(frames[a][field]) & set(frames[b][field]))
            overlaps[f"{field}_{a}_{b}"] = value
            if value:
                raise ValueError(f"Shared {field}: {a}/{b}")
    for s, frame in frames.items():
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(manifest["sample_ids"][s]):
            raise ValueError(f"Duplicate or mismatched IDs: {s}")
        digest = hashlib.sha256((work / f"{s}_ids.csv").read_bytes()).hexdigest()
        if digest != split_manifest["split_hashes"][s] or digest != metrics["split_hashes"][s]:
            raise ValueError(f"Changed {s} partition")
    checkpoint = work / f"seed{args.seed}/model.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint hash differs from the audited model")
    test = frames["test"]
    expected_y = test.label.map({label:i for i,label in enumerate(LABELS)}).to_numpy()
    cav_dir = work / f"seed{args.seed}/cav_tcav"
    cav, tcav, cav_evidence = verify_cav_tcav(cav_dir, frames, manifest["sha256"], figures, tables)
    classification_dir = work / f"seed{args.seed}/untrained_classification"
    classification_manifest = json.loads((classification_dir / "manifest.json").read_text())
    classification = pd.read_csv(classification_dir / "classification_metrics.csv")
    paired = pd.read_csv(classification_dir / "paired_comparisons.csv")
    if classification_manifest.get("checkpoint_sha256") != manifest["sha256"] or classification_manifest.get("split_hashes") != split_manifest["split_hashes"]:
        raise ValueError("Untrained-classification model or partitions differ from the reported run")
    if classification_manifest.get("n_untrained_models") != 5 or len(classification) != 7 or len(paired) != 7:
        raise ValueError("Expected five random classifiers, trained and majority baselines, and seven paired contrasts")
    with np.load(classification_dir / "test_predictions.npz") as z:
        if not np.array_equal(z["sample_ids"], test.sample_id.to_numpy(dtype=str)):
            raise ValueError("Untrained-classification predictions are not aligned to test IDs")
        np.testing.assert_array_equal(z["y"], expected_y)
        preds = list(z["untrained_predictions"]) + [z["majority_prediction"], z["trained_prediction"]]
        for row, pred in zip(classification.itertuples(), preds):
            np.testing.assert_allclose(row.accuracy, np.mean(expected_y == pred), atol=1e-10)
            np.testing.assert_allclose(row.macro_f1, f1_score(expected_y, pred, labels=range(7), average="macro", zero_division=0), atol=1e-10)
    mean_random = classification.query("model.str.startswith('untrained_')", engine="python")
    if len(mean_random) != 5:
        raise ValueError("Expected five untrained classification rows")
    comparison = paired.set_index("comparison")
    if "trained_minus_mean_of_five_untrained" not in comparison.index or "trained_minus_train_majority_constant" not in comparison.index:
        raise ValueError("Paired classification comparisons are missing")
    random_mean_point = float(np.mean([f1_score(expected_y, pred, labels=range(7), average="macro", zero_division=0) for pred in preds[:5]]))
    trained_point = float(classification.loc[classification.model == "trained_seed42", "macro_f1"].iloc[0])
    np.testing.assert_allclose(comparison.loc["trained_minus_mean_of_five_untrained", "delta_macro_f1"], trained_point-random_mean_point, atol=1e-10)
    classification.to_csv(tables / "untrained_classification.csv", index=False)
    paired.to_csv(tables / "paired_classification_comparisons.csv", index=False)
    dec = pd.read_csv(run / "decodability_baselines.csv").set_index("concept").loc[CONCEPTS].reset_index()
    iv = pd.read_csv(run / "interventions_test.csv")
    if len(iv) != 30 or len(dec) != 5 or not np.allclose(dec.excess_over_untrained, dec.r2_trained-dec.r2_untrained_mean):
        raise ValueError("Unexpected result layout or inconsistent probe excess")
    test = frames["test"]
    _, patient_index = np.unique(test.patient_id, return_inverse=True)
    n_patients = len(np.unique(patient_index))
    rng = np.random.default_rng(42)
    # Multinomial multiplicities equal resampling whole patients with replacement.
    weights = rng.multinomial(n_patients, np.full(n_patients, 1/n_patients), size=args.n_boot)
    results = []
    with np.load(run / "interventions_test.npz") as z:
        if not np.array_equal(z["sample_ids"], test.sample_id.to_numpy(dtype=str)):
            raise ValueError("Saved predictions are not aligned with ordered test IDs")
        y, base = z["y"], z["base_pred"]
        expected_y = test.label.map({label:i for i,label in enumerate(LABELS)}).to_numpy()
        np.testing.assert_array_equal(y, expected_y)
        base_f1 = f1_score(y,base,labels=range(7),average="macro",zero_division=0)
        np.testing.assert_allclose(base_f1, metrics["test"]["macro_f1"], atol=1e-10)
        np.testing.assert_allclose(np.mean(y == base), metrics["test"]["accuracy"], atol=1e-10)
        base_boot = bootstrap_f1(y,base,patient_index,weights)
        ci_low,ci_high = np.quantile(base_boot,[.025,.975])
        zero = int(np.flatnonzero(z["alphas"] == 0)[0])
        for row in iv.itertuples():
            dose = int(np.flatnonzero(z["alphas"] == row.alpha)[0])
            for variant in ("isolated","coherent"):
                prefix = f"{row.concept}__{variant}__"
                np.testing.assert_allclose(z[prefix+"dlogit"][zero],0,atol=1e-7)
                np.testing.assert_array_equal(z[prefix+"pred"][zero],base)
                np.testing.assert_allclose(z[prefix+"actual"][zero],0,atol=1e-7)
                np.testing.assert_allclose(z[prefix+"clip"][zero],0,atol=1e-7)
                pred = z[prefix+"pred"][dose]
                delta = f1_score(y,pred,labels=range(7),average="macro",zero_division=0)-base_f1
                np.testing.assert_allclose(delta,getattr(row,variant+"_dF1"),atol=1e-10)
                for field, col in [("dlogit","dlogit"),("actual","actual_change"),("clip","clip")]:
                    np.testing.assert_allclose(z[prefix+field][dose].mean(),getattr(row,variant+"_"+col),atol=1e-6)
                delta_boot = bootstrap_f1(y,pred,patient_index,weights)-base_boot
                low,high = np.quantile(delta_boot,[.025,.975])
                results.append({"concept":row.concept,"variant":variant,"alpha":row.alpha,
                                "delta_macro_f1":delta,"patient_ci95_low":low,"patient_ci95_high":high,
                                "mean_dlogit":getattr(row,variant+"_dlogit"),
                                "mean_actual_dose":getattr(row,variant+"_actual_change"),
                                "mean_clipped_fraction":getattr(row,variant+"_clip")})
        per_class = f1_score(y,base,labels=range(7),average=None,zero_division=0)
        np.testing.assert_allclose(per_class,metrics["test"]["f1_per_class"],atol=1e-10)
    interventions = pd.DataFrame(results)
    interventions.to_csv(tables / "verified_interventions.csv",index=False)
    parity_rows = []
    for (concept, variant), group in interventions.groupby(["concept", "variant"]):
        for dose in (0.5, 1.0, 2.0):
            negative = group.loc[group.alpha == -dose].iloc[0]
            positive = group.loc[group.alpha == dose].iloc[0]
            even = (positive.delta_macro_f1 + negative.delta_macro_f1) / 2
            odd = (positive.delta_macro_f1 - negative.delta_macro_f1) / 2
            np.testing.assert_allclose([even + odd, even - odd], [positive.delta_macro_f1, negative.delta_macro_f1])
            parity_rows.append({"concept":concept, "variant":variant, "requested_absolute_dose":dose,
                                "even_delta_f1":even, "odd_delta_f1":odd,
                                "negative_actual_dose":negative.mean_actual_dose,
                                "positive_actual_dose":positive.mean_actual_dose})
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(tables / "edit_parity.csv", index=False)
    dec.to_csv(tables / "decodability.csv",index=False)
    pd.DataFrame({"class":LABELS,"f1":per_class,"test_count":test.label.value_counts().reindex(LABELS).to_numpy()}).to_csv(tables / "class_performance.csv",index=False)
    history = pd.read_csv(work / f"seed{args.seed}/model.history.csv")
    best_row = history.loc[history.val_macro_f1.idxmax()]
    if int(best_row.epoch) != metrics["best_epoch"]:
        raise ValueError("Recorded best epoch does not match validation history")
    fig,ax = plt.subplots(figsize=(7.5,2.8),layout="constrained")
    ax.plot(history.epoch,history.train_macro_f1,label="Training")
    ax.plot(history.epoch,history.val_macro_f1,label="Validation")
    ax.axvline(metrics["best_epoch"],color="grey",linestyle="--",lw=1,label="Selected epoch")
    ax.set(xlabel="Epoch",ylabel="Macro-F1",ylim=(0,1));ax.legend(fontsize=8)
    fig.savefig(figures / "learning_curve.pdf")
    fig.savefig(figures / "learning_curve.png",dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,2.8),layout="constrained")
    x=np.arange(5)
    for j,(field,label) in enumerate([("r2_trained","Trained GINE"),("r2_untrained_mean","Untrained GINE (5 initializations)"),("r2_raw_other_features","Other raw features")]):
        ax.bar(x+(j-1)*.25,dec[field],.25,label=label)
    ax.set(xticks=x,xticklabels=NAMES,ylabel="Descriptive $R^2$",ylim=(0,1.06));ax.legend(fontsize=7,loc="lower right")
    fig.savefig(figures / "decodability.pdf")
    fig.savefig(figures / "decodability.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3),layout="constrained")
    for variant,ax in zip(("isolated","coherent"),axes):
        for concept,name in zip(CONCEPTS,NAMES):
            d=interventions.query("variant == @variant and concept == @concept").sort_values("alpha")
            d=pd.concat([d,pd.DataFrame([{"alpha":0.,"delta_macro_f1":0.,"patient_ci95_low":0.,"patient_ci95_high":0.}])]).sort_values("alpha")
            ax.plot(d.alpha,d.delta_macro_f1,"o-",ms=3,label=name)
            ax.fill_between(d.alpha,d.patient_ci95_low,d.patient_ci95_high,alpha=.09)
        ax.axhline(0,color="grey",lw=.7);ax.set(title="Isolated edits" if variant=="isolated" else "Correlation-propagated edits",xlabel="Requested dose (TRAIN SD)",ylabel="Change in Macro-F1")
    axes[1].legend(fontsize=7)
    fig.savefig(figures / "interventions.pdf")
    fig.savefig(figures / "interventions.png",dpi=150)
    plt.close(fig)
    evidence={"protocol":split_manifest["protocol"],"source_run":str(run.relative_to(ROOT)),"seed":args.seed,
              "trained_checkpoints_reported":1,"checkpoint_sha256":manifest["sha256"],
              "counts":split_manifest["counts"],"all_pairwise_id_overlaps":overlaps,
              "test_macro_f1_recomputed":base_f1,"test_accuracy_recomputed":metrics["test"]["accuracy"],
              "patient_bootstrap_ci95":[ci_low,ci_high],"n_boot":args.n_boot,"bootstrap_seed":42,
              "bootstrap_unit":"patient","n_test_patients":n_patients,
              "checks":"All 60 variant/dose F1, logit, actual-dose and clipping means verified against arrays; ordered labels and IDs, dose-zero identity, partition and checkpoint hashes verified.",
              "scope":"Single trained seed; patient-disjoint source-restored test on available graphs; prior exploratory exposure; descriptor sensitivity only."}
    evidence["cav_tcav"] = cav_evidence
    evidence["untrained_classification"] = {
        "n_random_initializations": 5,
        "random_macro_f1_mean": float(mean_random.macro_f1.mean()),
        "random_macro_f1_sd": float(mean_random.macro_f1.std(ddof=1)),
        "random_macro_f1_range": [float(mean_random.macro_f1.min()), float(mean_random.macro_f1.max())],
        "trained_macro_f1": float(classification.loc[classification.model == "trained_seed42", "macro_f1"].iloc[0]),
        "trained_minus_random_mean_paired_ci95": [float(comparison.loc["trained_minus_mean_of_five_untrained", "paired_patient_bootstrap_95_low"]), float(comparison.loc["trained_minus_mean_of_five_untrained", "paired_patient_bootstrap_95_high"])],
        "trained_minus_majority_paired_ci95": [float(comparison.loc["trained_minus_train_majority_constant", "paired_patient_bootstrap_95_low"]), float(comparison.loc["trained_minus_train_majority_constant", "paired_patient_bootstrap_95_high"])],
        "checks": "Checkpoint and split hashes, ordered IDs, labels, all seven accuracy/macro-F1 values recomputed from saved predictions; paired contrasts computed on shared patient-bootstrap resamples.",
        "scope": "Five frozen random initializations; intervals condition on the fitted fixed predictors and test patients."
    }
    paths=[run/"manifest.json",run/"interventions_test.npz",run/"interventions_test.csv",run/"decodability_baselines.csv",
           work/"split_manifest.json",work/f"seed{args.seed}/model.metrics.json",work/f"seed{args.seed}/model.history.csv",
           ROOT/"data/bracs/BRACS.xlsx",*[work/f"{s}_ids.csv" for s in frames]]
    paths += [cav_dir/name for name in ("manifest.json", "cav_scores.csv", "tcav_scores.csv", "directions_and_sensitivities.npz", "validation_representations.npz", "test_representations.npz")]
    paths += [classification_dir/name for name in ("manifest.json", "classification_metrics.csv", "paired_comparisons.csv", "test_predictions.npz")]
    evidence["source_hashes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    evidence["figures"]={name:{extension+"_sha256":hashlib.sha256((figures/f"{name}.{extension}").read_bytes()).hexdigest()
                              for extension in ("pdf","png")}
                         for name in ("learning_curve","decodability","interventions","tcav","cav_auc")}
    (output/"verification.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    probe_rows="\n".join(f"{name} & {r.r2_trained:.3f} & {r.r2_untrained_mean:.3f} & {r.r2_raw_other_features:.3f} & {r.excess_over_untrained:+.3f}"+r"\\" for name,r in zip(NAMES,dec.itertuples()))
    dose_rows=[]
    for concept,name in zip(CONCEPTS,NAMES):
        a=interventions.query("concept==@concept and variant=='isolated' and alpha==-1").iloc[0]
        b=interventions.query("concept==@concept and variant=='isolated' and alpha==1").iloc[0]
        dose_rows.append(f"{name} & {a.delta_macro_f1:+.3f} & {b.delta_macro_f1:+.3f} & {100*a.mean_clipped_fraction:.1f} & {a.mean_actual_dose:.3f}"+r"\\")
    class_rows="\n".join(f"{label} & {n} & {score:.3f}"+r"\\" for label,n,score in zip(LABELS,test.label.value_counts().reindex(LABELS),per_class))
    split_rows="\n".join(f"{s.capitalize()} & {c['graphs']} & {c['slides']} & {c['groups']}"+r"\\" for s,c in split_manifest["counts"].items())
    replacements={"@F1@":f"{base_f1:.4f}","@ACCURACY@":f"{metrics['test']['accuracy']:.4f}",
                  "@VALF1@":f"{metrics['validation']['macro_f1']:.4f}","@BESTEPOCH@":metrics["best_epoch"],
                  "@LOW@":f"{ci_low:.3f}","@HIGH@":f"{ci_high:.3f}","@PROBES@":probe_rows,
                  "@DOSES@":"\n".join(dose_rows),"@CLASSES@":class_rows,"@SPLITS@":split_rows}
    replacements["@NBOOT@"] = f"{args.n_boot:,}".replace(",", r"\,")
    auc_table = cav.pivot(index="concept", columns="split", values="auc")
    replacements["@CAVROWS@"] = "\n".join(f"{name} & {auc_table.loc[concept, 'validation']:.4f} & {auc_table.loc[concept, 'test']:.4f}"+r"\\" for concept,name in zip(CONCEPTS,NAMES))
    replacements["@TCAVPMIN@"] = f"{tcav.query('split == \"test\"').control_reference_p.min():.3f}"
    distance_parity = parity.query("concept == 'spatial_distance' and variant == 'isolated' and requested_absolute_dose == 1").iloc[0]
    replacements["@DISTANCEEVEN@"] = f"{distance_parity.even_delta_f1:+.4f}"
    replacements["@DISTANCEODD@"] = f"{distance_parity.odd_delta_f1:+.4f}"
    replacements["@RANDOMF1MEAN@"] = f"{mean_random.macro_f1.mean():.4f}"
    replacements["@RANDOMF1SD@"] = f"{mean_random.macro_f1.std(ddof=1):.4f}"
    replacements["@RANDOMF1LOW@"] = f"{mean_random.macro_f1.min():.4f}"
    replacements["@RANDOMF1HIGH@"] = f"{mean_random.macro_f1.max():.4f}"
    replacements["@RANDOMACCMEAN@"] = f"{mean_random.accuracy.mean():.4f}"
    replacements["@RANDOMACCLOW@"] = f"{mean_random.accuracy.min():.4f}"
    replacements["@RANDOMACCHIGH@"] = f"{mean_random.accuracy.max():.4f}"
    majority_row = classification.loc[classification.model == "train_majority_constant"].iloc[0]
    trained_row = classification.loc[classification.model == "trained_seed42"].iloc[0]
    random_diff = comparison.loc["trained_minus_mean_of_five_untrained"]
    majority_diff = comparison.loc["trained_minus_train_majority_constant"]
    replacements["@MAJORITYF1@"] = f"{majority_row.macro_f1:.4f}"
    replacements["@TRAINEDF1CLASS@"] = f"{trained_row.macro_f1:.4f}"
    replacements["@TRAINEDACCCLASS@"] = f"{trained_row.accuracy:.4f}"
    replacements["@RANDOMDIFF@"] = f"{random_diff.delta_macro_f1:.4f}"
    replacements["@RANDOMDIFFLOW@"] = f"{random_diff.paired_patient_bootstrap_95_low:.4f}"
    replacements["@RANDOMDIFFHIGH@"] = f"{random_diff.paired_patient_bootstrap_95_high:.4f}"
    replacements["@MAJORITYDIFF@"] = f"{majority_diff.delta_macro_f1:.4f}"
    replacements["@MAJORITYDIFFLOW@"] = f"{majority_diff.paired_patient_bootstrap_95_low:.4f}"
    replacements["@MAJORITYDIFFHIGH@"] = f"{majority_diff.paired_patient_bootstrap_95_high:.4f}"
    tex=(output/"research_manuscript.template.tex").read_text(encoding="utf-8")
    for key,value in replacements.items():tex=tex.replace(key,str(value))
    (output/"graph_bracs_research_classification_results.tex").write_text(tex,encoding="utf-8")
    # Standalone early abstract, using exactly the manuscript's verified numbers.
    abstract=tex.split(r"\begin{abstract}",1)[1].split(r"\end{abstract}",1)[0].strip()
    abstract_tex=r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern,hyperref}
\usepackage[margin=2.2cm]{geometry}
\setlength{\emergencystretch}{2em}
\title{Auditing Concept Reliance in Histopathology Graph Neural Networks}
\author{Mahugnon Aime Loick Gohouede}
\date{Early empirical abstract --- 16 October 2026}
\begin{document}\maketitle
\begin{abstract}
"""+abstract+r"""
\end{abstract}
\paragraph{Research positioning.} This study complements concept discovery in GCExplainer and its histopathology application~\cite{gc,hist}, and is motivated by established concerns about concept entanglement and CAV reliability~\cite{cav}. Prior probing work on randomly initialized BERT also motivates this untrained-representation control, while offering no direct evidence for cell-graph GNNs~\cite{random}. The empirical contribution concerns descriptor-based cell graphs; the general distinction between decodability and reliance is not claimed as new.
\begin{thebibliography}{4}\small
\bibitem{gc} L. C. Magister et al. GCExplainer: Human-in-the-Loop Concept-based Explanations for Graph Neural Networks. 2021. \url{https://arxiv.org/abs/2107.11889}.
\bibitem{hist} A. Farace di Villaforesta et al. Digital Histopathology with Graph Neural Networks: Concepts and Explanations for Clinicians. 2023. \url{https://arxiv.org/abs/2312.02225}.
\bibitem{cav} A. Nicolson et al. Explaining Explainability: Recommendations for Effective Use of Concept Activation Vectors. TMLR, 2025. \url{https://openreview.net/forum?id=7CUluLpLxV}.
\bibitem{random} E. Hernandez and J. Andreas. The Low-Dimensional Linear Geometry of Contextualized Word Representations. CoNLL, pp. 82--93, 2021. \url{https://aclanthology.org/2021.conll-1.7/}.
\end{thebibliography}
\end{document}
"""
    (output/"graph_bracs_abstract_classification_results.tex").write_text(abstract_tex,encoding="utf-8")
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    main()
