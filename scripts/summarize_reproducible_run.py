"""Export results of completed, independently trained runs without invented estimates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_completed_report import bootstrap_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default="results/reproducible_run")
    args = ap.parse_args()
    work = Path(args.work_dir)
    split = json.loads((work / "split_manifest.json").read_text())
    metrics, probes, interventions, seen = [], [], [], set()
    for p in sorted(work.glob("seed*/audit/manifest.json")):
        m = json.loads(p.read_text())
        if m.get("status") != "completed":
            raise ValueError(f"Incomplete audit: {p.parent}")
        if m.get("descriptive_only") or not m.get("training_ids_verified") or any(m["leakage"].values()):
            raise ValueError(f"Invalid provenance or overlap: {p.parent}")
        if m["sha256"] in seen:
            raise ValueError("A repeated checkpoint is not an independent training run")
        seen.add(m["sha256"])
        expected = {s: set(pd.read_csv(work / f"{s}_ids.csv").sample_id)
                    for s in ("train", "validation", "test")}
        for s, ids in expected.items():
            if set(m["sample_ids"][s]) != ids:
                raise ValueError(f"Run uses different frozen {s} IDs: {p.parent}")
        score = json.loads((p.parent / "classification_test.json").read_text())
        metrics.append({"seed": m["seed"], "macro_f1": score["macro_f1"],
                        "slide_ci_low": score["slide_bootstrap_ci95"][0],
                        "slide_ci_high": score["slide_bootstrap_ci95"][1]})
        test_frame = pd.read_csv(work / "test_ids.csv", dtype={"patient_id": str})
        if "patient_id" in test_frame:
            _, group_index = np.unique(test_frame.patient_id, return_inverse=True)
            n_groups = len(np.unique(group_index))
            weights = np.random.default_rng(42).multinomial(n_groups, np.full(n_groups,1/n_groups),size=2000)
            with np.load(p.parent / "interventions_test.npz") as arrays:
                if not np.array_equal(arrays["sample_ids"],test_frame.sample_id.to_numpy(dtype=str)):
                    raise ValueError("Patient bootstrap requires aligned sample IDs")
                values = bootstrap_f1(arrays["y"],arrays["base_pred"],group_index,weights)
            lo,hi = np.quantile(values,[.025,.975])
            metrics[-1].update(patient_ci_low=lo,patient_ci_high=hi)
        probes.append(pd.read_csv(p.parent / "decodability_baselines.csv").assign(seed=m["seed"]))
        interventions.append(pd.read_csv(p.parent / "interventions_test.csv").assign(seed=m["seed"]))
    if not metrics:
        raise SystemExit("No completed new-model audits; no results can be exported yet.")
    out = work / "summary"
    out.mkdir(exist_ok=True)
    scores = pd.DataFrame(metrics)
    if scores.seed.duplicated().any():
        raise ValueError("Repeated training seed")
    scores.to_csv(out / "classifier_scores.csv", index=False)
    pd.concat(probes).to_csv(out / "decodability_by_seed.csv", index=False)
    iv = pd.concat(interventions)
    iv.to_csv(out / "interventions_by_seed.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.bar(scores.seed.astype(str), scores.macro_f1)
    ax.set(xlabel="Independent training seed", ylabel="Internal holdout Macro-F1", ylim=(0,1))
    fig.savefig(out / "classifier_scores.pdf");plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    means = iv.groupby(["concept", "alpha"]).isolated_dF1.mean().reset_index()
    for c, d in means.groupby("concept"):
        d = pd.concat([d, pd.DataFrame([{"alpha":0.,"isolated_dF1":0.}])]).sort_values("alpha")
        ax.plot(d.alpha, d.isolated_dF1, "o-", label=c)
    ax.axhline(0, color="grey", lw=.7)
    ax.set(xlabel="Requested dose (TRAIN SD)", ylabel="Mean change in Macro-F1")
    ax.legend(fontsize=8);fig.savefig(out / "intervention_response.pdf");plt.close(fig)
    n = len(scores)
    sd = f"{scores.macro_f1.std(ddof=1):.4f}" if n > 1 else "not estimable from one training run"
    evaluation_note = ("Original BRACS source partitions restored on available graphs. Prior exploratory exposure means this is not an untouched confirmatory test."
                       if split['protocol'] == 'restored_bracs_partitions_available_graphs' else
                       "This is a newly partitioned internal study, not an external or untouched confirmatory test.")
    text = f"""# Executed Graph-BRACS experiment

Protocol: {split['protocol']}. Grouping: {split['grouping']}.
Patient independence verified from supplied mapping: {split['patient_independence_verified']}.
Completed independent training seeds: {n} ({', '.join(map(str, scores.seed))}).
Mean internal holdout Macro-F1: {scores.macro_f1.mean():.4f}.
Standard deviation across training runs: {sd}.

All retained runs passed graph-ID and slide-separation checks and record the training IDs.
Patient bootstrap intervals (when patient IDs are available) and historical slide intervals are
separately named in classifier_scores.csv. Both condition on each trained model; they do not
measure variability across independently trained models.
{evaluation_note}
Descriptor-edit controls remain exploratory; propagated edits are not biologically validated,
and edge edits do not have matched edge-feature controls. Historical results are not mixed in.

Figures: classifier_scores.pdf and intervention_response.pdf.
Tables: classifier_scores.csv, decodability_by_seed.csv, interventions_by_seed.csv.
"""
    (out / "executed_results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
