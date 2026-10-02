"""Required guards before running corrected audits; never writes the GINE."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

import concept_audit_phase2 as phase2
from concept_audit.config import CHECKPOINT, GRAPH_CONCEPT, NODE_CONCEPTS, NODE_CONCEPT_FEATURE_NAMES
from concept_audit.data import load_context, load_frozen_model
from concept_audit.utils import encode_batch
from train_gine_baseline import resolve_graph_path


def main():
    torch.set_num_threads(4)
    data = load_context()
    phase2.INTERVENTION_SCALER = data
    profile = pd.read_csv("processed/graph-bracs/feature_stats_train.csv").set_index("feature_index")
    expected = {"cell_size": "area", "cell_shape": "circularity",
                "cell_intensity": "mean_intensity_gray_scale", "cell_texture": "mean_contrast"}
    candidates = pd.read_csv("processed/graph-bracs/concept_candidates_train.csv").set_index("sample_id")
    summaries = {c: [] for c in NODE_CONCEPTS}
    max_errors = {c: 0. for c in NODE_CONCEPTS}
    for c, idx in NODE_CONCEPTS.items():
        assert profile.loc[idx, "feature"] == expected[c], (c, idx)
    for n, row in enumerate(data["train_df"].itertuples()):
        graph = torch.load(resolve_graph_path(row.graph_path), weights_only=False)
        for c, idx in NODE_CONCEPTS.items():
            raw = graph.x[:, idx].double().numpy()
            assert np.isfinite(raw).all(), (row.sample_id, c)
            value = float(raw.mean())
            reference = float(candidates.loc[row.sample_id, NODE_CONCEPT_FEATURE_NAMES[c]])
            assert np.isclose(value, reference, rtol=1e-6, atol=1e-8), (row.sample_id, c, value, reference)
            max_errors[c] = max(max_errors[c], abs(value - reference))
            summaries[c].append(value)
        if n % 500 == 0:
            print(f"Raw mapping verified: {n + 1}/{len(data['train_df'])}", flush=True)
    device = torch.device("cpu")
    model, _ = load_frozen_model(device)
    batch = next(iter(data["val_loader"]))
    with torch.no_grad():
        _, _, baseline = encode_batch(model, batch)
        for c in [*NODE_CONCEPTS, GRAPH_CONCEPT]:
            for random in [False, True]:
                out = phase2.apply_concept_intervention(batch, c, 0., random, np.random.default_rng(42))
                for attr in ["x", "edge_attr", "edge_index", "batch", "y"]:
                    assert torch.equal(getattr(batch, attr), getattr(out, attr)), (c, attr)
                _, _, logits = encode_batch(model, out)
                assert torch.equal(baseline, logits), c
                assert torch.equal(baseline.argmax(1), logits.argmax(1)), c
            for alpha in phase2.INTERVENTION_ALPHAS:
                out = phase2.apply_concept_intervention(batch, c, float(alpha), False, np.random.default_rng(42))
                original = batch.x[:, NODE_CONCEPTS[c]] if c in NODE_CONCEPTS else batch.edge_attr[:, 0]
                changed = out.x[:, NODE_CONCEPTS[c]] if c in NODE_CONCEPTS else out.edge_attr[:, 0]
                assert torch.all((changed - original) * float(alpha) >= 0), (c, alpha)
                assert torch.equal(batch.edge_index, out.edge_index)
                if c in NODE_CONCEPTS:
                    columns = [i for i in range(batch.x.shape[1]) if i != NODE_CONCEPTS[c]]
                    assert torch.equal(batch.x[:, columns], out.x[:, columns])
                    assert torch.equal(batch.edge_attr, out.edge_attr)
                else:
                    assert torch.equal(batch.x, out.x)
    # Deliberate outliers demonstrate that the no-op precedes any clipping.
    synthetic = Batch.from_data_list([Data(x=torch.full((2, 96), 8.),
        edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.tensor([[9.], [-9.]]))])
    for c in [*NODE_CONCEPTS, GRAPH_CONCEPT]:
        for random in [False, True]:
            out = phase2.apply_concept_intervention(synthetic, c, 0., random, np.random.default_rng(42))
            assert torch.equal(out.x, synthetic.x) and torch.equal(out.edge_attr, synthetic.edge_attr)
        for alpha in [-2., 2.]:
            out = phase2.apply_concept_intervention(synthetic, c, alpha, False, np.random.default_rng(42))
            old = synthetic.x[:, NODE_CONCEPTS[c]] if c in NODE_CONCEPTS else synthetic.edge_attr[:, 0]
            new = out.x[:, NODE_CONCEPTS[c]] if c in NODE_CONCEPTS else out.edge_attr[:, 0]
            assert torch.all((new - old) * alpha >= 0)
    report = {"protocol": "v2", "train_graphs_checked": len(data["train_df"]),
              "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "identity_features_edges_logits_predictions": "passed",
              "monotonic_doses_and_unchanged_non_targets": "passed",
              "spatial_concept": GRAPH_CONCEPT,
              "concepts": {c: {"index": NODE_CONCEPTS[c], "feature": expected[c],
                  "graph_mean_q20": float(np.quantile(v, .2)), "graph_mean_q80": float(np.quantile(v, .8)),
                  "max_error_against_prior_profiling": max_errors[c]} for c, v in summaries.items()}}
    output = Path("results/audit_v2_preflight.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
