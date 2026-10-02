
"""
Graph-BRACS GCIA — Phase 2
=================================

Phase 2 adds:
1. CAVs: binary linear concept classifiers on frozen GINE representations.
2. TCAV: class-conditional directional sensitivity of the frozen GINE
   to each CAV.
3. Random-CAV controls.
4. Controlled input-space concept interventions with dose-response.
5. Matched random-sign interventions with the same per-element magnitude.

Important:
- CAV/TCAV are computed from the frozen model; no model retraining.
- CAVs are learned on TRAIN only.
- TCAV and intervention results are reported on VALIDATION and OFFICIAL TEST.
- The continuous concepts used here are:
    cell_size       -> feature 23
    cell_shape     -> feature 27
    cell_intensity -> feature 90
    cell_texture   -> feature 55
    spatial_distance -> mean edge distance
- Positive spatial_distance means larger edge distance; connectivity is fixed.

Run from project root:
    python scripts/concept_audit_phase2.py

Output:
    results/concept_audit_phase2_v2.npz
"""

from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from torch_geometric.nn import global_mean_pool, global_max_pool

from concept_audit.config import (
    GRAPH_CONCEPT,
    NODE_CONCEPTS,
    NODE_CONCEPT_FEATURE_NAMES,
    CHECKPOINT,
    TRAIN_CSV,
    VAL_CSV,
    METADATA_CSV,
    SCALER_PATH,
    BATCH_SIZE,
    EXPECTED_TRAIN,
    EXPECTED_VAL,
    EXPECTED_TEST,
    SEED,
)
from concept_audit.data import load_context, load_frozen_model
from concept_audit.utils import (
    encode_batch,
    graph_mean_edge_distance,
    sample_nodes,
    set_seed,
    require_v2_preflight,
)


# ---------------------------------------------------------------------
# Phase-2 configuration
# ---------------------------------------------------------------------

OUTPUT = Path("results/concept_audit_phase2_v2.npz")

# High/low concept examples for the CAV:
# q20 = non-concept / low-value side
# q80 = concept / high-value side
POSITIVE_QUANTILE = 0.80
NEGATIVE_QUANTILE = 0.20

# Maximum node activations retained for CAV fitting per graph.
MAX_CAV_NODES_PER_GRAPH = 50

# Random-label CAV controls.
NUM_RANDOM_CAVS = 10
NUM_RANDOM_INTERVENTIONS = 10

# Standardized input-space intervention doses.
# For node features, alpha means alpha training-set standard deviations.
# For edge distance, alpha means alpha training-set edge-distance SDs.
INTERVENTION_ALPHAS = np.asarray(
    [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    dtype=np.float64,
)

# To avoid pathological out-of-distribution interventions, clip the
# standardized input feature after intervention to this range.
INTERVENTION_CLIP = 5.0


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def raw_node_values(batch, feature_idx, x_mean, x_std):
    """Recover raw feature values from the standardized graph batch."""
    return (
        batch.x[:, feature_idx] * float(x_std[feature_idx])
        + float(x_mean[feature_idx])
    )


def fit_cav(activations, binary_labels, seed):
    """
    Fit a logistic CAV.

    The classifier is trained on standardized activation coordinates.
    Its coefficient vector is transformed back into the original
    activation coordinates and normalized to unit length.

    Returns:
        direction: unit CAV in the original activation space
        accuracy, macro_f1, auc
    """
    X = np.asarray(activations, dtype=np.float64)
    y = np.asarray(binary_labels, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError(f"CAV X must be 2D, got {X.shape}")

    if len(np.unique(y)) != 2:
        raise ValueError("CAV requires both positive and negative examples.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(Xs, y)

    pred = clf.predict(Xs)
    prob = clf.predict_proba(Xs)[:, 1]

    # beta_std is normal to the separating hyperplane in standardized space.
    beta_std = clf.coef_[0].astype(np.float64)

    # Transform the normal vector back to original coordinates.
    # If x_s = (x - mean) / scale, beta_s^T x_s =
    # (beta_s / scale)^T x + const.
    scale = np.where(scaler.scale_ < 1e-12, 1.0, scaler.scale_)
    beta_raw = beta_std / scale

    norm = np.linalg.norm(beta_raw)
    if norm < 1e-12:
        raise RuntimeError("CAV has near-zero norm.")

    direction = beta_raw / norm

    return {
        "direction": direction,
        "classifier": clf,
        "activation_scaler": scaler,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, prob)),
        "n": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }


def make_balanced_quantile_set(X, targets, seed):
    """
    Construct a balanced high-vs-low concept dataset.

    High values (>= q80) are the positive concept side.
    Low values (<= q20) are the negative side.

    Middle values are excluded from CAV fitting.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    q_low = np.quantile(y, NEGATIVE_QUANTILE)
    q_high = np.quantile(y, POSITIVE_QUANTILE)

    pos = np.flatnonzero(y >= q_high)
    neg = np.flatnonzero(y <= q_low)

    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Quantile concept split produced an empty side.")

    n = min(len(pos), len(neg))

    rng = np.random.default_rng(seed)
    pos = rng.choice(pos, size=n, replace=False)
    neg = rng.choice(neg, size=n, replace=False)

    idx = np.concatenate([pos, neg])
    labels = np.concatenate(
        [
            np.ones(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
        ]
    )

    order = rng.permutation(len(idx))

    return (
        X[idx[order]],
        labels[order],
        float(q_low),
        float(q_high),
    )


def graph_target_values(batch, edge_mean, edge_std):
    return graph_mean_edge_distance(
        batch,
        edge_mean,
        edge_std,
    ).detach().cpu().numpy()


def get_graph_sizes(batch):
    return torch.bincount(
        batch.batch,
        minlength=batch.num_graphs,
    )


# ---------------------------------------------------------------------
# CAV fitting data
# ---------------------------------------------------------------------

def collect_cav_training_data(
    model,
    loader,
    x_mean,
    x_std,
    edge_mean,
    edge_std,
    device,
):
    """
    Collect train-only node and graph representations and concept targets.
    """
    node_names = list(NODE_CONCEPTS.keys())

    node_H = {name: [] for name in node_names}
    node_Y = {name: [] for name in node_names}

    graph_Z = []
    graph_Y = []

    processed = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            h, z, _ = encode_batch(model, batch)

            selected = sample_nodes(
                batch,
                MAX_CAV_NODES_PER_GRAPH,
            )

            H = h[selected].detach().cpu().numpy()

            for concept in node_names:
                idx = NODE_CONCEPTS[concept]
                target = raw_node_values(
                    batch,
                    idx,
                    x_mean,
                    x_std,
                )[selected]

                node_H[concept].append(H)
                node_Y[concept].append(
                    target.detach().cpu().numpy()
                )

            graph_Z.append(z.detach().cpu().numpy())
            graph_Y.append(
                graph_target_values(
                    batch,
                    edge_mean,
                    edge_std,
                )
            )

            processed += batch.num_graphs

            if processed % 250 < batch.num_graphs or processed == len(loader.dataset):
                print(
                    f"  collected {processed:,}/{len(loader.dataset):,}"
                )

    for concept in node_names:
        node_H[concept] = np.concatenate(
            node_H[concept],
            axis=0,
        )
        node_Y[concept] = np.concatenate(
            node_Y[concept],
            axis=0,
        )

    graph_Z = np.concatenate(graph_Z, axis=0)
    graph_Y = np.concatenate(graph_Y, axis=0)

    return node_H, node_Y, graph_Z, graph_Y


def fit_all_cavs(
    node_H,
    node_Y,
    graph_Z,
    graph_Y,
):
    """
    Fit the real CAV and random-label CAVs for every concept.
    """
    cavs = {}

    for concept in NODE_CONCEPTS:
        X, labels, q_low, q_high = make_balanced_quantile_set(
            node_H[concept],
            node_Y[concept],
            SEED + 100 + list(NODE_CONCEPTS).index(concept),
        )

        real = fit_cav(
            X,
            labels,
            SEED + 200 + list(NODE_CONCEPTS).index(concept),
        )

        random_dirs = []

        rng = np.random.default_rng(
            SEED + 10000 + list(NODE_CONCEPTS).index(concept)
        )

        for r in range(NUM_RANDOM_CAVS):
            random_labels = rng.permutation(labels)
            rnd = fit_cav(
                X,
                random_labels,
                SEED + 20000 + r,
            )
            random_dirs.append(rnd["direction"])

        cavs[concept] = {
            "level": "node",
            "direction": real["direction"],
            "classifier": real["classifier"],
            "activation_scaler": real["activation_scaler"],
            "random_directions": np.asarray(random_dirs),
            "probe_accuracy": real["accuracy"],
            "probe_macro_f1": real["macro_f1"],
            "probe_auc": real["auc"],
            "n": real["n"],
            "n_positive": real["n_positive"],
            "n_negative": real["n_negative"],
            "q20": q_low,
            "q80": q_high,
        }

    X, labels, q_low, q_high = make_balanced_quantile_set(
        graph_Z,
        graph_Y,
        SEED + 500,
    )

    real = fit_cav(
        X,
        labels,
        SEED + 600,
    )

    random_dirs = []

    rng = np.random.default_rng(SEED + 10500)

    for r in range(NUM_RANDOM_CAVS):
        random_labels = rng.permutation(labels)
        rnd = fit_cav(
            X,
            random_labels,
            SEED + 20500 + r,
        )
        random_dirs.append(rnd["direction"])

    cavs[GRAPH_CONCEPT] = {
        "level": "graph",
        "direction": real["direction"],
            "classifier": real["classifier"],
            "activation_scaler": real["activation_scaler"],
        "random_directions": np.asarray(random_dirs),
        "probe_accuracy": real["accuracy"],
        "probe_macro_f1": real["macro_f1"],
        "probe_auc": real["auc"],
        "n": real["n"],
        "n_positive": real["n_positive"],
        "n_negative": real["n_negative"],
        "q20": q_low,
        "q80": q_high,
    }

    return cavs


# ---------------------------------------------------------------------
# TCAV
# ---------------------------------------------------------------------

def tcav_on_split(
    model,
    loader,
    cavs,
    device,
):
    """
    Compute class-conditional TCAV.

    For node concepts:
      - CAV lives in the node hidden representation h.
      - We compute d logit_k / d h_i for the true class k.
      - Per graph sensitivity is the mean over its nodes:
            mean_i <grad_h_i, CAV>
      - TCAV is the fraction of class-k graphs with positive sensitivity.

    For spatial_distance:
      - CAV lives in graph representation z.
      - Sensitivity is <grad_z, CAV>.

    Random CAVs are scored using the same gradients.
    """
    concepts = list(NODE_CONCEPTS.keys()) + [GRAPH_CONCEPT]
    num_classes = 7

    values = {
        concept: {
            "sensitivity": {k: [] for k in range(num_classes)},
            "random_sensitivity": {
                k: [[] for _ in range(NUM_RANDOM_CAVS)]
                for k in range(num_classes)
            },
        }
        for concept in concepts
    }

    processed = 0

    for batch in loader:
        batch = batch.to(device)

        # The GINE is frozen, so its representation does not require grad.
        # TCAV only needs d(logit)/d(representation), not gradients through
        # the GINE parameters. Detach the final node representation, make
        # it a gradient-tracked leaf, then recompute pooling + classifier.
        h, _, _ = encode_batch(model, batch)
        h = h.detach().requires_grad_(True)

        mean_pool = global_mean_pool(h, batch.batch)
        max_pool = global_max_pool(h, batch.batch)
        z = torch.cat([mean_pool, max_pool], dim=1)
        logits = model.classifier(z)

        true_logits = logits[
            torch.arange(batch.num_graphs, device=device),
            batch.y,
        ]

        # Compute the two required Jacobian-vector ingredients separately.
        # This avoids treating z as an independent input while also
        # differentiating through z back to the node representation h.
        grad_h = torch.autograd.grad(
            true_logits.sum(),
            h,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        grad_z = torch.autograd.grad(
            true_logits.sum(),
            z,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        # Map every node to its graph and precompute node counts.
        # These are required to aggregate node-level directional
        # sensitivities back to one sensitivity value per graph.
        graph_ids = batch.batch
        node_counts = torch.bincount(
            graph_ids,
            minlength=batch.num_graphs,
        ).to(dtype=h.dtype)

        # We need to store all concepts after the gradients exist.
        for concept in NODE_CONCEPTS:
            cav = torch.as_tensor(
                cavs[concept]["direction"],
                dtype=h.dtype,
                device=device,
            )

            node_s = (grad_h * cav.unsqueeze(0)).sum(dim=1)
            sums = torch.zeros(
                batch.num_graphs,
                dtype=node_s.dtype,
                device=device,
            )
            sums.scatter_add_(0, graph_ids, node_s)

            sensitivity = sums / torch.clamp(
                node_counts,
                min=1.0,
            )

            for local_i in range(batch.num_graphs):
                k = int(batch.y[local_i].item())
                values[concept]["sensitivity"][k].append(
                    float(sensitivity[local_i].detach().cpu())
                )

            for r in range(NUM_RANDOM_CAVS):
                d = torch.as_tensor(
                    cavs[concept]["random_directions"][r],
                    dtype=h.dtype,
                    device=device,
                )
                rs = (grad_h * d.unsqueeze(0)).sum(dim=1)

                rsum = torch.zeros(
                    batch.num_graphs,
                    dtype=rs.dtype,
                    device=device,
                )
                rsum.scatter_add_(0, graph_ids, rs)

                rmean = rsum / torch.clamp(
                    node_counts,
                    min=1.0,
                )

                for local_i in range(batch.num_graphs):
                    k = int(batch.y[local_i].item())
                    values[concept]["random_sensitivity"][k][r].append(
                        float(rmean[local_i].detach().cpu())
                    )

        # Graph concept.
        cav = torch.as_tensor(
            cavs[GRAPH_CONCEPT]["direction"],
            dtype=z.dtype,
            device=device,
        )

        sensitivity = (grad_z * cav.unsqueeze(0)).sum(dim=1)

        for local_i in range(batch.num_graphs):
            k = int(batch.y[local_i].item())
            values[GRAPH_CONCEPT]["sensitivity"][k].append(
                float(sensitivity[local_i].detach().cpu())
            )

        for r in range(NUM_RANDOM_CAVS):
            d = torch.as_tensor(
                cavs[GRAPH_CONCEPT]["random_directions"][r],
                dtype=z.dtype,
                device=device,
            )
            rs = (grad_z * d.unsqueeze(0)).sum(dim=1)

            for local_i in range(batch.num_graphs):
                k = int(batch.y[local_i].item())
                values[GRAPH_CONCEPT]["random_sensitivity"][k][r].append(
                    float(rs[local_i].detach().cpu())
                )

        processed += batch.num_graphs
        if processed % 250 < batch.num_graphs or processed == len(loader.dataset):
            print(f"  TCAV processed {processed:,}/{len(loader.dataset):,}")

    # Convert sensitivities to compact metrics.
    output = {}

    for concept in concepts:
        output[concept] = {
            "tcav": np.full(num_classes, np.nan, dtype=np.float64),
            "mean_sensitivity": np.full(num_classes, np.nan, dtype=np.float64),
            "n": np.zeros(num_classes, dtype=np.int64),
            "random_tcav_mean": np.full(num_classes, np.nan, dtype=np.float64),
            "random_tcav_std": np.full(num_classes, np.nan, dtype=np.float64),
            "tcav_minus_random": np.full(num_classes, np.nan, dtype=np.float64),
            "random_tcav": np.full(
                (num_classes, NUM_RANDOM_CAVS),
                np.nan,
                dtype=np.float64,
            ),
        }

        for k in range(num_classes):
            s = np.asarray(
                values[concept]["sensitivity"][k],
                dtype=np.float64,
            )

            if len(s) == 0:
                continue

            random_scores = []

            for r in range(NUM_RANDOM_CAVS):
                rs = np.asarray(
                    values[concept]["random_sensitivity"][k][r],
                    dtype=np.float64,
                )
                random_scores.append(
                    np.mean(rs > 0.0)
                )

            random_scores = np.asarray(
                random_scores,
                dtype=np.float64,
            )

            output[concept]["tcav"][k] = np.mean(s > 0.0)
            output[concept]["mean_sensitivity"][k] = np.mean(s)
            output[concept]["n"][k] = len(s)
            output[concept]["random_tcav_mean"][k] = np.mean(random_scores)
            output[concept]["random_tcav_std"][k] = np.std(
                random_scores,
                ddof=1 if len(random_scores) > 1 else 0,
            )
            output[concept]["tcav_minus_random"][k] = (
                output[concept]["tcav"][k]
                - output[concept]["random_tcav_mean"][k]
            )
            output[concept]["random_tcav"][k, :] = random_scores

    return output


# ---------------------------------------------------------------------
# Controlled input-space interventions
# ---------------------------------------------------------------------

def clone_batch(batch):
    return batch.clone()


def apply_concept_intervention(
    batch,
    concept,
    alpha,
    random_control,
    rng,
):
    """
    Apply an input-space intervention.

    For continuous node concepts:
        x[:, feature] += alpha
    because x is standardized with the train-only scaler.

    For spatial distance:
        edge_attr[:, 0] += alpha

    Random matched control:
        same absolute per-element perturbation magnitude,
        but independent random +/- signs.

    This tests directional response without retraining.
    """
    out = clone_batch(batch)
    if float(alpha) == 0.0:
        return out

    if concept in NODE_CONCEPTS:
        idx = NODE_CONCEPTS[concept]

        if random_control:
            signs = rng.choice(
                np.asarray([-1.0, 1.0]),
                size=out.x.shape[0],
            )
            delta = torch.as_tensor(
                signs * abs(alpha),
                dtype=out.x.dtype,
                device=out.x.device,
            )
        else:
            delta = torch.full(
                (out.x.shape[0],),
                float(alpha),
                dtype=out.x.dtype,
                device=out.x.device,
            )

        out.x[:, idx] = bounded_shift(out.x[:, idx], delta, concept)

    elif concept == GRAPH_CONCEPT:
        if random_control:
            signs = rng.choice(
                np.asarray([-1.0, 1.0]),
                size=out.edge_attr.shape[0],
            )
            delta = torch.as_tensor(
                signs * abs(alpha),
                dtype=out.edge_attr.dtype,
                device=out.edge_attr.device,
            )
        else:
            delta = torch.full(
                (out.edge_attr.shape[0],),
                float(alpha),
                dtype=out.edge_attr.dtype,
                device=out.edge_attr.device,
            )

        out.edge_attr[:, 0] = bounded_shift(out.edge_attr[:, 0], delta, concept)

    else:
        raise KeyError(concept)

    return out


def bounded_shift(original, delta, concept):
    # Expand clipping bounds to include the starting point: zero is an identity
    # and a positive dose cannot decrease a pre-existing outlier.
    lo, hi = -INTERVENTION_CLIP, INTERVENTION_CLIP
    if INTERVENTION_SCALER is None:
        raise RuntimeError("Configure the TRAIN scaler before interventions")
    if concept in NODE_CONCEPTS:
        idx = NODE_CONCEPTS[concept]
        mean, std = INTERVENTION_SCALER["x_mean"][idx], INTERVENTION_SCALER["x_std"][idx]
    else:
        mean, std = INTERVENTION_SCALER["edge_mean"], INTERVENTION_SCALER["edge_std"]
    lo = max(lo, float(-mean / std))
    if concept == "cell_intensity":
        hi = min(hi, float((255.0 - mean) / std))
    lower = torch.minimum(original, torch.full_like(original, lo))
    upper = torch.maximum(original, torch.full_like(original, hi))
    return torch.maximum(lower, torch.minimum(upper, original + delta))


INTERVENTION_SCALER = None


def evaluate_cavs(model, loader, cavs, data, device):
    H, Y, Z, G = collect_cav_training_data(
        model, loader, data["x_mean"], data["x_std"],
        data["edge_mean"], data["edge_std"], device)
    metrics = {}
    for concept, cav in cavs.items():
        X, target = (H[concept], Y[concept]) if concept in NODE_CONCEPTS else (Z, G)
        mask = (target <= cav["q20"]) | (target >= cav["q80"])
        y = (target[mask] >= cav["q80"]).astype(np.int64)
        if len(y) == 0:
            raise RuntimeError(f"No held-out CAV examples for {concept}")
        Xs = cav["activation_scaler"].transform(X[mask])
        clf = cav["classifier"]
        pred, prob = clf.predict(Xs), clf.predict_proba(Xs)[:, 1]
        metrics[concept] = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
            "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
            "n": len(y), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        }
        print(f"  held-out {concept}: {metrics[concept]}")
    return metrics


def evaluate_interventions(model, loader, concepts, device):
    # Aggregate globally; preserve the full null distribution across repeats.
    keys = ["accuracy", "macro_f1", "true_logit_delta", "true_prob_delta", "flip_rate",
            "requested_change", "actual_change", "requested_abs_change", "actual_abs_change", "clipping_rate"]
    records = {(c, mode, r, i): {k: [] for k in ["y", "pred", "logit", "prob", "flip"]}
               for c in concepts for mode in ["concept", "random"]
               for r in range(1 if mode == "concept" else NUM_RANDOM_INTERVENTIONS)
               for i in range(len(INTERVENTION_ALPHAS))}
    for rec in records.values():
        rec.update(count=0, requested=0., actual=0., requested_abs=0., actual_abs=0., clipped=0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            _, _, base = encode_batch(model, batch)
            base_prob = base.softmax(dim=1)
            row = torch.arange(batch.num_graphs, device=device)
            y = batch.y.cpu().tolist()
            for ci, concept in enumerate(concepts):
                original = batch.x[:, NODE_CONCEPTS[concept]] if concept in NODE_CONCEPTS else batch.edge_attr[:, 0]
                for mode in ["concept", "random"]:
                    for r in range(1 if mode == "concept" else NUM_RANDOM_INTERVENTIONS):
                        seed = SEED + 50000 + batch_idx * 10000 + ci * 100 + r
                        for i, alpha in enumerate(INTERVENTION_ALPHAS):
                            modified = apply_concept_intervention(batch, concept, float(alpha), mode == "random", np.random.default_rng(seed))
                            actual_values = modified.x[:, NODE_CONCEPTS[concept]] if concept in NODE_CONCEPTS else modified.edge_attr[:, 0]
                            if alpha == 0:
                                for attr in ["x", "edge_attr", "edge_index"]:
                                    if not torch.equal(getattr(batch, attr), getattr(modified, attr)):
                                        raise AssertionError(f"Zero intervention changed {attr}")
                                logits = base
                            else:
                                _, _, logits = encode_batch(model, modified)
                            rec = records[concept, mode, r, i]
                            rec["y"].extend(y)
                            rec["pred"].extend(logits.argmax(1).cpu().tolist())
                            rec["logit"].extend((logits[row, batch.y] - base[row, batch.y]).cpu().tolist())
                            rec["prob"].extend((logits.softmax(1)[row, batch.y] - base_prob[row, batch.y]).cpu().tolist())
                            rec["flip"].extend((logits.argmax(1) != base.argmax(1)).cpu().tolist())
                            if mode == "random":
                                requested = torch.as_tensor(np.random.default_rng(seed).choice([-1., 1.], size=len(original)) * abs(alpha), dtype=original.dtype, device=device)
                            else:
                                requested = torch.full_like(original, float(alpha))
                            actual = actual_values - original
                            candidate = original + requested
                            rec["count"] += len(original)
                            rec["requested"] += requested.sum().item()
                            rec["actual"] += actual.sum().item()
                            rec["requested_abs"] += requested.abs().sum().item()
                            rec["actual_abs"] += actual.abs().sum().item()
                            rec["clipped"] += (actual_values != candidate).sum().item()
            if batch_idx % 25 == 0:
                print(f"  interventions batch {batch_idx + 1}/{len(loader)}", flush=True)
    final = {}
    for concept in concepts:
        out = {}
        for mode in ["concept", "random"]:
            arrays = {k: np.empty((1 if mode == "concept" else NUM_RANDOM_INTERVENTIONS, len(INTERVENTION_ALPHAS))) for k in keys}
            for r in range(len(arrays["accuracy"])):
                for i in range(len(INTERVENTION_ALPHAS)):
                    rec = records[concept, mode, r, i]
                    values = [accuracy_score(rec["y"], rec["pred"]),
                              f1_score(rec["y"], rec["pred"], average="macro", labels=list(range(7)), zero_division=0),
                              np.mean(rec["logit"]), np.mean(rec["prob"]), np.mean(rec["flip"]),
                              rec["requested"] / rec["count"], rec["actual"] / rec["count"],
                              rec["requested_abs"] / rec["count"], rec["actual_abs"] / rec["count"], rec["clipped"] / rec["count"]]
                    for k, v in zip(keys, values):
                        arrays[k][r, i] = v
            for k, arr in arrays.items():
                out[f"{mode}_{k}"] = arr.mean(axis=0)
                if mode == "random":
                    out[f"random_{k}_repeats"] = arr
                    out[f"random_{k}_std"] = arr.std(axis=0, ddof=1)
                    out[f"random_{k}_q025"] = np.quantile(arr, .025, axis=0)
                    out[f"random_{k}_q975"] = np.quantile(arr, .975, axis=0)
        out["concept_minus_random_true_logit_delta"] = out["concept_true_logit_delta"] - out["random_true_logit_delta"]
        final[concept] = out
    return final


# ---------------------------------------------------------------------
# Saving / reporting
# ---------------------------------------------------------------------

CAV_EVALUATION = {}


def save_results(
    cavs,
    tcav_results,
    intervention_results,
):
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    concepts = list(NODE_CONCEPTS.keys()) + [
        GRAPH_CONCEPT
    ]

    payload = {
        "concept_names": np.asarray(concepts),
        "intervention_alphas": INTERVENTION_ALPHAS,
    }

    for concept in concepts:
        safe = concept.replace(" ", "_")

        payload[f"{safe}__cav_direction"] = cavs[
            concept
        ]["direction"]

        payload[f"{safe}__cav_random_directions"] = cavs[
            concept
        ]["random_directions"]

        payload[f"{safe}__cav_probe_accuracy"] = np.asarray(
            cavs[concept]["probe_accuracy"]
        )
        payload[f"{safe}__cav_probe_macro_f1"] = np.asarray(
            cavs[concept]["probe_macro_f1"]
        )
        payload[f"{safe}__cav_probe_auc"] = np.asarray(
            cavs[concept]["probe_auc"]
        )

        payload[f"{safe}__tcav_validation"] = tcav_results[
            "validation"
        ][concept]["tcav"]
        payload[f"{safe}__tcav_test"] = tcav_results[
            "official_test"
        ][concept]["tcav"]

        payload[f"{safe}__tcav_random_validation"] = tcav_results[
            "validation"
        ][concept]["random_tcav"]
        payload[f"{safe}__tcav_random_test"] = tcav_results[
            "official_test"
        ][concept]["random_tcav"]

        payload[f"{safe}__tcav_random_mean_validation"] = tcav_results[
            "validation"
        ][concept]["random_tcav_mean"]
        payload[f"{safe}__tcav_random_mean_test"] = tcav_results[
            "official_test"
        ][concept]["random_tcav_mean"]

        payload[f"{safe}__tcav_minus_random_validation"] = tcav_results[
            "validation"
        ][concept]["tcav_minus_random"]
        payload[f"{safe}__tcav_minus_random_test"] = tcav_results[
            "official_test"
        ][concept]["tcav_minus_random"]

        for split_name, split_key in [
            ("validation", "validation"),
            ("official_test", "official_test"),
        ]:
            ir = intervention_results[split_name][concept]

            for key, value in ir.items():
                payload[
                    f"{safe}__{split_key}__{key}"
                ] = value

    payload["protocol_version"] = np.asarray("v2")
    payload["checkpoint_sha256"] = np.asarray(require_v2_preflight())
    payload["node_feature_indices"] = np.asarray(list(NODE_CONCEPTS.values()))
    for concept in concepts:
        payload[f"{concept}__q20"] = np.asarray(cavs[concept]["q20"])
        payload[f"{concept}__q80"] = np.asarray(cavs[concept]["q80"])
        for split_name, metrics in CAV_EVALUATION.items():
            for key, value in metrics[concept].items():
                payload[f"{concept}__cav_{split_name}_{key}"] = np.asarray(value)
        for split_name in ["validation", "official_test"]:
            null = tcav_results[split_name][concept]["random_tcav"]
            payload[f"{concept}__tcav_random_std_{split_name}"] = np.std(null, axis=1, ddof=1)
            payload[f"{concept}__tcav_random_q025_{split_name}"] = np.quantile(null, .025, axis=1)
            payload[f"{concept}__tcav_random_q975_{split_name}"] = np.quantile(null, .975, axis=1)
    np.savez(
        OUTPUT,
        **payload,
    )

    return OUTPUT


def print_cav_summary(cavs):
    print()
    print("==========================================")
    print("CAV SUMMARY")
    print("==========================================")

    for concept in list(NODE_CONCEPTS.keys()) + [GRAPH_CONCEPT]:
        c = cavs[concept]
        print(
            f"{concept:20s} "
            f"AUC={c['probe_auc']:.4f} "
            f"F1={c['probe_macro_f1']:.4f} "
            f"n={c['n']:,} "
            f"q20={c['q20']:.6g} "
            f"q80={c['q80']:.6g}"
        )


def print_tcav_summary(tcav_results):
    classes = [
        "ADH",
        "DCIS",
        "FEA",
        "IC",
        "N",
        "PB",
        "UDH",
    ]

    for split_name in ["validation", "official_test"]:
        print()
        print("==========================================")
        print(f"TCAV — {split_name}")
        print("==========================================")

        for concept in list(NODE_CONCEPTS.keys()) + [GRAPH_CONCEPT]:
            result = tcav_results[split_name][concept]

            print(f"\n{concept}")
            for k, class_name in enumerate(classes):
                if result["n"][k] == 0:
                    continue

                print(
                    f"  {class_name:4s}: "
                    f"TCAV={result['tcav'][k]:.4f} "
                    f"random={result['random_tcav_mean'][k]:.4f} "
                    f"delta={result['tcav_minus_random'][k]:+.4f} "
                    f"n={result['n'][k]}"
                )


def print_intervention_summary(intervention_results):
    for split_name in ["validation", "official_test"]:
        print()
        print("==========================================")
        print(f"CONTROLLED INTERVENTIONS — {split_name}")
        print("==========================================")

        for concept in list(NODE_CONCEPTS.keys()) + [GRAPH_CONCEPT]:
            r = intervention_results[split_name][concept]

            print(f"\n{concept}")
            for i, alpha in enumerate(INTERVENTION_ALPHAS):
                print(
                    f"  alpha={alpha:+.1f} "
                    f"F1={r['concept_macro_f1'][i]:.4f} "
                    f"random_F1={r['random_macro_f1'][i]:.4f} "
                    f"true_logit_delta="
                    f"{r['concept_true_logit_delta'][i]:+.5f} "
                    f"random_delta="
                    f"{r['random_true_logit_delta'][i]:+.5f} "
                    f"flip={r['concept_flip_rate'][i]:.4f}"
                )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global INTERVENTION_SCALER, CAV_EVALUATION
    require_v2_preflight()
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    warnings.filterwarnings(
        "ignore",
        message=".*torch.jit.script.*Python 3.14.*",
    )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("==========================================")
    print("GRAPH-BRACS CONCEPT AUDIT — PHASE 2")
    print("==========================================")
    print(f"Device: {device}")
    print(f"Seed:   {SEED}")

    data = load_context()
    INTERVENTION_SCALER = data

    model, model_config = load_frozen_model(device)

    print()
    print("Frozen GINE loaded.")
    print(
        f"Hidden dimension: {model_config['hidden_dim']}"
    )
    print(
        f"Layers:           {model_config['num_layers']}"
    )

    print()
    print("==========================================")
    print("FITTING CAVs FROM TRAIN")
    print("==========================================")

    node_H, node_Y, graph_Z, graph_Y = (
        collect_cav_training_data(
            model,
            data["train_loader"],
            data["x_mean"],
            data["x_std"],
            data["edge_mean"],
            data["edge_std"],
            device,
        )
    )

    cavs = fit_all_cavs(
        node_H,
        node_Y,
        graph_Z,
        graph_Y,
    )

    print_cav_summary(cavs)
    CAV_EVALUATION = {}
    for split_name, loader in [("validation", data["val_loader"]), ("official_test", data["test_loader"])]:
        print(f"Evaluating CAVs on {split_name} with fixed TRAIN thresholds")
        CAV_EVALUATION[split_name] = evaluate_cavs(model, loader, cavs, data, device)


    print()
    print("==========================================")
    print("TCAV")
    print("==========================================")

    tcav_results = {}

    for split_name, loader in [
        ("validation", data["val_loader"]),
        ("official_test", data["test_loader"]),
    ]:
        print()
        print(f"Running TCAV on {split_name}...")
        tcav_results[split_name] = tcav_on_split(
            model,
            loader,
            cavs,
            device,
        )

    print_tcav_summary(tcav_results)

    print()
    print("==========================================")
    print("CONTROLLED INPUT INTERVENTIONS")
    print("==========================================")

    concepts = list(NODE_CONCEPTS.keys()) + [
        GRAPH_CONCEPT
    ]

    intervention_results = {}

    for split_name, loader in [
        ("validation", data["val_loader"]),
        ("official_test", data["test_loader"]),
    ]:
        print()
        print(f"Running interventions on {split_name}...")
        intervention_results[split_name] = (
            evaluate_interventions(
                model,
                loader,
                concepts,
                device,
            )
        )

    print_intervention_summary(
        intervention_results
    )

    output = save_results(
        cavs,
        tcav_results,
        intervention_results,
    )

    print()
    print("==========================================")
    print("PHASE 2 COMPLETE")
    print("==========================================")
    print(f"Saved: {output}")
    print()
    print(
        "CAVs were fitted on TRAIN only. "
        "Validation and official test were used only for evaluation."
    )
    print(
        "The GINE was frozen throughout Phase 2; no model parameters were modified."
    )


if __name__ == "__main__":
    main()
