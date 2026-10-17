"""Decodability baselines: is a high probe R2 informative, or expected by construction?

The concepts are INPUT features and the GINE keeps a residual path from a linear
projection of the input, so a linear probe on the final node embedding can recover them
even from an untrained network. These baselines quantify that:

  * untrained GINE  : same architecture, random weights (BN statistics calibrated on TRAIN)
  * raw redundancy  : probe from the OTHER 95 input features (target column removed) for node
                      concepts, and from mean+max pooled raw node features for the graph concept

Excess decodability = R2(trained) - R2(untrained). Only this excess says something about learning.
"""
import numpy as np
import torch
from torch_geometric.nn import global_max_pool, global_mean_pool

from concept_decodability_audit.config import (
    GRAPH_CONCEPT,
    MAX_NODE_SAMPLES_PER_GRAPH,
    NODE_CONCEPTS,
    RIDGE_ALPHA,
)
from concept_decodability_audit.ridge import (
    create_prediction_stats,
    create_ridge_stats,
    finalize_prediction_stats,
    fit_ridge_from_stats,
    predict_probe,
    update_prediction_stats,
    update_ridge_stats,
)
from concept_decodability_audit.utils import (
    encode_batch,
    graph_mean_edge_distance,
    sample_nodes,
)
from train_gine_baseline import GINEBaseline

CONCEPTS = list(NODE_CONCEPTS) + [GRAPH_CONCEPT]


def make_untrained_model(model_config, seed, device, calibration_loader=None, calibration_batches=30):
    """Random-init GINE with the trained architecture. BN running stats are set from TRAIN data
    (no weight update) so that the comparison with the trained model is not an artefact of BN."""
    torch.manual_seed(int(seed))
    model = GINEBaseline(**model_config).to(device)
    if calibration_loader is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.reset_running_stats()
                module.momentum = None  # cumulative average
        model.train()
        saved_dropout, model.dropout = model.dropout, 0.0
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= calibration_batches:
                    break
                batch = batch.to(device)
                model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        model.dropout = saved_dropout
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---- feature providers: batch -> (node_getter(concept) -> [N, d], graph_features [B, d_g]) -------

def trained_provider(model):
    def provider(batch):
        h, z, _ = encode_batch(model, batch)
        return (lambda concept: h), z

    return provider


def raw_provider():
    keep = {c: [j for j in range(96) if j != idx] for c, idx in NODE_CONCEPTS.items()}

    def provider(batch):
        z = torch.cat([global_mean_pool(batch.x, batch.batch), global_max_pool(batch.x, batch.batch)], dim=1)
        return (lambda concept: batch.x[:, keep[concept]]), z

    return provider


def _targets(batch, selected, x_mean, x_std, edge_mean, edge_std):
    cols = []
    for concept, idx in NODE_CONCEPTS.items():
        vals = batch.x[selected, idx] * float(x_std[idx]) + float(x_mean[idx])
        cols.append(vals.detach().cpu().numpy())
    node_y = np.column_stack(cols)
    graph_y = graph_mean_edge_distance(batch, edge_mean, edge_std).detach().cpu().numpy().reshape(-1, 1)
    return node_y, graph_y


def probe_suite(provider, train_loader, eval_loaders, scales, device, ridge_alpha=RIDGE_ALPHA):
    """Fit ridge probes on TRAIN, return {split: {concept: R2}}. ``scales`` = x_mean,x_std,edge_mean,edge_std."""
    x_mean, x_std, edge_mean, edge_std = scales
    names = list(NODE_CONCEPTS)
    node_stats, graph_stats = {}, None
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            getter, z = provider(batch)
            sel = sample_nodes(batch, MAX_NODE_SAMPLES_PER_GRAPH)
            node_y, graph_y = _targets(batch, sel, x_mean, x_std, edge_mean, edge_std)
            for k, c in enumerate(names):
                H = getter(c)[sel].detach().cpu().numpy()
                if c not in node_stats:
                    node_stats[c] = create_ridge_stats(H.shape[1], 1)
                update_ridge_stats(node_stats[c], H, node_y[:, k])
            Z = z.detach().cpu().numpy()
            if graph_stats is None:
                graph_stats = create_ridge_stats(Z.shape[1], 1)
            update_ridge_stats(graph_stats, Z, graph_y)
    node_probes = {c: fit_ridge_from_stats(node_stats[c], ridge_alpha) for c in names}
    graph_probe = fit_ridge_from_stats(graph_stats, ridge_alpha)

    results = {}
    for split, loader in eval_loaders.items():
        pstats = {c: create_prediction_stats(1) for c in CONCEPTS}
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                getter, z = provider(batch)
                sel = sample_nodes(batch, MAX_NODE_SAMPLES_PER_GRAPH)
                node_y, graph_y = _targets(batch, sel, x_mean, x_std, edge_mean, edge_std)
                for k, c in enumerate(names):
                    H = getter(c)[sel].detach().cpu().numpy()
                    update_prediction_stats(pstats[c], node_y[:, k], predict_probe(H, node_probes[c]))
                update_prediction_stats(
                    pstats[GRAPH_CONCEPT], graph_y, predict_probe(z.detach().cpu().numpy(), graph_probe)
                )
        results[split] = {c: float(np.ravel(finalize_prediction_stats(pstats[c])["r2"])[0]) for c in CONCEPTS}
    return results


def run_baselines(model, model_config, train_loader, eval_loaders, scales, device, n_untrained=5, seed=0):
    """Trained vs untrained vs raw-redundancy R2 for every concept and split."""
    out = {"trained": probe_suite(trained_provider(model), train_loader, eval_loaders, scales, device)}
    untrained = []
    for s in range(n_untrained):
        m = make_untrained_model(model_config, seed + s, device, calibration_loader=train_loader)
        untrained.append(probe_suite(trained_provider(m), train_loader, eval_loaders, scales, device))
    out["untrained"] = untrained
    out["raw_redundancy"] = probe_suite(raw_provider(), train_loader, eval_loaders, scales, device)
    return out


def summarize(baselines):
    """Rows: split, concept, trained, untrained mean/sd, raw, excess (trained - untrained mean)."""
    rows = []
    for split in baselines["trained"]:
        for c in CONCEPTS:
            un = np.array([u[split][c] for u in baselines["untrained"]])
            tr = baselines["trained"][split][c]
            rows.append(
                dict(
                    split=split,
                    concept=c,
                    r2_trained=tr,
                    r2_untrained_mean=float(un.mean()),
                    r2_untrained_sd=float(un.std(ddof=1)) if len(un) > 1 else float("nan"),
                    r2_raw_other_features=baselines["raw_redundancy"][split][c],
                    excess_over_untrained=tr - float(un.mean()),
                )
            )
    return rows
