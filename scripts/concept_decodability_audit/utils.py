import random
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_batch(model, batch):
    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr

    x = model.input_projection(x)

    for conv, norm in zip(model.convs, model.norms):
        residual = x

        x = conv(x, edge_index, edge_attr)
        x = norm(x)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(
            x,
            p=model.dropout,
            training=False,
        )
        x = x + residual

    mean_pool = global_mean_pool(x, batch.batch)
    max_pool = global_max_pool(x, batch.batch)

    graph_embedding = torch.cat(
        [mean_pool, max_pool],
        dim=1,
    )

    logits = model.classifier(graph_embedding)

    return x, graph_embedding, logits


def sample_nodes(batch, max_per_graph):
    selected = []

    for graph_id in range(batch.num_graphs):
        indices = torch.where(batch.batch == graph_id)[0]
        n = len(indices)

        if n <= max_per_graph:
            selected.append(indices)
        else:
            positions = torch.linspace(
                0,
                n - 1,
                steps=max_per_graph,
                device=indices.device,
            ).long()

            selected.append(indices[positions])

    if not selected:
        return torch.empty(
            0,
            dtype=torch.long,
            device=batch.x.device,
        )

    return torch.cat(selected)


def graph_mean_edge_distance(batch, edge_mean, edge_std):
    edge_values = (
        batch.edge_attr[:, 0] * edge_std
        + edge_mean
    )

    edge_graph_ids = batch.batch[
        batch.edge_index[0]
    ]

    sums = torch.zeros(
        batch.num_graphs,
        device=batch.x.device,
        dtype=edge_values.dtype,
    )

    counts = torch.zeros(
        batch.num_graphs,
        device=batch.x.device,
        dtype=edge_values.dtype,
    )

    sums.scatter_add_(
        0,
        edge_graph_ids,
        edge_values,
    )

    counts.scatter_add_(
        0,
        edge_graph_ids,
        torch.ones_like(edge_values),
    )

    counts = torch.clamp(
        counts,
        min=1.0,
    )

    return sums / counts


def erase_direction(X, center, direction):
    centered = X - center

    projection = (
        centered @ direction
    ).unsqueeze(1)

    return (
        centered
        - projection * direction.unsqueeze(0)
        + center
    )


def require_v2_preflight():
    from .config import CHECKPOINT
    report = json.loads(Path("results/audit_v2_preflight.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != report["checkpoint_sha256"]:
        raise RuntimeError("GINE checkpoint differs from validated checkpoint")
    if report["identity_features_edges_logits_predictions"] != "passed":
        raise RuntimeError("Audit preflight did not pass")
    return digest
