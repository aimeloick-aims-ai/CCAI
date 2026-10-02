import numpy as np
import torch

from .config import (
    MAX_NODE_SAMPLES_PER_GRAPH,
    NODE_CONCEPTS,
)
from .ridge import (
    create_prediction_stats,
    create_ridge_stats,
    finalize_prediction_stats,
    predict_probe,
    update_prediction_stats,
    update_ridge_stats,
)
from .utils import (
    encode_batch,
    graph_mean_edge_distance,
    sample_nodes,
)


def collect_training_statistics(
    model,
    loader,
    x_mean,
    x_std,
    edge_mean,
    edge_std,
    device,
):
    node_dim = (
        model.classifier[0].in_features // 2
    )

    graph_dim = node_dim * 2

    node_stats = create_ridge_stats(
        node_dim,
        len(NODE_CONCEPTS),
    )

    graph_stats = create_ridge_stats(
        graph_dim,
        1,
    )

    concept_names = list(
        NODE_CONCEPTS.keys()
    )

    processed_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            h, z, _ = encode_batch(
                model,
                batch,
            )

            selected = sample_nodes(
                batch,
                MAX_NODE_SAMPLES_PER_GRAPH,
            )

            H = (
                h[selected]
                .detach()
                .cpu()
                .numpy()
            )

            Y = []

            for concept in concept_names:
                feature_idx = NODE_CONCEPTS[
                    concept
                ]

                values = (
                    batch.x[
                        selected,
                        feature_idx,
                    ]
                    * x_std[feature_idx]
                    + x_mean[feature_idx]
                )

                Y.append(
                    values.detach()
                    .cpu()
                    .numpy()
                )

            Y = np.column_stack(Y)

            update_ridge_stats(
                node_stats,
                H,
                Y,
            )

            graph_target = (
                graph_mean_edge_distance(
                    batch,
                    edge_mean,
                    edge_std,
                )
            )

            Z = (
                z.detach()
                .cpu()
                .numpy()
            )

            GY = (
                graph_target.detach()
                .cpu()
                .numpy()
                .reshape(-1, 1)
            )

            update_ridge_stats(
                graph_stats,
                Z,
                GY,
            )

            processed_graphs += batch.num_graphs

            if (
                processed_graphs % 250
                < batch.num_graphs
                or processed_graphs
                == len(loader.dataset)
            ):
                print(
                    f"  processed "
                    f"{processed_graphs:,}/"
                    f"{len(loader.dataset):,}"
                )

    return node_stats, graph_stats


def evaluate_probes(
    model,
    loader,
    node_probe,
    graph_probe,
    x_mean,
    x_std,
    edge_mean,
    edge_std,
    device,
):
    node_prediction_stats = (
        create_prediction_stats(
            len(NODE_CONCEPTS)
        )
    )

    graph_prediction_stats = (
        create_prediction_stats(1)
    )

    concept_names = list(
        NODE_CONCEPTS.keys()
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            h, z, _ = encode_batch(
                model,
                batch,
            )

            selected = sample_nodes(
                batch,
                MAX_NODE_SAMPLES_PER_GRAPH,
            )

            H = (
                h[selected]
                .detach()
                .cpu()
                .numpy()
            )

            Y = []

            for concept in concept_names:
                feature_idx = NODE_CONCEPTS[
                    concept
                ]

                values = (
                    batch.x[
                        selected,
                        feature_idx,
                    ]
                    * x_std[feature_idx]
                    + x_mean[feature_idx]
                )

                Y.append(
                    values.detach()
                    .cpu()
                    .numpy()
                )

            Y = np.column_stack(Y)

            Y_pred = predict_probe(
                H,
                node_probe,
            )

            update_prediction_stats(
                node_prediction_stats,
                Y,
                Y_pred,
            )

            graph_target = (
                graph_mean_edge_distance(
                    batch,
                    edge_mean,
                    edge_std,
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 1)
            )

            Z = (
                z.detach()
                .cpu()
                .numpy()
            )

            Z_pred = predict_probe(
                Z,
                graph_probe,
            )

            update_prediction_stats(
                graph_prediction_stats,
                graph_target,
                Z_pred,
            )

    return (
        finalize_prediction_stats(
            node_prediction_stats
        ),
        finalize_prediction_stats(
            graph_prediction_stats
        ),
    )
