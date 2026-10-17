import numpy as np
import torch
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
)

from .config import (
    GRAPH_CONCEPT,
    MAX_NODE_SAMPLES_PER_GRAPH,
    NODE_CONCEPTS,
    NUM_RANDOM_CONTROLS,
    RIDGE_ALPHA,
)
from .metrics import (
    confusion_metrics,
    create_confusion,
    update_confusion,
)
from .ridge import (
    create_prediction_stats,
    create_ridge_stats,
    finalize_prediction_stats,
    fit_ridge_from_stats,
    predict_probe,
    update_prediction_stats,
    update_ridge_stats,
)
from .utils import (
    encode_batch,
    erase_direction,
    graph_mean_edge_distance,
    sample_nodes,
)


def collect_residual_probe_statistics(
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
    concept_names = list(
        NODE_CONCEPTS.keys()
    )

    node_dim = len(
        node_probe["mean_x"]
    )

    graph_dim = len(
        graph_probe["mean_x"]
    )

    node_residual_stats = {
        concept: create_ridge_stats(
            node_dim,
            1,
        )
        for concept in concept_names
    }

    graph_residual_stats = (
        create_ridge_stats(
            graph_dim,
            1,
        )
    )

    node_center = torch.tensor(
        node_probe["mean_x"],
        dtype=torch.float32,
        device=device,
    )

    node_directions = torch.tensor(
        node_probe["directions"],
        dtype=torch.float32,
        device=device,
    )

    graph_center = torch.tensor(
        graph_probe["mean_x"],
        dtype=torch.float32,
        device=device,
    )

    graph_direction = torch.tensor(
        graph_probe["directions"][0],
        dtype=torch.float32,
        device=device,
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

            H = h[selected]

            Y_all = []

            for concept in concept_names:
                feature_idx = NODE_CONCEPTS[
                    concept
                ]

                target = (
                    batch.x[
                        selected,
                        feature_idx,
                    ]
                    * x_std[feature_idx]
                    + x_mean[feature_idx]
                )

                Y_all.append(
                    target.detach()
                    .cpu()
                    .numpy()
                )

            for j, concept in enumerate(
                concept_names
            ):
                H_erased = erase_direction(
                    H,
                    node_center,
                    node_directions[j],
                )

                H_np = (
                    H_erased.detach()
                    .cpu()
                    .numpy()
                )

                Y_np = (
                    Y_all[j]
                    .reshape(-1, 1)
                )

                update_ridge_stats(
                    node_residual_stats[
                        concept
                    ],
                    H_np,
                    Y_np,
                )

            Z_erased = erase_direction(
                z,
                graph_center,
                graph_direction,
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

            update_ridge_stats(
                graph_residual_stats,
                Z_erased.detach()
                .cpu()
                .numpy(),
                graph_target,
            )

    node_residual_probes = {}

    for concept in concept_names:
        node_residual_probes[
            concept
        ] = fit_ridge_from_stats(
            node_residual_stats[
                concept
            ],
            RIDGE_ALPHA,
        )

    graph_residual_probe = (
        fit_ridge_from_stats(
            graph_residual_stats,
            RIDGE_ALPHA,
        )
    )

    return (
        node_residual_probes,
        graph_residual_probe,
    )


def evaluate_erasure(
    model,
    loader,
    node_probe,
    graph_probe,
    node_residual_probes,
    graph_residual_probe,
    x_mean,
    x_std,
    edge_mean,
    edge_std,
    device,
    random_directions,
):
    concept_names = list(
        NODE_CONCEPTS.keys()
    )

    baseline_confusion = create_confusion()

    node_erasure_confusions = {
        concept: create_confusion()
        for concept in concept_names
    }

    graph_erasure_confusion = (
        create_confusion()
    )

    random_confusions = {
        concept: [
            create_confusion()
            for _ in range(
                NUM_RANDOM_CONTROLS
            )
        ]
        for concept in (
            concept_names
            + [GRAPH_CONCEPT]
        )
    }

    node_residual_prediction_stats = {
        concept: create_prediction_stats(1)
        for concept in concept_names
    }

    graph_residual_prediction_stats = (
        create_prediction_stats(1)
    )

    node_center = torch.tensor(
        node_probe["mean_x"],
        dtype=torch.float32,
        device=device,
    )

    node_directions = torch.tensor(
        node_probe["directions"],
        dtype=torch.float32,
        device=device,
    )

    graph_center = torch.tensor(
        graph_probe["mean_x"],
        dtype=torch.float32,
        device=device,
    )

    graph_direction = torch.tensor(
        graph_probe["directions"][0],
        dtype=torch.float32,
        device=device,
    )

    random_node_directions = {
        concept: [
            torch.tensor(
                d,
                dtype=torch.float32,
                device=device,
            )
            for d in directions
        ]
        for concept, directions
        in random_directions["node"].items()
    }

    random_graph_directions = [
        torch.tensor(
            d,
            dtype=torch.float32,
            device=device,
        )
        for d in random_directions["graph"]
    ]

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            h, z, logits = encode_batch(
                model,
                batch,
            )

            update_confusion(
                baseline_confusion,
                batch.y,
                logits.argmax(dim=1),
            )

            for j, concept in enumerate(
                concept_names
            ):
                h_erased = erase_direction(
                    h,
                    node_center,
                    node_directions[j],
                )

                mean_pool = global_mean_pool(
                    h_erased,
                    batch.batch,
                )

                max_pool = global_max_pool(
                    h_erased,
                    batch.batch,
                )

                z_erased = torch.cat(
                    [mean_pool, max_pool],
                    dim=1,
                )

                erased_logits = model.classifier(
                    z_erased
                )

                update_confusion(
                    node_erasure_confusions[
                        concept
                    ],
                    batch.y,
                    erased_logits.argmax(dim=1),
                )

                for r, direction in enumerate(
                    random_node_directions[
                        concept
                    ]
                ):
                    h_random = erase_direction(
                        h,
                        node_center,
                        direction,
                    )

                    mean_random = global_mean_pool(
                        h_random,
                        batch.batch,
                    )

                    max_random = global_max_pool(
                        h_random,
                        batch.batch,
                    )

                    z_random = torch.cat(
                        [
                            mean_random,
                            max_random,
                        ],
                        dim=1,
                    )

                    random_logits = model.classifier(
                        z_random
                    )

                    update_confusion(
                        random_confusions[
                            concept
                        ][r],
                        batch.y,
                        random_logits.argmax(
                            dim=1
                        ),
                    )

                selected = sample_nodes(
                    batch,
                    MAX_NODE_SAMPLES_PER_GRAPH,
                )

                H_erased_selected = (
                    h_erased[selected]
                )

                H_np = (
                    H_erased_selected.detach()
                    .cpu()
                    .numpy()
                )

                feature_idx = NODE_CONCEPTS[
                    concept
                ]

                target = (
                    batch.x[
                        selected,
                        feature_idx,
                    ]
                    * x_std[feature_idx]
                    + x_mean[feature_idx]
                )

                target_np = (
                    target.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 1)
                )

                prediction = predict_probe(
                    H_np,
                    node_residual_probes[
                        concept
                    ],
                )

                update_prediction_stats(
                    node_residual_prediction_stats[
                        concept
                    ],
                    target_np,
                    prediction,
                )

            z_graph_erased = erase_direction(
                z,
                graph_center,
                graph_direction,
            )

            graph_erased_logits = model.classifier(
                z_graph_erased
            )

            update_confusion(
                graph_erasure_confusion,
                batch.y,
                graph_erased_logits.argmax(
                    dim=1
                ),
            )

            for r, direction in enumerate(
                random_graph_directions
            ):
                z_random = erase_direction(
                    z,
                    graph_center,
                    direction,
                )

                random_logits = model.classifier(
                    z_random
                )

                update_confusion(
                    random_confusions[
                        GRAPH_CONCEPT
                    ][r],
                    batch.y,
                    random_logits.argmax(
                        dim=1
                    ),
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

            graph_prediction = predict_probe(
                z_graph_erased.detach()
                .cpu()
                .numpy(),
                graph_residual_probe,
            )

            update_prediction_stats(
                graph_residual_prediction_stats,
                graph_target,
                graph_prediction,
            )

    results = {}

    results["baseline"] = confusion_metrics(
        baseline_confusion
    )

    results["node_erasure"] = {}
    results["random"] = {}

    for concept in concept_names:
        results["node_erasure"][
            concept
        ] = confusion_metrics(
            node_erasure_confusions[
                concept
            ]
        )

        results["random"][
            concept
        ] = [
            confusion_metrics(c)
            for c in random_confusions[
                concept
            ]
        ]

    results["graph_erasure"] = (
        confusion_metrics(
            graph_erasure_confusion
        )
    )

    results["random"][
        GRAPH_CONCEPT
    ] = [
        confusion_metrics(c)
        for c in random_confusions[
            GRAPH_CONCEPT
        ]
    ]

    residual_results = {}

    for concept in concept_names:
        residual_results[
            concept
        ] = finalize_prediction_stats(
            node_residual_prediction_stats[
                concept
            ]
        )

    residual_results[
        GRAPH_CONCEPT
    ] = finalize_prediction_stats(
        graph_residual_prediction_stats
    )

    return results, residual_results
