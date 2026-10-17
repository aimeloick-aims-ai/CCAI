import numpy as np
from .utils import require_v2_preflight

from .config import (
    GRAPH_CONCEPT,
    NODE_CONCEPTS,
    OUTPUT,
)


def save_results(
    node_probe,
    graph_probe,
    split_probe_results,
    erasure_results,
):
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_names = np.asarray(
        list(NODE_CONCEPTS.keys())
    )

    validation_node_r2 = (
        split_probe_results[
            "validation"
        ]["node"]["r2"]
    )

    test_node_r2 = (
        split_probe_results[
            "official_test"
        ]["node"]["r2"]
    )

    validation_graph_r2 = (
        split_probe_results[
            "validation"
        ]["graph"]["r2"][0]
    )

    test_graph_r2 = (
        split_probe_results[
            "official_test"
        ]["graph"]["r2"][0]
    )

    validation_baseline_f1 = (
        erasure_results[
            "validation"
        ]["classification"][
            "baseline"
        ]["macro_f1"]
    )

    test_baseline_f1 = (
        erasure_results[
            "official_test"
        ]["classification"][
            "baseline"
        ]["macro_f1"]
    )

    validation_node_erased_f1 = np.asarray(
        [
            erasure_results[
                "validation"
            ]["classification"][
                "node_erasure"
            ][concept]["macro_f1"]
            for concept in NODE_CONCEPTS
        ]
    )

    test_node_erased_f1 = np.asarray(
        [
            erasure_results[
                "official_test"
            ]["classification"][
                "node_erasure"
            ][concept]["macro_f1"]
            for concept in NODE_CONCEPTS
        ]
    )

    validation_node_residual_r2 = np.asarray(
        [
            erasure_results[
                "validation"
            ]["residual"][
                concept
            ]["r2"][0]
            for concept in NODE_CONCEPTS
        ]
    )

    test_node_residual_r2 = np.asarray(
        [
            erasure_results[
                "official_test"
            ]["residual"][
                concept
            ]["r2"][0]
            for concept in NODE_CONCEPTS
        ]
    )

    validation_graph_erased_f1 = (
        erasure_results[
            "validation"
        ]["classification"][
            "graph_erasure"
        ]["macro_f1"]
    )

    test_graph_erased_f1 = (
        erasure_results[
            "official_test"
        ]["classification"][
            "graph_erasure"
        ]["macro_f1"]
    )

    validation_graph_residual_r2 = (
        erasure_results[
            "validation"
        ]["residual"][
            GRAPH_CONCEPT
        ]["r2"][0]
    )

    test_graph_residual_r2 = (
        erasure_results[
            "official_test"
        ]["residual"][
            GRAPH_CONCEPT
        ]["r2"][0]
    )

    np.savez(
        OUTPUT,
        protocol_version=np.asarray("v2"),
        checkpoint_sha256=np.asarray(require_v2_preflight()),
        node_feature_indices=np.asarray(list(NODE_CONCEPTS.values())),
        concept_names=node_names,
        node_probe_coef_std=node_probe[
            "coef_std"
        ],
        node_probe_directions=node_probe[
            "directions"
        ],
        node_probe_mean_x=node_probe[
            "mean_x"
        ],
        node_probe_std_x=node_probe[
            "std_x"
        ],
        node_probe_mean_y=node_probe[
            "mean_y"
        ],
        node_probe_std_y=node_probe[
            "std_y"
        ],
        graph_probe_coef_std=graph_probe[
            "coef_std"
        ],
        graph_probe_directions=graph_probe[
            "directions"
        ],
        graph_probe_mean_x=graph_probe[
            "mean_x"
        ],
        graph_probe_std_x=graph_probe[
            "std_x"
        ],
        graph_probe_mean_y=graph_probe[
            "mean_y"
        ],
        graph_probe_std_y=graph_probe[
            "std_y"
        ],
        validation_node_r2=validation_node_r2,
        test_node_r2=test_node_r2,
        validation_graph_r2=np.asarray(
            validation_graph_r2
        ),
        test_graph_r2=np.asarray(
            test_graph_r2
        ),
        validation_baseline_f1=np.asarray(
            validation_baseline_f1
        ),
        test_baseline_f1=np.asarray(
            test_baseline_f1
        ),
        validation_node_erased_f1=(
            validation_node_erased_f1
        ),
        test_node_erased_f1=(
            test_node_erased_f1
        ),
        validation_node_residual_r2=(
            validation_node_residual_r2
        ),
        test_node_residual_r2=(
            test_node_residual_r2
        ),
        validation_graph_erased_f1=np.asarray(
            validation_graph_erased_f1
        ),
        test_graph_erased_f1=np.asarray(
            test_graph_erased_f1
        ),
        validation_graph_residual_r2=np.asarray(
            validation_graph_residual_r2
        ),
        test_graph_residual_r2=np.asarray(
            test_graph_residual_r2
        ),
    )

    return OUTPUT
