import numpy as np
import torch

from .config import (
    GRAPH_CONCEPT,
    NODE_CONCEPTS,
    NODE_CONCEPT_FEATURE_NAMES,
    OUTPUT,
    RIDGE_ALPHA,
    SEED,
)
from .controls import make_random_directions
from .data import (
    load_context,
    load_frozen_model,
)
from .erasure import (
    collect_residual_probe_statistics,
    evaluate_erasure,
)
from .probes import (
    collect_training_statistics,
    evaluate_probes,
)
from .results import save_results
from .ridge import fit_ridge_from_stats
from .utils import set_seed, require_v2_preflight


def main():
    require_v2_preflight()
    torch.set_num_threads(4)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("==========================================")
    print("GRAPH-BRACS CONCEPT AUDIT â€” PHASE 1")
    print("==========================================")
    print(f"Device: {device}")
    print(f"Seed:   {SEED}")

    print()
    print("==========================================")
    print("CONCEPT TARGETS")
    print("==========================================")

    for concept, idx in NODE_CONCEPTS.items():
        print(
            f"{concept:20s} -> "
            f"node feature {idx}: "
            f"{NODE_CONCEPT_FEATURE_NAMES[concept]}"
        )

    print(
        f"{GRAPH_CONCEPT:20s} -> "
        f"graph mean(edge distance)"
    )

    data = load_context()

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    x_mean = data["x_mean"]
    x_std = data["x_std"]
    edge_mean = data["edge_mean"]
    edge_std = data["edge_std"]

    print()
    print(
        f"Train:         "
        f"{len(data['train_df']):,}"
    )
    print(
        f"Validation:    "
        f"{len(data['val_df']):,}"
    )
    print(
        f"Official test: "
        f"{len(data['test_df']):,}"
    )

    model, model_config = load_frozen_model(
        device
    )

    print()
    print("Frozen GINE loaded.")
    print(
        f"Hidden dimension: "
        f"{model_config['hidden_dim']}"
    )
    print(
        f"Layers:           "
        f"{model_config['num_layers']}"
    )

    print()
    print("==========================================")
    print("FITTING CONCEPT PROBES")
    print("==========================================")

    node_stats, graph_stats = (
        collect_training_statistics(
            model,
            train_loader,
            x_mean,
            x_std,
            edge_mean,
            edge_std,
            device,
        )
    )

    node_probe = fit_ridge_from_stats(
        node_stats,
        RIDGE_ALPHA,
    )

    graph_probe = fit_ridge_from_stats(
        graph_stats,
        RIDGE_ALPHA,
    )

    print()
    print(
        f"Node probe samples:  "
        f"{node_probe['n']:,}"
    )
    print(
        f"Graph probe samples: "
        f"{graph_probe['n']:,}"
    )

    print()
    print("==========================================")
    print("CONCEPT DECODABILITY")
    print("==========================================")

    split_probe_results = {}

    for split_name, loader in [
        ("validation", val_loader),
        ("official_test", test_loader),
    ]:
        node_results, graph_results = (
            evaluate_probes(
                model,
                loader,
                node_probe,
                graph_probe,
                x_mean,
                x_std,
                edge_mean,
                edge_std,
                device,
            )
        )

        split_probe_results[
            split_name
        ] = {
            "node": node_results,
            "graph": graph_results,
        }

        print()
        print(f"--- {split_name} ---")

        for j, concept in enumerate(
            NODE_CONCEPTS
        ):
            print(
                f"{concept:20s} "
                f"R2={node_results['r2'][j]:.4f} "
                f"MAE={node_results['mae'][j]:.6f}"
            )

        print(
            f"{GRAPH_CONCEPT:20s} "
            f"R2={graph_results['r2'][0]:.4f} "
            f"MAE={graph_results['mae'][0]:.6f}"
        )

    print()
    print("==========================================")
    print("FITTING RESIDUAL PROBES AFTER ERASURE")
    print("==========================================")

    (
        node_residual_probes,
        graph_residual_probe,
    ) = collect_residual_probe_statistics(
        model,
        train_loader,
        node_probe,
        graph_probe,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
        device,
    )

    print("Residual probes fitted.")

    node_dim = len(
        node_probe["mean_x"]
    )

    graph_dim = len(
        graph_probe["mean_x"]
    )

    random_directions = (
        make_random_directions(
            node_dim,
            graph_dim,
            SEED + 1000,
        )
    )

    print()
    print("==========================================")
    print("REPRESENTATIONAL UTILIZATION")
    print("==========================================")

    erasure_results = {}

    for split_name, loader in [
        ("validation", val_loader),
        ("official_test", test_loader),
    ]:
        print()
        print(f"--- {split_name} ---")

        classification, residual = (
            evaluate_erasure(
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
            )
        )

        erasure_results[
            split_name
        ] = {
            "classification": classification,
            "residual": residual,
        }

        baseline_f1 = (
            classification[
                "baseline"
            ]["macro_f1"]
        )

        print(
            f"Baseline Macro-F1 = "
            f"{baseline_f1:.4f}"
        )

        for concept in NODE_CONCEPTS:
            erased_f1 = (
                classification[
                    "node_erasure"
                ][concept]["macro_f1"]
            )

            drop = (
                baseline_f1
                - erased_f1
            )

            random_f1 = np.mean(
                [
                    r["macro_f1"]
                    for r in classification[
                        "random"
                    ][concept]
                ]
            )

            random_drop = (
                baseline_f1
                - random_f1
            )

            residual_r2 = (
                residual[
                    concept
                ]["r2"][0]
            )

            print(
                f"{concept:20s} "
                f"erased F1={erased_f1:.4f} "
                f"drop={drop:+.4f} "
                f"random_drop={random_drop:+.4f} "
                f"residual_R2={residual_r2:.4f}"
            )

        erased_f1 = (
            classification[
                "graph_erasure"
            ]["macro_f1"]
        )

        drop = (
            baseline_f1
            - erased_f1
        )

        random_f1 = np.mean(
            [
                r["macro_f1"]
                for r in classification[
                    "random"
                ][GRAPH_CONCEPT]
            ]
        )

        random_drop = (
            baseline_f1
            - random_f1
        )

        residual_r2 = (
            residual[
                GRAPH_CONCEPT
            ]["r2"][0]
        )

        print(
            f"{GRAPH_CONCEPT:20s} "
            f"erased F1={erased_f1:.4f} "
            f"drop={drop:+.4f} "
            f"random_drop={random_drop:+.4f} "
            f"residual_R2={residual_r2:.4f}"
        )

    output = save_results(
        node_probe,
        graph_probe,
        split_probe_results,
        erasure_results,
    )

    print()
    print("==========================================")
    print("PHASE 1 COMPLETE")
    print("==========================================")
    print(f"Saved: {output}")
    print()
    print("No input graph was modified.")
    print(
        "Official test was not used for "
        "probe fitting or GINE training; "
        "it was used only for final evaluation."
    )


if __name__ == "__main__":
    main()
