from pathlib import Path

import numpy as np
import pandas as pd
import torch


TRAIN_IDS = Path(
    "processed/graph-bracs/train_ids.csv"
)

OUTPUT = Path(
    "processed/graph-bracs/scaler.npz"
)


def resolve_graph_path(graph_path):
    path = Path(graph_path)

    if path.exists():
        return path

    alternative = Path("data") / path

    if alternative.exists():
        return alternative

    raise FileNotFoundError(
        f"Graph not found. Tried:\n"
        f"  {path}\n"
        f"  {alternative}"
    )


def main():

    df = pd.read_csv(TRAIN_IDS)

    print("==========================================")
    print("GRAPH-BRACS TRAIN-ONLY SCALER")
    print("==========================================")
    print(f"Training graphs: {len(df):,}")

    x_sum = None
    x_sum_sq = None
    x_count = 0

    edge_sum = 0.0
    edge_sum_sq = 0.0
    edge_count = 0

    feature_dim = None

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):

        # ---------------------------------------------
        # Resolve graph path
        # ---------------------------------------------

        path = resolve_graph_path(row.graph_path)

        # ---------------------------------------------
        # Load graph
        # ---------------------------------------------

        graph = torch.load(
            path,
            weights_only=False,
        )

        x = (
            graph.x
            .detach()
            .cpu()
            .numpy()
        )

        edge_attr = (
            graph.edge_attr
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        # ---------------------------------------------
        # Check node features
        # ---------------------------------------------

        if x.ndim != 2:
            raise ValueError(
                f"Invalid x shape in {path}: {x.shape}"
            )

        if feature_dim is None:

            feature_dim = x.shape[1]

            x_sum = np.zeros(
                feature_dim,
                dtype=np.float64,
            )

            x_sum_sq = np.zeros(
                feature_dim,
                dtype=np.float64,
            )

        elif x.shape[1] != feature_dim:

            raise ValueError(
                f"Feature dimension mismatch in {path}: "
                f"{x.shape[1]} != {feature_dim}"
            )

        # ---------------------------------------------
        # Accumulate node statistics
        # ---------------------------------------------

        x64 = x.astype(
            np.float64,
            copy=False,
        )

        x_sum += x64.sum(axis=0)

        x_sum_sq += (
            np.square(x64).sum(axis=0)
        )

        x_count += x64.shape[0]

        # ---------------------------------------------
        # Accumulate edge statistics
        # ---------------------------------------------

        if len(edge_attr) > 0:

            edge64 = edge_attr.astype(
                np.float64,
                copy=False,
            )

            edge_sum += edge64.sum()

            edge_sum_sq += np.square(
                edge64
            ).sum()

            edge_count += len(edge64)

        if i % 250 == 0 or i == len(df):

            print(
                f"Processed {i:,}/{len(df):,}"
            )

    # ---------------------------------------------
    # Compute node statistics
    # ---------------------------------------------

    x_mean = x_sum / x_count

    x_var = (
        x_sum_sq / x_count
        - np.square(x_mean)
    )

    x_var = np.maximum(
        x_var,
        0.0,
    )

    x_std = np.sqrt(x_var)

    constant_features = x_std < 1e-8

    x_std[constant_features] = 1.0

    # ---------------------------------------------
    # Compute edge statistics
    # ---------------------------------------------

    if edge_count > 0:

        edge_mean = (
            edge_sum / edge_count
        )

        edge_var = (
            edge_sum_sq / edge_count
            - edge_mean**2
        )

        edge_var = max(
            edge_var,
            0.0,
        )

        edge_std = np.sqrt(edge_var)

    else:

        edge_mean = 0.0
        edge_std = 1.0

    if edge_std < 1e-8:
        edge_std = 1.0

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT,
        x_mean=x_mean,
        x_std=x_std,
        edge_mean=np.array(
            edge_mean,
            dtype=np.float64,
        ),
        edge_std=np.array(
            edge_std,
            dtype=np.float64,
        ),
        feature_dim=np.array(
            feature_dim,
            dtype=np.int64,
        ),
        node_count=np.array(
            x_count,
            dtype=np.int64,
        ),
        edge_count=np.array(
            edge_count,
            dtype=np.int64,
        ),
    )

    # ---------------------------------------------
    # Report
    # ---------------------------------------------

    print("\n==========================================")
    print("SCALER SUMMARY")
    print("==========================================")

    print(
        f"Node feature dimension: {feature_dim}"
    )

    print(
        f"Total nodes used: {x_count:,}"
    )

    print(
        f"Total edges used: {edge_count:,}"
    )

    print(
        f"Node features with std < 1e-8: "
        f"{constant_features.sum()}"
    )

    print(
        f"Edge mean: {edge_mean:.6f}"
    )

    print(
        f"Edge std:  {edge_std:.6f}"
    )

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()