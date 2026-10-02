from pathlib import Path

import numpy as np
import pandas as pd
import torch


METADATA_FILE = Path("data/graph-bracs/metadata.csv")
GRAPH_DIR = Path("data/graph-bracs/data")
OUTPUT_DIR = Path("processed/graph-bracs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_NAMES = [
    "probability",

    # RGB intensity
    "min_intensity_R",
    "min_intensity_G",
    "min_intensity_B",
    "max_intensity_R",
    "max_intensity_G",
    "max_intensity_B",
    "mean_intensity_R",
    "mean_intensity_G",
    "mean_intensity_B",
    "std_intensity_R",
    "std_intensity_G",
    "std_intensity_B",
    "skew_intensity_R",
    "skew_intensity_G",
    "skew_intensity_B",
    "kurtosis_intensity_R",
    "kurtosis_intensity_G",
    "kurtosis_intensity_B",

    # Morphology
    "orientation",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "area",
    "perimeter",
    "solidity",
    "extent",
    "circularity",
    "elongation",

    # Fourier shape descriptors
    *[f"FD_20_{i}" for i in range(10)],
    *[f"FD_30_{i}" for i in range(15)],

    # Texture
    *[
        f"{stat}_{prop}"
        for stat in [
            "mean",
            "std",
            "skew",
            "kurtosis",
            "min",
            "max",
        ]
        for prop in [
            "ASM",
            "contrast",
            "correlation",
            "dissimilarity",
            "energy",
            "homogeneity",
        ]
    ],

    # Grayscale intensity
    "mean_intensity_gray_scale",
    "std_intensity_gray_scale",
    "skew_intensity_gray_scale",
    "kurtosis_intensity_gray_scale",
    "min_intensity_gray_scale",
    "max_intensity_gray_scale",
]


def main():

    metadata = pd.read_csv(METADATA_FILE)

    train = metadata[metadata["split"] == "train"].copy()

    print("======================================")
    print("GRAPH-BRACS FEATURE PROFILE")
    print("======================================")
    print(f"Train graphs : {len(train):,}")
    print(f"Features     : {len(FEATURE_NAMES)}")

    all_features = []
    node_counts = []
    edge_counts = []
    edge_distances = []

    for i, row in enumerate(train.itertuples(index=False)):

        graph_path = GRAPH_DIR / Path(row.graph_path).name

        graph = torch.load(
            graph_path,
            weights_only=False,
            map_location="cpu",
        )

        x = graph.x.detach().cpu().numpy()

        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"{graph_path}: expected "
                f"{len(FEATURE_NAMES)} features, "
                f"got {x.shape[1]}"
            )

        if not np.isfinite(x).all():
            raise ValueError(
                f"Non-finite node feature found in {graph_path}"
            )

        all_features.append(x)

        node_counts.append(x.shape[0])
        edge_counts.append(graph.edge_index.shape[1])

        if graph.edge_attr is not None:
            d = graph.edge_attr.detach().cpu().numpy().reshape(-1)
            edge_distances.append(d)

        if i % 500 == 0:
            print(f"{i:,} / {len(train):,} graphs processed")


    X = np.concatenate(all_features, axis=0)

    # -----------------------------
    # Feature statistics
    # -----------------------------

    rows = []

    for j, name in enumerate(FEATURE_NAMES):

        values = X[:, j]

        rows.append(
            {
                "feature_index": j,
                "feature": name,
                "min": np.min(values),
                "q01": np.percentile(values, 1),
                "q25": np.percentile(values, 25),
                "median": np.median(values),
                "mean": np.mean(values),
                "q75": np.percentile(values, 75),
                "q99": np.percentile(values, 99),
                "max": np.max(values),
                "std": np.std(values),
            }
        )

    feature_stats = pd.DataFrame(rows)

    feature_stats.to_csv(
        OUTPUT_DIR / "feature_stats_train.csv",
        index=False,
    )

    # -----------------------------
    # Graph size statistics
    # -----------------------------

    graph_stats = pd.DataFrame(
        {
            "n_nodes": node_counts,
            "n_edges": edge_counts,
        }
    )

    graph_stats.to_csv(
        OUTPUT_DIR / "graph_size_stats_train.csv",
        index=False,
    )

    # -----------------------------
    # Edge distance statistics
    # -----------------------------

    if edge_distances:

        distances = np.concatenate(edge_distances)

        edge_stats = pd.DataFrame(
            [
                {
                    "n_edges": len(distances),
                    "min": np.min(distances),
                    "q01": np.percentile(distances, 1),
                    "q25": np.percentile(distances, 25),
                    "median": np.median(distances),
                    "mean": np.mean(distances),
                    "q75": np.percentile(distances, 75),
                    "q99": np.percentile(distances, 99),
                    "max": np.max(distances),
                    "std": np.std(distances),
                }
            ]
        )

        edge_stats.to_csv(
            OUTPUT_DIR / "edge_distance_stats_train.csv",
            index=False,
        )

    print("\n======================================")
    print("DONE")
    print("======================================")

    print(
        f"Total nodes analyzed : {len(X):,}"
    )

    print(
        f"Saved : "
        f"{OUTPUT_DIR / 'feature_stats_train.csv'}"
    )

    print(
        f"Saved : "
        f"{OUTPUT_DIR / 'graph_size_stats_train.csv'}"
    )

    print(
        f"Saved : "
        f"{OUTPUT_DIR / 'edge_distance_stats_train.csv'}"
    )


if __name__ == "__main__":
    main()