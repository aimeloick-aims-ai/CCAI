from pathlib import Path

import numpy as np
import pandas as pd
import torch


METADATA_FILE = Path("data/graph-bracs/metadata.csv")
GRAPH_DIR = Path("data/graph-bracs/data")
OUTPUT_DIR = Path("processed/graph-bracs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Exact feature indices from Graph-BRACS
# ============================================================

FEATURES = {
    "mean_intensity_R": 7,
    "mean_intensity_G": 8,
    "mean_intensity_B": 9,

    "axis_major_length": 20,
    "axis_minor_length": 21,
    "eccentricity": 22,
    "area": 23,
    "circularity": 27,
    "elongation": 28,

    "mean_contrast": 55,
    "mean_homogeneity": 59,

    "mean_intensity_gray_scale": 90,
}


FOURIER_INDICES = list(range(29, 54))


def graph_concepts(graph):

    x = graph.x.detach().cpu().numpy().astype(np.float64)

    n_nodes = x.shape[0]
    n_edges = graph.edge_index.shape[1]

    concepts = {}

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    concepts["cell_size_area"] = np.mean(
        x[:, FEATURES["area"]]
    )

    concepts["cell_major_axis"] = np.mean(
        x[:, FEATURES["axis_major_length"]]
    )

    concepts["cell_minor_axis"] = np.mean(
        x[:, FEATURES["axis_minor_length"]]
    )

    concepts["cell_eccentricity"] = np.mean(
        x[:, FEATURES["eccentricity"]]
    )

    concepts["cell_circularity"] = np.mean(
        x[:, FEATURES["circularity"]]
    )

    concepts["cell_elongation"] = np.mean(
        x[:, FEATURES["elongation"]]
    )

    # --------------------------------------------------------
    # Intensity / colour
    # --------------------------------------------------------

    concepts["cell_mean_intensity_R"] = np.mean(
        x[:, FEATURES["mean_intensity_R"]]
    )

    concepts["cell_mean_intensity_G"] = np.mean(
        x[:, FEATURES["mean_intensity_G"]]
    )

    concepts["cell_mean_intensity_B"] = np.mean(
        x[:, FEATURES["mean_intensity_B"]]
    )

    concepts["cell_mean_intensity_gray"] = np.mean(
        x[:, FEATURES["mean_intensity_gray_scale"]]
    )

    # --------------------------------------------------------
    # Texture
    # --------------------------------------------------------

    concepts["cell_texture_contrast"] = np.mean(
        x[:, FEATURES["mean_contrast"]]
    )

    concepts["cell_texture_homogeneity"] = np.mean(
        x[:, FEATURES["mean_homogeneity"]]
    )

    # --------------------------------------------------------
    # Fourier shape descriptor
    # --------------------------------------------------------

    fourier = x[:, FOURIER_INDICES]

    concepts["fourier_shape_energy"] = np.mean(
        np.sqrt(np.sum(fourier ** 2, axis=1))
    )

    # --------------------------------------------------------
    # Graph / spatial descriptors
    # --------------------------------------------------------

    if graph.edge_attr is not None and graph.edge_attr.numel() > 0:

        distances = (
            graph.edge_attr
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )

        concepts["mean_edge_distance"] = np.mean(
            distances
        )

        concepts["median_edge_distance"] = np.median(
            distances
        )

    else:

        concepts["mean_edge_distance"] = np.nan
        concepts["median_edge_distance"] = np.nan

    concepts["edge_count_per_node"] = (
        n_edges / n_nodes
    )

    concepts["n_nodes"] = n_nodes
    concepts["n_edges"] = n_edges

    return concepts


def main():

    metadata = pd.read_csv(METADATA_FILE)

    train = metadata[
        metadata["split"] == "train"
    ].copy()

    print("======================================")
    print("BUILDING CONCEPT CANDIDATES")
    print("======================================")
    print(f"Train graphs: {len(train):,}")

    records = []

    for i, row in enumerate(
        train.itertuples(index=False)
    ):

        graph_path = (
            GRAPH_DIR /
            Path(row.graph_path).name
        )

        graph = torch.load(
            graph_path,
            weights_only=False,
            map_location="cpu",
        )

        concepts = graph_concepts(graph)

        record = {
            "sample_id": graph.sample_id,
            "label": graph.label,
        }

        record.update(concepts)

        records.append(record)

        if i % 500 == 0:
            print(
                f"{i:,} / {len(train):,} graphs processed"
            )

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save raw graph-level concept values
    # --------------------------------------------------------

    concept_file = (
        OUTPUT_DIR /
        "concept_candidates_train.csv"
    )

    df.to_csv(
        concept_file,
        index=False,
    )

    # --------------------------------------------------------
    # Quantile thresholds
    # --------------------------------------------------------

    concept_columns = [
        c
        for c in df.columns
        if c not in [
            "sample_id",
            "label",
        ]
        and not c.startswith("n_")
    ]

    summary = []

    for concept in concept_columns:

        values = df[concept].dropna()

        q20 = values.quantile(0.20)
        q25 = values.quantile(0.25)
        q50 = values.quantile(0.50)
        q75 = values.quantile(0.75)
        q80 = values.quantile(0.80)

        summary.append(
            {
                "concept": concept,
                "min": values.min(),
                "q20": q20,
                "q25": q25,
                "median": q50,
                "q75": q75,
                "q80": q80,
                "max": values.max(),
                "std": values.std(),
                "n": len(values),
            }
        )

    summary_df = pd.DataFrame(summary)

    summary_file = (
        OUTPUT_DIR /
        "concept_candidates_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Positive / negative candidate sets
    # --------------------------------------------------------

    concept_sets = []

    for concept in concept_columns:

        values = df[concept]

        q20 = values.quantile(0.20)
        q80 = values.quantile(0.80)

        for _, row in df.iterrows():

            value = row[concept]

            if pd.isna(value):
                continue

            if value <= q20:

                concept_sets.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "concept": concept,
                        "concept_value": value,
                        "concept_label": 0,
                    }
                )

            elif value >= q80:

                concept_sets.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "concept": concept,
                        "concept_value": value,
                        "concept_label": 1,
                    }
                )

    concept_sets_df = pd.DataFrame(
        concept_sets
    )

    concept_sets_file = (
        OUTPUT_DIR /
        "concept_sets_train.csv"
    )

    concept_sets_df.to_csv(
        concept_sets_file,
        index=False,
    )

    # --------------------------------------------------------
    # Correlations between candidate concepts
    # --------------------------------------------------------

    correlation_df = df[
        concept_columns
    ].corr(method="spearman")

    correlation_file = (
        OUTPUT_DIR /
        "concept_candidate_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_file
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n======================================")
    print("CONCEPT CANDIDATES")
    print("======================================")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n======================================")
    print("FILES")
    print("======================================")

    print(
        f"Saved: {concept_file}"
    )

    print(
        f"Saved: {summary_file}"
    )

    print(
        f"Saved: {concept_sets_file}"
    )

    print(
        f"Saved: {correlation_file}"
    )


if __name__ == "__main__":
    main()