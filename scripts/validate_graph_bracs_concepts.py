from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import kruskal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


INPUT_FILE = Path(
    "processed/graph-bracs/concept_candidates_train.csv"
)

OUTPUT_DIR = Path("processed/graph-bracs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Five primary concept candidates
# ============================================================

CONCEPTS = {
    "cell_size": "cell_size_area",
    "cell_shape": "cell_circularity",
    "cell_intensity": "cell_mean_intensity_gray",
    "cell_texture": "cell_texture_contrast",
    "spatial_proximity": "mean_edge_distance",
}


def main():

    df = pd.read_csv(INPUT_FILE)

    print("==========================================")
    print("GRAPH-BRACS CONCEPT VALIDATION")
    print("==========================================")
    print(f"Graphs: {len(df):,}")
    print(f"Classes: {df['label'].nunique()}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = ["sample_id", "label"] + list(
        CONCEPTS.values()
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # 1. Global concept statistics
    # --------------------------------------------------------

    summary_rows = []

    for concept_name, column in CONCEPTS.items():

        values = df[column].dropna()

        q20 = values.quantile(0.20)
        q80 = values.quantile(0.80)

        summary_rows.append(
            {
                "concept": concept_name,
                "source_feature": column,
                "n": len(values),
                "min": values.min(),
                "q20": q20,
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "q80": q80,
                "max": values.max(),
                "mean": values.mean(),
                "std": values.std(),
                "n_unique": values.nunique(),
                "low_count": int((values <= q20).sum()),
                "high_count": int((values >= q80).sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "concept_validation_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Concept statistics by target class
    # --------------------------------------------------------

    class_rows = []

    for concept_name, column in CONCEPTS.items():

        for label, group in df.groupby("label"):

            values = group[column].dropna()

            class_rows.append(
                {
                    "concept": concept_name,
                    "label": label,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "q25": values.quantile(0.25),
                    "median": values.median(),
                    "q75": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                }
            )

    class_summary = pd.DataFrame(class_rows)

    class_summary.to_csv(
        OUTPUT_DIR / "concept_class_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Low / high concept groups by target class
    # --------------------------------------------------------

    low_high_rows = []

    for concept_name, column in CONCEPTS.items():

        values = df[column]

        q20 = values.quantile(0.20)
        q80 = values.quantile(0.80)

        for label, group in df.groupby("label"):

            concept_values = group[column]

            n = concept_values.notna().sum()

            low = (
                concept_values.le(q20)
                .fillna(False)
                .sum()
            )

            high = (
                concept_values.ge(q80)
                .fillna(False)
                .sum()
            )

            middle = n - low - high

            low_high_rows.append(
                {
                    "concept": concept_name,
                    "label": label,
                    "n": int(n),
                    "low": int(low),
                    "middle": int(middle),
                    "high": int(high),
                    "low_fraction": (
                        low / n if n > 0 else np.nan
                    ),
                    "high_fraction": (
                        high / n if n > 0 else np.nan
                    ),
                }
            )

    low_high = pd.DataFrame(low_high_rows)

    low_high.to_csv(
        OUTPUT_DIR / "concept_low_high_by_class.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Correlation between selected concepts
    # --------------------------------------------------------

    selected_columns = list(
        CONCEPTS.values()
    )

    correlations = df[
        selected_columns
    ].corr(method="spearman")

    correlations.index = CONCEPTS.keys()
    correlations.columns = CONCEPTS.keys()

    correlations.to_csv(
        OUTPUT_DIR / "concept_validation_correlations.csv"
    )

    # --------------------------------------------------------
    # 5. Kruskal-Wallis test across the 7 target classes
    # --------------------------------------------------------

    kruskal_rows = []

    for concept_name, column in CONCEPTS.items():

        groups = []

        for _, group in df.groupby("label"):

            values = group[column].dropna().to_numpy()

            if len(values) > 0:
                groups.append(values)

        if SCIPY_AVAILABLE and len(groups) >= 2:

            statistic, pvalue = kruskal(*groups)

        else:

            statistic = np.nan
            pvalue = np.nan

        kruskal_rows.append(
            {
                "concept": concept_name,
                "kruskal_H": statistic,
                "kruskal_p": pvalue,
            }
        )

    kruskal_df = pd.DataFrame(
        kruskal_rows
    )

    kruskal_df.to_csv(
        OUTPUT_DIR / "concept_class_tests.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Human-readable report
    # --------------------------------------------------------

    report_file = (
        OUTPUT_DIR /
        "concept_validation_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "GRAPH-BRACS CONCEPT VALIDATION REPORT\n"
        )
        f.write(
            "=====================================\n\n"
        )

        f.write(
            f"Number of train graphs: {len(df):,}\n"
        )

        f.write(
            f"Number of target classes: "
            f"{df['label'].nunique()}\n\n"
        )

        f.write(
            "TARGET CLASS COUNTS\n"
        )
        f.write(
            "-------------------\n"
        )

        f.write(
            df["label"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        f.write("\n\n")

        f.write(
            "CONCEPT SUMMARY\n"
        )
        f.write(
            "---------------\n"
        )

        f.write(
            summary[
                [
                    "concept",
                    "median",
                    "q20",
                    "q80",
                    "std",
                    "low_count",
                    "high_count",
                ]
            ].to_string(index=False)
        )

        f.write("\n\n")

        f.write(
            "SPEARMAN CORRELATION\n"
        )
        f.write(
            "--------------------\n"
        )

        f.write(
            correlations.to_string()
        )

        f.write("\n\n")

        f.write(
            "KRUSKAL-WALLIS TESTS\n"
        )
        f.write(
            "--------------------\n"
        )

        f.write(
            kruskal_df.to_string(index=False)
        )

        f.write("\n")

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n==========================================")
    print("GLOBAL CONCEPT SUMMARY")
    print("==========================================")

    print(
        summary[
            [
                "concept",
                "median",
                "q20",
                "q80",
                "std",
                "low_count",
                "high_count",
            ]
        ].to_string(index=False)
    )

    print("\n==========================================")
    print("SPEARMAN CORRELATIONS")
    print("==========================================")

    print(correlations.to_string())

    print("\n==========================================")
    print("KRUSKAL-WALLIS TESTS")
    print("==========================================")

    print(
        kruskal_df.to_string(index=False)
    )

    print("\n==========================================")
    print("FILES SAVED")
    print("==========================================")

    print(
        "concept_validation_summary.csv"
    )
    print(
        "concept_class_summary.csv"
    )
    print(
        "concept_low_high_by_class.csv"
    )
    print(
        "concept_validation_correlations.csv"
    )
    print(
        "concept_class_tests.csv"
    )
    print(
        "concept_validation_report.txt"
    )


if __name__ == "__main__":
    main()