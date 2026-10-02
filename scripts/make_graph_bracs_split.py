from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
)


# ============================================================
# Configuration
# ============================================================

METADATA = Path("data/graph-bracs/metadata.csv")
OUTPUT_DIR = Path("processed/graph-bracs")

VAL_SIZE = 0.20
RANDOM_STATE = 42


# Explicit group columns that would indicate shared provenance.
# sample_id is deliberately excluded because it identifies
# individual graphs, not necessarily patients.
GROUP_CANDIDATES = [
    "patient_id",
    "patient",
    "case_id",
    "case",
    "subject_id",
    "subject",
    "specimen_id",
    "specimen",
    "slide_id",
    "slide",
    "wsi_id",
    "wsi",
]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    if not METADATA.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA}"
        )

    df = pd.read_csv(METADATA)

    print("==========================================")
    print("GRAPH-BRACS SPLIT")
    print("==========================================")

    print(f"Metadata rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = {
        "sample_id",
        "label",
        "split",
        "graph_path",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required metadata columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Check official split
    # --------------------------------------------------------

    print("\nOriginal split:")
    print(
        df["split"]
        .value_counts(dropna=False)
        .to_string()
    )

    official_test = df[
        df["split"].astype(str).str.lower() == "test"
    ].copy()

    official_train = df[
        df["split"].astype(str).str.lower() == "train"
    ].copy()

    if len(official_train) == 0:
        raise ValueError(
            "No rows with split='train' found."
        )

    if len(official_test) == 0:
        raise ValueError(
            "No rows with split='test' found."
        )

    print(
        f"\nOfficial train: {len(official_train):,}"
    )

    print(
        f"Official test:  {len(official_test):,}"
    )

    # --------------------------------------------------------
    # Check duplicate sample IDs
    # --------------------------------------------------------

    duplicated_ids = official_train[
        official_train["sample_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicated_ids) > 0:
        raise ValueError(
            "Duplicate sample_id values found in "
            "official training data."
        )

    # --------------------------------------------------------
    # Detect possible grouping variable
    # --------------------------------------------------------

    available_groups = [
        col
        for col in GROUP_CANDIDATES
        if col in df.columns
    ]

    print("\n==========================================")
    print("GROUPING CHECK")
    print("==========================================")

    if available_groups:
        print(
            "Potential grouping columns found:"
        )

        for col in available_groups:
            print(
                f"  {col}: "
                f"{official_train[col].nunique()} "
                f"unique groups"
            )

    else:
        print(
            "No explicit patient/case/slide grouping "
            "column found in metadata."
        )

    # --------------------------------------------------------
    # Select grouping column
    #
    # Prefer the first explicit patient/case-level column.
    # Slide-level grouping is used only if no patient/case
    # identifier is available.
    # --------------------------------------------------------

    preferred_order = [
        "patient_id",
        "patient",
        "case_id",
        "case",
        "subject_id",
        "subject",
        "specimen_id",
        "specimen",
        "slide_id",
        "slide",
        "wsi_id",
        "wsi",
    ]

    group_column = None

    for candidate in preferred_order:
        if candidate in official_train.columns:
            group_column = candidate
            break

    # --------------------------------------------------------
    # Create train / validation split
    # --------------------------------------------------------

    if group_column is not None:

        print(
            f"\nUsing grouped split: {group_column}"
        )

        groups = official_train[
            group_column
        ].astype(str)

        y = official_train["label"].astype(str)

        # Five folds gives approximately 20% validation.
        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        train_idx, val_idx = next(
            sgkf.split(
                official_train,
                y,
                groups,
            )
        )

        train_df = official_train.iloc[
            train_idx
        ].copy()

        val_df = official_train.iloc[
            val_idx
        ].copy()

        # Safety check: no group overlap
        train_groups = set(
            train_df[group_column].astype(str)
        )

        val_groups = set(
            val_df[group_column].astype(str)
        )

        overlap = train_groups & val_groups

        if overlap:
            raise RuntimeError(
                "Group leakage detected between "
                "train and validation."
            )

    else:

        print(
            "\nNo explicit grouping variable available."
        )

        print(
            "Using stratified graph-level split."
        )

        print(
            "WARNING: patient-level leakage cannot be "
            "assessed from the available metadata."
        )

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
        )

        train_idx, val_idx = next(
            splitter.split(
                official_train,
                official_train["label"],
            )
        )

        train_df = official_train.iloc[
            train_idx
        ].copy()

        val_df = official_train.iloc[
            val_idx
        ].copy()

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    train_ids = set(
        train_df["sample_id"]
    )

    val_ids = set(
        val_df["sample_id"]
    )

    test_ids = set(
        official_test["sample_id"]
    )

    if train_ids & val_ids:
        raise RuntimeError(
            "Train/validation sample overlap detected."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test sample overlap detected."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Validation/test sample overlap detected."
        )

    # --------------------------------------------------------
    # Save IDs
    # --------------------------------------------------------

    train_out = OUTPUT_DIR / "train_ids.csv"
    val_out = OUTPUT_DIR / "val_ids.csv"

    train_df[
        [
            "sample_id",
            "label",
            "graph_path",
        ]
    ].to_csv(
        train_out,
        index=False,
    )

    val_df[
        [
            "sample_id",
            "label",
            "graph_path",
        ]
    ].to_csv(
        val_out,
        index=False,
    )

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    summary_rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("official_test", official_test),
    ]:

        counts = (
            split_df["label"]
            .value_counts()
            .sort_index()
        )

        for label, count in counts.items():

            summary_rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "fraction": (
                        count / len(split_df)
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_out = (
        OUTPUT_DIR /
        "split_summary.csv"
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print("\n==========================================")
    print("FINAL SPLIT")
    print("==========================================")

    print(
        f"Train:      {len(train_df):,}"
    )

    print(
        f"Validation: {len(val_df):,}"
    )

    print(
        f"Test:       {len(official_test):,}"
    )

    print(
        f"Train fraction of official train: "
        f"{len(train_df) / len(official_train):.3f}"
    )

    print(
        f"Validation fraction of official train: "
        f"{len(val_df) / len(official_train):.3f}"
    )

    print("\nClass distribution:")

    distribution = (
    summary
    .pivot(
        index="label",
        columns="split",
        values="count",
    )
    .fillna(0)
    .astype(int)
    )

    distribution = distribution.reindex(
        columns=["train", "validation", "official_test"],
        fill_value=0,
    )

    print(
        distribution.to_string()
    )

    print("\n==========================================")
    print("FILES SAVED")
    print("==========================================")

    print(train_out)
    print(val_out)
    print(summary_out)

    print("\nOfficial test remains untouched.")


if __name__ == "__main__":
    main()