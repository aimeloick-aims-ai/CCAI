"""WSI-grouped audit split for Graph-BRACS (replaces the ROI-level stratified split).

BRACS sample ids look like ``BRACS_<WSI>_<CLASS>_<ROI>``. ROIs cut from the same
whole-slide image (WSI) are highly correlated, so a stratified split at ROI level
leaks slide identity between TRAIN and VALIDATION. This script keeps every WSI
entirely on one side, stratifying by class as well as possible.

Outputs (never overwrites existing files):
    processed/graph-bracs/train_ids_grouped.csv
    processed/graph-bracs/val_ids_grouped.csv
    processed/graph-bracs/split_summary_grouped.csv

Usage:
    python scripts/make_graph_bracs_split_grouped.py [--seed 42] [--val-fraction 0.2]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROCESSED = Path("processed/graph-bracs")
METADATA_CSV = Path("data/graph-bracs/metadata.csv")
ID_PATTERN = re.compile(r"^BRACS_(\d+)_([A-Za-z]+)_(\d+)$")


def parse_wsi(sample_ids):
    parsed = sample_ids.str.extract(ID_PATTERN)
    if parsed[0].isna().any():
        bad = sample_ids[parsed[0].isna()].head(5).tolist()
        raise ValueError(f"Cannot parse WSI id from sample_id, e.g. {bad}")
    return parsed[0]


def load_official_train():
    """Official TRAIN partition = old audit TRAIN + old audit VALIDATION."""
    if METADATA_CSV.exists():
        meta = pd.read_csv(METADATA_CSV)
        return meta[meta["split"] == "train"][["sample_id", "label", "graph_path"]], meta
    old = [pd.read_csv(PROCESSED / n) for n in ("train_ids.csv", "val_ids.csv")]
    return pd.concat(old, ignore_index=True), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    out_train = PROCESSED / "train_ids_grouped.csv"
    out_val = PROCESSED / "val_ids_grouped.csv"
    out_sum = PROCESSED / "split_summary_grouped.csv"
    for path in (out_train, out_val, out_sum):
        if path.exists():
            sys.exit(f"Refusing to overwrite {path}")

    df, meta = load_official_train()
    df = df.reset_index(drop=True)
    df["wsi"] = parse_wsi(df["sample_id"])
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in official TRAIN")

    n_splits = int(round(1.0 / args.val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, df["label"], groups=df["wsi"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    # Hard guarantees
    assert not set(train_df["wsi"]) & set(val_df["wsi"]), "WSI leakage between TRAIN and VALIDATION"
    assert not set(train_df["sample_id"]) & set(val_df["sample_id"])
    assert len(train_df) + len(val_df) == len(df)

    # Official test must also be slide-disjoint from official TRAIN (check when metadata exists)
    if meta is not None:
        test_wsi = set(parse_wsi(meta.loc[meta["split"] == "test", "sample_id"]))
        shared = test_wsi & set(df["wsi"])
        print(f"Official test WSIs also present in official train: {len(shared)}")
        if shared:
            raise ValueError("Official train/test share slides. Reconcile source metadata before generating splits.")

    counts = (
        pd.concat([train_df.assign(split="train"), val_df.assign(split="validation")])
        .groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int)
    )
    print(counts)
    print(f"Train {len(train_df)} ({train_df['wsi'].nunique()} WSIs) | "
          f"Val {len(val_df)} ({val_df['wsi'].nunique()} WSIs) | WSI overlap: 0")

    cols = ["sample_id", "label", "graph_path"]
    train_df[cols].to_csv(out_train, index=False)
    val_df[cols].to_csv(out_val, index=False)
    counts.to_csv(out_sum)


if __name__ == "__main__":
    main()
