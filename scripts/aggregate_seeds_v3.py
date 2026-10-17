"""Aggregate v3 audit outputs over several trained checkpoints (seeds).

A finding counts only if it replicates across independently trained models: report, for each
concept and dose, in how many seeds the effect lies beyond the random-feature null.

Usage:
    python scripts/aggregate_seeds_v3.py results/audit_v3_grouped_seed* --out results/audit_v3_summary
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--split", default="official_test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen_hashes, seeds = set(), set()
    for directory in map(Path, a.dirs):
        manifest = json.loads((directory / "manifest.json").read_text())
        digest = manifest["sha256"]
        if manifest.get("seed") is None:
            sys.exit("Missing training seed in manifest.")
        if digest in seen_hashes or manifest["seed"] in seeds:
            sys.exit("Repeated checkpoint/seed: reruns are not independent training seeds.")
        if any(manifest.get("leakage", {}).values()):
            sys.exit("Slide overlap in run: refusing inferential aggregation.")
        if manifest.get("descriptive_only") or not manifest.get("training_ids_verified", False):
            sys.exit("Unverified checkpoint training provenance: refusing inferential aggregation.")
        for name in (f"interventions_{a.split}.csv", "decodability_baselines.csv"):
            if not (directory / name).exists():
                sys.exit(f"Incomplete run: {directory / name}")
        seen_hashes.add(digest)
        seeds.add(manifest["seed"])
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    iv, dec = [], []
    for d in map(Path, a.dirs):
        tag = d.name
        f = d / f"interventions_{a.split}.csv"
        if f.exists():
            iv.append(pd.read_csv(f).assign(run=tag))
        g = d / "decodability_baselines.csv"
        if g.exists():
            dec.append(pd.read_csv(g).query("split == @a.split").assign(run=tag))
    if iv:
        iv = pd.concat(iv)
        agg = (
            iv.groupby(["concept", "alpha"])
            .agg(n_runs=("run", "nunique"),
                 isolated_dlogit_mean=("isolated_dlogit", "mean"), isolated_dlogit_sd=("isolated_dlogit", "std"),
                 isolated_beyond_null=("isolated_beyond_null", "sum"),
                 coherent_dlogit_mean=("coherent_dlogit", "mean"), coherent_beyond_null=("coherent_beyond_null", "sum"),
                 isolated_dF1_mean=("isolated_dF1", "mean"), isolated_dF1_sd=("isolated_dF1", "std"),
                 coherent_dF1_mean=("coherent_dF1", "mean"))
            .reset_index()
        )
        agg.to_csv(out / "interventions_over_seeds.csv", index=False)
        print(agg.round(3).to_string(index=False))
    if dec:
        dec = pd.concat(dec)
        agg = (dec.groupby("concept")
               .agg(n_runs=("run", "nunique"), r2_trained=("r2_trained", "mean"),
                    r2_untrained=("r2_untrained_mean", "mean"), r2_raw_other=("r2_raw_other_features", "mean"),
                    excess_mean=("excess_over_untrained", "mean"), excess_sd=("excess_over_untrained", "std"))
               .reset_index())
        agg.to_csv(out / "decodability_over_seeds.csv", index=False)
        print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
