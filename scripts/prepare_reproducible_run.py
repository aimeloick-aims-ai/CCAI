"""Create a frozen internal grouped split and inspect graph files before training.

Never changes the dataset-supplied split. All outputs use a new experiment folder.
Optional mapping CSV: sample_id,patient_id (one complete row per graph).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from concept_audit_v3.integrity import validate_splits
from concept_audit_v3.stats import wsi_ids

LABELS = ["ADH", "DCIS", "FEA", "IC", "N", "PB", "UDH"]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_partitions(metadata, seed=42, patient_mapping=None, source_partitions=None):
    required = {"sample_id", "label", "graph_path", "split"}
    if not required.issubset(metadata.columns) or metadata.sample_id.duplicated().any():
        raise ValueError("Metadata must contain unique sample_id,label,graph_path,split")
    if metadata[list(required)].isna().any().any() or set(metadata.label) != set(LABELS):
        raise ValueError("Missing metadata or unexpected/missing classes")
    frame = metadata.rename(columns={"split": "source_split"}).copy()
    frame["slide_id"] = wsi_ids(frame.sample_id)
    frame["group_id"] = frame.slide_id
    if patient_mapping is not None:
        mapping = patient_mapping[["sample_id", "patient_id"]].copy()
        if mapping.sample_id.duplicated().any() or mapping.patient_id.isna().any():
            raise ValueError("Patient mapping contains duplicate IDs or missing patients")
        frame = frame.merge(mapping, on="sample_id", how="left", validate="one_to_one")
        if frame.patient_id.isna().any():
            raise ValueError("Patient mapping does not cover every graph")
        if (frame.groupby("slide_id").patient_id.nunique() != 1).any():
            raise ValueError("One slide is assigned to multiple patients")
        frame["group_id"] = frame.patient_id.astype(str)
    if source_partitions is not None:
        if patient_mapping is None:
            raise ValueError("Source partition restoration requires verified patient mapping")
        assignment = source_partitions[["sample_id", "partition"]]
        if assignment.sample_id.duplicated().any():
            raise ValueError("Duplicate IDs in source partitions")
        frame = frame.merge(assignment, on="sample_id", how="left", validate="one_to_one")
        if frame.partition.isna().any() or set(frame.partition) != {"train", "validation", "test"}:
            raise ValueError("Source partitions must completely cover all graph IDs")
        frames = {s: frame.loc[frame.partition == s].copy() for s in ("train", "validation", "test")}
    else:
        # Fixed first fold only: never search across folds/seeds using model scores.
        pool, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            frame, frame.label, frame.group_id))
        remaining = frame.iloc[pool]
        train, val = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1).split(
            remaining, remaining.label, remaining.group_id))
        frames = {"train": remaining.iloc[train].copy(), "validation": remaining.iloc[val].copy(),
                  "test": frame.iloc[test].copy()}
    validate_splits(frames)
    for name, part in frames.items():
        if set(part.label) != set(LABELS):
            raise ValueError(f"Class missing from {name}; review grouping feasibility, not test scores")
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(frames[a].group_id) & set(frames[b].group_id):
                raise ValueError(f"Group overlap: {a}/{b}")
    return frames


def inspect_graphs(metadata, limit=0):
    import torch

    rows, errors = [], []
    for i, row in enumerate(metadata.itertuples(index=False)):
        if limit and i >= limit:
            break
        p = Path(row.graph_path)
        if not p.exists():
            p = Path("data") / p
        try:
            g = torch.load(p, map_location="cpu", weights_only=False)
            if g.x.ndim != 2 or g.x.shape[1] != 96 or not len(g.x):
                raise ValueError("Expected nonempty node features [N,96]")
            if g.edge_index.ndim != 2 or g.edge_index.shape[0] != 2 or g.edge_index.shape[1] == 0:
                raise ValueError("Expected nonempty edge_index [2,E]")
            if g.edge_index.dtype != torch.long:
                raise ValueError("edge_index must be int64")
            if g.edge_attr.shape != (g.edge_index.shape[1], 1):
                raise ValueError("edge_attr shape must be [E,1]")
            if not torch.isfinite(g.x).all() or not torch.isfinite(g.edge_attr).all():
                raise ValueError("Nonfinite descriptor values")
            if g.edge_index.min() < 0 or g.edge_index.max() >= len(g.x):
                raise ValueError("Edge index outside node range")
            if g.edge_attr.min() < 0:
                raise ValueError("Negative raw distance")
            if getattr(g, "sample_id", row.sample_id) != row.sample_id:
                raise ValueError("Graph sample ID disagrees with metadata")
            if getattr(g, "label", row.label) != row.label:
                raise ValueError("Graph label disagrees with metadata")
            rows.append({"sample_id": row.sample_id, "nodes": len(g.x), "edges": g.edge_index.shape[1],
                         "distance_min": float(g.edge_attr.min()), "distance_max": float(g.edge_attr.max())})
        except Exception as exc:
            errors.append({"sample_id": row.sample_id, "error": str(exc)})
        if (i + 1) % 250 == 0:
            print(f"Inspected {i+1}/{len(metadata)} graphs", flush=True)
    return rows, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default="data/graph-bracs/metadata.csv")
    ap.add_argument("--out-dir", default="results/reproducible_run")
    ap.add_argument("--patient-mapping", help="CSV with sample_id,patient_id")
    ap.add_argument("--source-partitions", help="CSV with sample_id,partition from original BRACS metadata")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-graphs", type=int, default=0, help="0 inspects every graph; positive values are preflight only")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite {out}")
    meta = pd.read_csv(args.metadata)
    mapping = pd.read_csv(args.patient_mapping, dtype=str) if args.patient_mapping else None
    source = pd.read_csv(args.source_partitions) if args.source_partitions else None
    frames = make_partitions(meta, args.seed, mapping, source)
    out.mkdir(parents=True, exist_ok=True)
    rows, errors = inspect_graphs(meta, args.max_graphs)
    pd.DataFrame(rows).to_csv(out / "graph_inventory.csv", index=False)
    pd.DataFrame(errors, columns=["sample_id", "error"]).to_csv(out / "graph_errors.csv", index=False)
    manifest = {"protocol": "restored_bracs_partitions_available_graphs" if source is not None else "internal_grouped_exploratory", "split_seed": args.seed if source is None else None,
                "grouping": "patient" if mapping is not None else "slide",
                "patient_independence_verified": mapping is not None,
                "source_metadata_sha256": fingerprint(args.metadata),
                "source_metadata_path": args.metadata,
                "patient_mapping_sha256": fingerprint(args.patient_mapping) if args.patient_mapping else None,
                "source_partitions_sha256": fingerprint(args.source_partitions) if args.source_partitions else None,
                "graph_files_checked": len(rows) + len(errors), "graph_errors": len(errors),
                "full_graph_check": len(rows) + len(errors) == len(meta),
                "ready_for_training": not errors and len(rows) == len(meta),
                "note": "Source BRACS assignments restored on available graphs; prior exploratory exposure means not an untouched external test." if source is not None else "New internal split; not the official BRACS test or an untouched external cohort."}
    manifest["counts"] = {k: {"graphs": len(v), "slides": int(v.slide_id.nunique()),
                              "groups": int(v.group_id.nunique())} for k,v in frames.items()}
    for name, part in frames.items():
        part.to_csv(out / f"{name}_ids.csv", index=False)
    manifest["split_hashes"] = {name: fingerprint(out / f"{name}_ids.csv") for name in frames}
    pd.concat([v.assign(partition=k) for k,v in frames.items()]).groupby(
        ["label", "partition"]).size().unstack().to_csv(out / "class_counts.csv")
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit("Graph validation failed; see graph_errors.csv. Do not train.")


if __name__ == "__main__":
    main()
