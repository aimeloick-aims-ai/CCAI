"""Run the v3 audit (baselines, controlled interventions, joint interventions) for one checkpoint.

Example (after retraining on the grouped split, see docs/V3_README.md):
    python scripts/run_audit_v3.py --checkpoint results/gine_grouped_seed42.pt \
        --train-csv processed/graph-bracs/train_ids_grouped.csv \
        --val-csv   processed/graph-bracs/val_ids_grouped.csv \
        --out-dir   results/audit_v3_grouped_seed42

Independent-test claims require verified slide/patient separation and training provenance.
The current local official TEST shares slide IDs with TRAIN/VALIDATION. Historical
reanalysis requires --allow-slide-overlap and remains descriptive.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from concept_audit_v3 import baselines, interventions_v3, report_v3  # noqa: E402
from concept_audit_v3.integrity import validate_splits
from concept_audit_v3.stats import permutation_pvalue, wsi_ids  # noqa: E402
from concept_decodability_audit.utils import set_seed  # noqa: E402
from train_gine_baseline import GINEBaseline, GraphBRACSDataset  # noqa: E402

METADATA_CSV = Path("data/graph-bracs/metadata.csv")


def make_loader(df, scaler, batch_size):
    ds = GraphBRACSDataset(df, scaler["x_mean"], scaler["x_std"], float(scaler["edge_mean"]), float(scaler["edge_std"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--allow-slide-overlap", action="store_true", help="Historical descriptive reanalysis only; no independent-test claims")
    ap.add_argument("--train-csv", default="processed/graph-bracs/train_ids.csv")
    ap.add_argument("--val-csv", default="processed/graph-bracs/val_ids.csv")
    ap.add_argument("--test-csv", help="Explicit internal holdout CSV")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=None, choices=["validation", "official_test", "test"])
    ap.add_argument("--n-untrained", type=int, default=5)
    ap.add_argument("--n-random-features", type=int, default=12)
    ap.add_argument("--min-abs-corr", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-graphs", type=int, default=0, help="smoke test: use first N graphs per split")
    ap.add_argument("--skip", nargs="*", default=[], choices=["baselines", "variants", "joint"])
    ap.add_argument("--v2-npz", default="", help="optional: v2 phase-2 npz, to recompute TCAV permutation p-values")
    args = ap.parse_args()
    if args.splits is None:
        args.splits = ["test" if args.test_csv else "official_test"]
    if args.test_csv and "official_test" in args.splits:
        ap.error("Use --splits test for an internal holdout, never official_test")
    if not args.test_csv and "test" in args.splits:
        ap.error("--splits test requires --test-csv")
    if min(args.n_untrained, args.batch_size, args.n_boot, args.threads) < 1 or args.n_random_features < 0:
        ap.error("Invalid run sizes")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"Refusing to write into non-empty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    scaler = ckpt["scaler"]  # statistics the model was trained with (TRAIN only)
    model = GINEBaseline(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(METADATA_CSV).query("split == 'test'")
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
    leakage = validate_splits({"train": train_df, "validation": val_df, "test": test_df}, args.allow_slide_overlap)
    provenance_verified = "train_ids" in ckpt and "validation_ids" in ckpt
    if provenance_verified:
        if set(ckpt["train_ids"]) != set(train_df.sample_id) or set(ckpt["validation_ids"]) != set(val_df.sample_id):
            raise ValueError("Checkpoint training/selection IDs do not match the supplied splits")
        if "test_ids" in ckpt and set(ckpt["test_ids"]) != set(test_df.sample_id):
            raise ValueError("Checkpoint holdout IDs do not match the requested test")
    else:
        print("WARNING: historical checkpoint has no training IDs; training provenance is unverified.")
    frames = {"validation": val_df, "test" if args.test_csv else "official_test": test_df}
    if args.max_graphs:
        train_df = train_df.iloc[: args.max_graphs * 3]
        frames = {k: v.iloc[: args.max_graphs] for k, v in frames.items()}

    # --- integrity checks: disjoint ids, and slide-level leakage report -------------------------
    ids = {"train": set(train_df.sample_id), "validation": set(val_df.sample_id), "test": set(test_df.sample_id)}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    wsi = {"train": set(wsi_ids(train_df.sample_id)), "validation": set(wsi_ids(val_df.sample_id)),
           "test": set(wsi_ids(test_df.sample_id))}
    print("Slide leakage report:", leakage)

    train_loader = make_loader(train_df, scaler, args.batch_size)
    loaders = {s: make_loader(frames[s], scaler, args.batch_size) for s in args.splits}
    scales = [scaler["x_mean"], scaler["x_std"], float(scaler["edge_mean"]), float(scaler["edge_std"])]
    groups = {s: wsi_ids(frames[s].sample_id) for s in args.splits}

    manifest = dict(status="running", checkpoint=str(ckpt_path), sha256=sha, seed=ckpt.get("seed"), audit_seed=args.seed, splits=args.splits,
                    descriptive_only=any(leakage.values()) or not provenance_verified or bool(args.max_graphs),
                    protocol=ckpt.get("protocol", "historical_unverified"),
                    smoke_test=bool(args.max_graphs),
                    training_ids_verified=provenance_verified, arguments=vars(args),
                    sample_ids={"train": train_df.sample_id.tolist(), "validation": val_df.sample_id.tolist(),
                                **{s: frames[s].sample_id.tolist() for s in args.splits}},
                    train_csv=args.train_csv, val_csv=args.val_csv, leakage=leakage,
                    n_untrained=args.n_untrained, n_random_features=args.n_random_features,
                    min_abs_corr=args.min_abs_corr,
                    checkpoint_val_macro_f1=float(ckpt.get("best_val_macro_f1", float("nan"))),
                    checkpoint_test_macro_f1=float(ckpt.get("official_test", {}).get("macro_f1", float("nan"))))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if "baselines" not in args.skip:
        print("== decodability baselines")
        res = baselines.run_baselines(model, cfg, train_loader, loaders, scales, device,
                                      n_untrained=args.n_untrained, seed=args.seed)
        pd.DataFrame(baselines.summarize(res)).to_csv(out_dir / "decodability_baselines.csv", index=False)
        print(pd.DataFrame(baselines.summarize(res)).round(3).to_string(index=False))

    if "variants" not in args.skip or "joint" not in args.skip:
        support = interventions_v3.fit_train_support(train_loader, device)
        np.savez_compressed(out_dir / "train_support.npz", corr=support.corr, x_min=support.x_min, x_max=support.x_max)

    for split in args.splits:
        if "variants" not in args.skip:
            print(f"== interventions ({split})")
            store, meta = interventions_v3.evaluate_variants(
                model, loaders[split], device, support, args.n_random_features, args.min_abs_corr, seed=args.seed)
            meta["sample_ids"] = frames[split].sample_id.to_numpy(dtype=str)
            # Persist the expensive per-graph computation before summarization so a reporting
            # failure does not discard the arrays needed for re-analysis.
            report_v3.save_npz(str(out_dir / f"interventions_{split}.npz"), store, meta)
            table = report_v3.intervention_table(store, meta, groups[split], n_boot=args.n_boot, seed=args.seed)
            table.to_csv(out_dir / f"interventions_{split}.csv", index=False)
            from concept_audit_v3.stats import cluster_bootstrap_macro_f1
            f1, low, high = cluster_bootstrap_macro_f1(meta["y"], meta["base_pred"], groups[split],
                                                     n_boot=args.n_boot, seed=args.seed)
            (out_dir / f"classification_{split}.json").write_text(json.dumps(
                {"macro_f1": f1, "slide_bootstrap_ci95": [low, high], "n_graphs": len(meta["y"]),
                 "n_slides": len(set(groups[split])), "uncertainty_unit": "slide"}, indent=2))
        if "joint" not in args.skip:
            print(f"== joint interventions ({split})")
            joint, combos = interventions_v3.evaluate_joint(model, loaders[split], device, support)
            jt = report_v3.joint_table(joint, combos, (-1.0, 1.0), groups[split], n_boot=args.n_boot, seed=args.seed)
            jt.to_csv(out_dir / f"joint_{split}.csv", index=False)

    if args.v2_npz:
        z = np.load(args.v2_npz, allow_pickle=True)
        rows = []
        for c in [str(n) for n in z["concept_names"]]:
            for split, key in (("validation", "validation"), ("official_test", "test")):
                obs, null = z[f"{c}__tcav_{key}"], z[f"{c}__tcav_random_{key}"]
                for k, p in enumerate(permutation_pvalue(obs, null)):
                    rows.append(dict(concept=c, split=split, class_index=k, tcav=obs[k], p_perm=p,
                                     n_perm=null.shape[1], min_attainable_p=1.0 / (null.shape[1] + 1)))
        pd.DataFrame(rows).to_csv(out_dir / "tcav_permutation_pvalues.csv", index=False)
    manifest["status"] = "completed"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done:", out_dir)


if __name__ == "__main__":
    main()
