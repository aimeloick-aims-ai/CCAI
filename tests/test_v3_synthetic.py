"""End-to-end smoke test of the v3 audit on SYNTHETIC graphs with planted ground truth.

Positive control : labels depend on edge distance -> spatial_distance must be flagged as functional.
Negative control : feature 55 (texture) is pure noise -> must NOT be flagged.
Decodability     : test the full-width (128D) GINE used in the real experiment.
The earlier 32D model compressed 96 inputs; its failed decoding threshold is preserved historically.

Run from the project root:  python tests/test_v3_synthetic.py
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_gine_baseline import GINEBaseline, GraphBRACSDataset, LABELS  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--hidden-dim', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=25)
    args = ap.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    tmp = Path(args.out_dir).resolve() if args.out_dir else Path(tempfile.mkdtemp())
    if args.out_dir:
        if tmp.exists():
            raise SystemExit(f"Refusing to overwrite {tmp}")
        tmp.mkdir(parents=True)
    print(f"Synthetic artifacts: {tmp}", flush=True)
    (tmp / "data/graph-bracs/data").mkdir(parents=True)
    (tmp / "processed/graph-bracs").mkdir(parents=True)


    def make_graph(cls):
        n = int(rng.integers(20, 50))
        x = rng.normal(50, 10, size=(n, 96)).astype(np.float32)
        area = rng.normal(45, 12, size=n)
        x[:, 23] = area
        x[:, 24] = 0.97 * area + rng.normal(0, 2, size=n)  # redundant with area (corr ~0.95+)
        x[:, 90] = rng.normal(90 + 1.0 * cls, 15, size=n)  # weak class signal
        x[:, 55] = rng.normal(1.5e-4, 5e-5, size=n)  # texture: pure noise
        m = n * 2
        src = rng.integers(0, n, size=m)
        dst = rng.integers(0, n, size=m)
        dist = (10 + 4 * cls) * rng.uniform(0.7, 1.3, size=m)  # planted class signal
        return Data(x=torch.tensor(x), edge_index=torch.tensor(np.stack([src, dst])),
                    edge_attr=torch.tensor(dist[:, None].astype(np.float32)))


    rows = []
    for split, n_wsi in (("train", 40), ("validation", 8), ("test", 10)):
        for w in range(n_wsi):
            wid = {"train": 1000, "validation": 2000, "test": 3000}[split] + w
            for k in range(int(rng.integers(3, 8))):
                cls = int(rng.integers(0, 7))
                sid = f"BRACS_{wid}_{LABELS[cls]}_{k}"
                path = f"graph-bracs/data/{sid}.pt"
                torch.save(make_graph(cls), tmp / "data" / path)
                rows.append(dict(sample_id=sid, label=LABELS[cls], graph_path=path, split=split))
    df = pd.DataFrame(rows)
    df[df.split == "train"][["sample_id", "label", "graph_path"]].to_csv(tmp / "processed/graph-bracs/train_ids.csv", index=False)
    df[df.split == "validation"][["sample_id", "label", "graph_path"]].to_csv(tmp / "processed/graph-bracs/val_ids.csv", index=False)
    df.rename(columns={"split": "split"}).to_csv(tmp / "data/graph-bracs/metadata.csv", index=False)
    meta = df.copy()
    meta["split"] = meta["split"].replace({"validation": "train"})  # official: val is part of official train
    meta.to_csv(tmp / "data/graph-bracs/metadata.csv", index=False)

    # scaler from TRAIN nodes/edges, as in the real pipeline
    import os  # noqa: E402

    os.chdir(tmp)
    tr = df[df.split == "train"]
    xs, es = [], []
    for p in tr.graph_path:
        g = torch.load(tmp / "data" / p, weights_only=False)
        xs.append(g.x.numpy())
        es.append(g.edge_attr.numpy())
    xs, es = np.concatenate(xs), np.concatenate(es)
    scaler = dict(x_mean=xs.mean(0), x_std=xs.std(0) + 1e-8, edge_mean=float(es.mean()), edge_std=float(es.std()))

    # quick training of a small GINE
    cfg = dict(input_dim=96, hidden_dim=args.hidden_dim, num_layers=3, edge_dim=1, num_classes=7, dropout=0.1)
    torch.manual_seed(0)
    model = GINEBaseline(**cfg)
    loader = DataLoader(GraphBRACSDataset(tr, scaler["x_mean"], scaler["x_std"], scaler["edge_mean"], scaler["edge_std"]),
                        batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(args.epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
            loss.backward()
            opt.step()
    print("final train loss", float(loss.detach()))
    torch.save(dict(model_state_dict=model.state_dict(), model_config=cfg, scaler=scaler, labels=LABELS,
                    seed=0, train_ids=tr.sample_id.tolist(), validation_ids=df[df.split == "validation"].sample_id.tolist(),
                    best_val_macro_f1=float("nan"), official_test={"macro_f1": float("nan")}), tmp / "ckpt.pt")

    cmd = [sys.executable, str(ROOT / "scripts/run_audit_v3.py"), "--checkpoint", "ckpt.pt", "--out-dir", "out",
           "--splits", "official_test", "--n-untrained", "2", "--n-random-features", "6", "--n-boot", "200",
           "--batch-size", "8"]
    r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")})
    print(r.stdout[-3500:])
    if r.returncode:
        print(r.stderr[-3000:])
        sys.exit(1)

    base = pd.read_csv(tmp / "out/decodability_baselines.csv")
    iv = pd.read_csv(tmp / "out/interventions_official_test.csv")
    print(base.round(3).to_string(index=False))
    cols = ["concept", "alpha", "isolated_dlogit", "isolated_vs_randfeat", "isolated_beyond_null", "coherent_beyond_null", "isolated_dF1"]
    print(iv[cols].round(3).to_string(index=False))
    assert (tmp / "out/joint_official_test.csv").exists()

    # positive / negative controls
    sp = iv[(iv.concept == "spatial_distance") & (iv.alpha.abs() == 2.0)]
    tx = iv[(iv.concept == "cell_texture") & (iv.alpha.abs() == 2.0)]
    sp_ok = bool(sp.isolated_beyond_null.any())
    tx_ok = not bool(tx.isolated_beyond_null.any())
    # untrained baseline should already decode the planted concept features well
    b = base[(base.split == "official_test") & (base.concept == "cell_size")].iloc[0]
    checks = {"spatial_positive": sp_ok, "texture_negative": tx_ok,
              "untrained_area_decoding_above_0_5": bool(b.r2_untrained_mean > 0.5)}
    status = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "r2_untrained_area": float(b.r2_untrained_mean), "model_config": cfg,
              "epochs": args.epochs, "threshold_changed": False}
    (tmp / "status.json").write_text(json.dumps(status, indent=2))
    assert all(checks.values()), f"Synthetic benchmark failed: {status}"
    print("ALL SYNTHETIC CHECKS PASSED")


if __name__ == '__main__':
    main()
