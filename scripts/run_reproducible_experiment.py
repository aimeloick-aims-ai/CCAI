"""Run one explicit stage; refuse incomplete prerequisites and preserve old runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(command, log_path):
    print("Running:", subprocess.list2cmdline(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ValueError(f"Log already exists: {log_path}; inspect the previous attempt first")
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise SystemExit(f"Stage failed. Read {log_path}; do not continue to later stages.")


def readiness(work):
    p = work / "split_manifest.json"
    if not p.exists():
        raise SystemExit("Run the prepare stage first")
    manifest = json.loads(p.read_text())
    if not manifest["ready_for_training"]:
        raise SystemExit("Full graph preflight did not pass; inspect graph_errors.csv")
    metadata_path = Path(manifest.get("source_metadata_path", "data/graph-bracs/metadata.csv"))
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != manifest["source_metadata_sha256"]:
        raise SystemExit("Source metadata changed after the full preflight")
    for split, digest in manifest["split_hashes"].items():
        if hashlib.sha256((work / f"{split}_ids.csv").read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Frozen {split} IDs were modified")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["prepare", "scaler", "train", "audit", "collect"])
    ap.add_argument("--work-dir", default="results/reproducible_run")
    ap.add_argument("--patient-mapping")
    ap.add_argument("--source-partitions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.chdir(ROOT)
    work = Path(args.work_dir)
    if args.stage == "prepare":
        if (work / "split_manifest.json").exists():
            m = readiness(work)
            if args.patient_mapping and not m["patient_independence_verified"]:
                raise SystemExit("Use a NEW work directory for patient-grouped partitions")
            if args.source_partitions and m.get("protocol") != "restored_bracs_partitions_available_graphs":
                raise SystemExit("Use a NEW work directory to restore source partitions")
            print(json.dumps(m, indent=2))
            return
        command = [sys.executable, "scripts/prepare_reproducible_run.py", "--out-dir", str(work)]
        if args.patient_mapping:
            command += ["--patient-mapping", args.patient_mapping]
        if args.source_partitions:
            command += ["--source-partitions", args.source_partitions]
        run(command, work.parent / f"{work.name}.prepare.log")
        return
    manifest = readiness(work)
    train, val, test = [str(work / f"{s}_ids.csv") for s in ("train", "validation", "test")]
    scaler = work / "scaler.npz"
    seed_dir = work / f"seed{args.seed}"
    checkpoint = seed_dir / "model.pt"
    if args.stage == "scaler":
        if scaler.exists():
            import numpy as np
            with np.load(scaler) as z:
                if "train_ids" not in z or set(z["train_ids"]) != set(pd.read_csv(train).sample_id):
                    raise SystemExit("Scaler training IDs differ from frozen split")
            print("Verified existing train-only scaler; no overwrite.")
            return
        run([sys.executable, "scripts/fit_graph_bracs_scaler.py", "--train-csv", train,
             "--output", str(scaler)], work / "scaler.log")
    elif args.stage == "train":
        if not scaler.exists():
            raise SystemExit("Run --stage scaler first")
        if checkpoint.exists():
            raise SystemExit(f"Checkpoint exists: {checkpoint}. Proceed to audit, or use another seed.")
        seed_dir.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "scripts/train_gine_baseline.py", "--seed", str(args.seed),
             "--train-csv", train, "--val-csv", val, "--test-csv", test, "--scaler", str(scaler),
             "--output", str(checkpoint), "--epochs", str(args.epochs), "--threads", "2"], seed_dir / "train.log")
    elif args.stage == "audit":
        if not checkpoint.exists():
            raise SystemExit("Train this seed first")
        run([sys.executable, "scripts/run_audit_v3.py", "--checkpoint", str(checkpoint),
             "--train-csv", train, "--val-csv", val, "--test-csv", test, "--out-dir", str(seed_dir / "audit"),
             "--splits", "test", "--n-untrained", "5", "--n-random-features", "12",
             "--n-boot", str(args.n_boot), "--skip", "joint", "--threads", "2"], seed_dir / "audit.log")
    elif args.stage == "collect":
        run([sys.executable, "scripts/summarize_reproducible_run.py", "--work-dir", str(work)],
            work / f"collect_{len(list(work.glob('seed*/audit/manifest.json')))}.log")
    print(f"Stage completed: {args.stage}. Grouping: {manifest['grouping']}", flush=True)


if __name__ == "__main__":
    main()
