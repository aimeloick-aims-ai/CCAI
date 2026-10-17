from pathlib import Path
import copy
import random
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from concept_audit_v3.integrity import validate_splits

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
    global_max_pool,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_CSV = Path(
    "processed/graph-bracs/train_ids.csv"
)

VAL_CSV = Path(
    "processed/graph-bracs/val_ids.csv"
)

METADATA_CSV = Path(
    "data/graph-bracs/metadata.csv"
)

SCALER_PATH = Path(
    "processed/graph-bracs/scaler.npz"
)

OUTPUT_PATH = Path(
    "results/gine_baseline.pt"
)

# Training
BATCH_SIZE = 8
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 80
PATIENCE = 12

NUM_CLASSES = 7

LABELS = [
    "ADH",
    "DCIS",
    "FEA",
    "IC",
    "N",
    "PB",
    "UDH",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_graph_path(graph_path):

    path = Path(graph_path)

    if path.exists():
        return path

    alternative = Path("data") / path

    if alternative.exists():
        return alternative

    raise FileNotFoundError(
        f"Graph not found. Tried:\n"
        f"  {path}\n"
        f"  {alternative}"
    )


# ============================================================
# DATASET
# ============================================================

class GraphBRACSDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.x_mean = torch.tensor(
            x_mean,
            dtype=torch.float32,
        )

        self.x_std = torch.tensor(
            x_std,
            dtype=torch.float32,
        )

        self.edge_mean = float(
            edge_mean
        )

        self.edge_std = float(
            edge_std
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = resolve_graph_path(
            row["graph_path"]
        )

        graph = torch.load(
            path,
            weights_only=False,
        )

        x = graph.x.float()

        edge_index = (
            graph.edge_index.long()
        )

        edge_attr = (
            graph.edge_attr.float()
        )

        # ----------------------------------------------------
        # NORMALIZATION
        #
        # x_mean/x_std and edge_mean/edge_std come ONLY
        # from the training set.
        # ----------------------------------------------------

        x = (
            x - self.x_mean
        ) / self.x_std

        edge_attr = (
            edge_attr - self.edge_mean
        ) / self.edge_std

        y = torch.tensor(
            LABEL_TO_ID[row["label"]],
            dtype=torch.long,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )


# ============================================================
# GINE MODEL
# ============================================================

class GINEBaseline(nn.Module):

    def __init__(
        self,
        input_dim=96,
        hidden_dim=128,
        num_layers=3,
        edge_dim=1,
        num_classes=7,
        dropout=0.25,
    ):

        super().__init__()

        self.dropout = dropout

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            mlp = nn.Sequential(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )

            conv = GINEConv(
                mlp,
                edge_dim=edge_dim,
                train_eps=True,
            )

            self.convs.append(conv)

            self.norms.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

        # Mean pooling + Max pooling
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):

        x = self.input_projection(x)

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):

            residual = x

            x = conv(
                x,
                edge_index,
                edge_attr,
            )

            x = norm(x)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

            x = x + residual

        mean_pool = global_mean_pool(
            x,
            batch,
        )

        max_pool = global_max_pool(
            x,
            batch,
        )

        graph_embedding = torch.cat(
            [
                mean_pool,
                max_pool,
            ],
            dim=1,
        )

        logits = self.classifier(
            graph_embedding
        )

        return logits


# ============================================================
# METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
    num_classes,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(
        y_true == y_pred
    )

    f1_values = []

    for c in range(num_classes):

        tp = np.sum(
            (y_true == c)
            & (y_pred == c)
        )

        fp = np.sum(
            (y_true != c)
            & (y_pred == c)
        )

        fn = np.sum(
            (y_true == c)
            & (y_pred != c)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        else:

            f1 = 0.0

        f1_values.append(f1)

    macro_f1 = float(
        np.mean(f1_values)
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "f1_per_class": f1_values,
    }


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_graphs = 0

    all_true = []
    all_pred = []

    for batch in loader:

        batch = batch.to(device)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            loss = criterion(
                logits,
                batch.y,
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        n = batch.y.size(0)

        total_loss += (
            loss.item() * n
        )

        total_graphs += n

        predictions = (
            logits.argmax(dim=1)
        )

        all_true.extend(
            batch.y.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = classification_metrics(
        all_true,
        all_pred,
        NUM_CLASSES,
    )

    metrics["loss"] = (
        total_loss / total_graphs
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def parse_args():
    import argparse

    ap = argparse.ArgumentParser(description="Train the Graph-BRACS GINE baseline")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-csv", default=str(TRAIN_CSV))
    ap.add_argument("--val-csv", default=str(VAL_CSV))
    ap.add_argument("--scaler", default=str(SCALER_PATH))
    ap.add_argument("--output", default=str(OUTPUT_PATH))
    ap.add_argument("--test-csv", help="Explicit internal holdout; not the dataset-supplied test")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    return ap.parse_args()


def main():

    global SEED, TRAIN_CSV, VAL_CSV, SCALER_PATH, OUTPUT_PATH, MAX_EPOCHS, PATIENCE, BATCH_SIZE, HIDDEN_DIM
    _a = parse_args()
    protocol_manifest = Path(_a.train_csv).parent / "split_manifest.json"
    experiment_protocol = (json.loads(protocol_manifest.read_text())["protocol"] if protocol_manifest.exists()
                           else "internal_grouped_exploratory" if _a.test_csv else "dataset_supplied_test")
    SEED, TRAIN_CSV, VAL_CSV = _a.seed, Path(_a.train_csv), Path(_a.val_csv)
    SCALER_PATH, OUTPUT_PATH = Path(_a.scaler), Path(_a.output)
    MAX_EPOCHS, PATIENCE, BATCH_SIZE, HIDDEN_DIM = _a.epochs, _a.patience, _a.batch_size, _a.hidden_dim
    if min(MAX_EPOCHS, PATIENCE, BATCH_SIZE, HIDDEN_DIM, _a.threads) < 1:
        raise ValueError("Epochs, patience, batch size, hidden dimension and threads must be positive")
    torch.set_num_threads(_a.threads)
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT_PATH}")

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("==========================================")
    print("GRAPH-BRACS GINE BASELINE")
    print("==========================================")
    print(f"Device: {device}")
    print(f"Seed:   {SEED}")

    # ========================================================
    # LOAD TRAIN / VALIDATION
    # ========================================================

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    # ========================================================
    # LOAD OFFICIAL TEST DIRECTLY FROM METADATA
    # ========================================================

    metadata_df = pd.read_csv(
        METADATA_CSV
    )

    test_df = metadata_df[
        metadata_df["split"] == "test"
    ].copy()
    if _a.test_csv:
        test_df = pd.read_csv(_a.test_csv)
    validate_splits({"train": train_df, "validation": val_df, "test": test_df})

    # ========================================================
    # VERIFY SPLITS
    # ========================================================

    for name, frame in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if set(frame.label) != set(LABELS):
            raise ValueError(f"All seven classes are required in {name}")
        print(f"{name}: {len(frame)} graphs", flush=True)
    print("Evaluation protocol:", experiment_protocol)

    # ========================================================
    # LOAD TRAIN-ONLY SCALER
    # ========================================================

    scaler = np.load(
        SCALER_PATH
    )
    if "train_ids" not in scaler or set(scaler["train_ids"]) != set(train_df.sample_id):
        raise ValueError("Scaler training IDs missing or mismatched. Refit it on the specified training CSV.")

    x_mean = scaler["x_mean"]
    x_std = scaler["x_std"]

    edge_mean = float(
        scaler["edge_mean"]
    )

    edge_std = float(
        scaler["edge_std"]
    )

    scaler_nodes = int(
        scaler["node_count"]
    )

    scaler_edges = int(
        scaler["edge_count"]
    )

    print()
    print(
        "Train-only scaler loaded:"
    )

    print(
        f"  Node features: {len(x_mean)}"
    )

    print(
        f"  Training nodes: "
        f"{scaler_nodes:,}"
    )

    print(
        f"  Training edges: "
        f"{scaler_edges:,}"
    )

    print(
        f"  Edge mean: {edge_mean:.6f}"
    )

    print(
        f"  Edge std:  {edge_std:.6f}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = GraphBRACSDataset(
        train_df,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
    )

    val_dataset = GraphBRACSDataset(
        val_df,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
    )

    test_dataset = GraphBRACSDataset(
        test_df,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    counts = (
        train_df["label"]
        .value_counts()
        .reindex(LABELS)
        .values
        .astype(np.float32)
    )

    class_weights_np = (
        counts.sum()
        / (
            len(LABELS)
            * counts
        )
    )

    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device,
    )

    print()
    print("Class weights:")

    for label, weight in zip(
        LABELS,
        class_weights_np,
    ):

        print(
            f"  {label}: {weight:.4f}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = GINEBaseline(
        input_dim=96,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        edge_dim=1,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
        )
    )

    print()
    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_f1 = -np.inf
    best_state = None
    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics[
                    "loss"
                ],
                "train_accuracy": train_metrics[
                    "accuracy"
                ],
                "train_macro_f1": train_metrics[
                    "macro_f1"
                ],
                "val_loss": val_metrics[
                    "loss"
                ],
                "val_accuracy": val_metrics[
                    "accuracy"
                ],
                "val_macro_f1": val_metrics[
                    "macro_f1"
                ],
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val F1 {val_metrics['macro_f1']:.4f}", flush=True
        )

        # ----------------------------------------------------
        # BEST VALIDATION MODEL
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            print(
                f"  -> new best validation "
                f"macro-F1: {best_val_f1:.4f}"
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "No best model was saved."
        )

    model.load_state_dict(
        best_state
    )

    print()
    print("==========================================")
    print("BEST MODEL")
    print("==========================================")

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation macro-F1: "
        f"{best_val_f1:.4f}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_val = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        training=False,
    )

    print()
    print("Validation:")
    print(
        f"  Accuracy: "
        f"{final_val['accuracy']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{final_val['macro_f1']:.4f}"
    )

    # ========================================================
    # OFFICIAL TEST
    #
    # The model has already been selected using validation.
    # The official test is evaluated only now.
    # ========================================================

    final_test = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        training=False,
    )

    print()
    print("==========================================")
    print("INTERNAL HOLDOUT" if _a.test_csv else "DATASET-SUPPLIED TEST")
    print("==========================================")

    print(
        f"Accuracy: "
        f"{final_test['accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{final_test['macro_f1']:.4f}"
    )

    print()
    print("Per-class F1:")

    for label, score in zip(
        LABELS,
        final_test["f1_per_class"],
    ):

        print(
            f"  {label}: {score:.4f}"
        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "model_config": {
            "input_dim": 96,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "edge_dim": 1,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
        },

        "labels": LABELS,

        "scaler": {
            "x_mean": x_mean,
            "x_std": x_std,
            "edge_mean": edge_mean,
            "edge_std": edge_std,
        },

        "best_epoch": best_epoch,

        "best_val_macro_f1": best_val_f1,

        "final_validation": final_val,

        "official_test": final_test,

        "history": history,

        "seed": SEED,
        "train_ids": train_df.sample_id.tolist(),
        "validation_ids": val_df.sample_id.tolist(),
        "test_ids": test_df.sample_id.tolist(),
        "protocol": experiment_protocol,
        "scaler_train_ids": scaler["train_ids"].tolist(),
        "split_hashes": {"train": hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest(),
                         "validation": hashlib.sha256(VAL_CSV.read_bytes()).hexdigest(),
                         "test": hashlib.sha256(Path(_a.test_csv or METADATA_CSV).read_bytes()).hexdigest()},
        "training_arguments": vars(_a),
    }

    torch.save(
        checkpoint,
        OUTPUT_PATH,
    )
    pd.DataFrame(history).to_csv(OUTPUT_PATH.with_suffix(".history.csv"), index=False)
    summary = {"seed": SEED, "best_epoch": best_epoch, "validation": final_val, "test": final_test,
               "protocol": checkpoint["protocol"], "split_hashes": checkpoint["split_hashes"]}
    OUTPUT_PATH.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(
        f"Checkpoint saved: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
