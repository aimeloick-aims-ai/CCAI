import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from train_gine_baseline import (
    GINEBaseline,
    GraphBRACSDataset,
)

from .config import (
    BATCH_SIZE,
    CHECKPOINT,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    EXPECTED_VAL,
    METADATA_CSV,
    SCALER_PATH,
    TRAIN_CSV,
    VAL_CSV,
)


def load_context():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    metadata_df = pd.read_csv(
        METADATA_CSV
    )

    test_df = metadata_df[
        metadata_df["split"] == "test"
    ].copy()

    ids = [set(frame["sample_id"]) for frame in (train_df, val_df, test_df)]
    for frame, split_ids in zip((train_df, val_df, test_df), ids):
        if len(frame) != len(split_ids):
            raise ValueError("Duplicate graph identifiers in an audit split")
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Audit splits contain overlapping graphs")

    if len(train_df) != EXPECTED_TRAIN:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN} train graphs, "
            f"found {len(train_df)}"
        )

    if len(val_df) != EXPECTED_VAL:
        raise ValueError(
            f"Expected {EXPECTED_VAL} validation graphs, "
            f"found {len(val_df)}"
        )

    if len(test_df) != EXPECTED_TEST:
        raise ValueError(
            f"Expected {EXPECTED_TEST} test graphs, "
            f"found {len(test_df)}"
        )

    scaler = np.load(
        SCALER_PATH
    )

    x_mean = scaler["x_mean"]
    x_std = scaler["x_std"]

    edge_mean = float(
        scaler["edge_mean"]
    )

    edge_std = float(
        scaler["edge_std"]
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "x_mean": x_mean,
        "x_std": x_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def load_frozen_model(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model_config = checkpoint[
        "model_config"
    ]

    model = GINEBaseline(
        **model_config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, model_config
