from pathlib import Path

SEED = 42

CHECKPOINT = Path("results/gine_baseline.pt")
TRAIN_CSV = Path("processed/graph-bracs/train_ids.csv")
VAL_CSV = Path("processed/graph-bracs/val_ids.csv")
METADATA_CSV = Path("data/graph-bracs/metadata.csv")
SCALER_PATH = Path("processed/graph-bracs/scaler.npz")
OUTPUT = Path("results/concept_audit_phase1_v2.npz")

BATCH_SIZE = 4
MAX_NODE_SAMPLES_PER_GRAPH = 100
RIDGE_ALPHA = 1e-2
NUM_RANDOM_CONTROLS = 3

NODE_CONCEPTS = {
    "cell_size": 23,
    "cell_shape": 27,
    "cell_intensity": 90,
    "cell_texture": 55,
}

NODE_CONCEPT_FEATURE_NAMES = {
    "cell_size": "cell_size_area",
    "cell_shape": "cell_circularity",
    "cell_intensity": "cell_mean_intensity_gray",
    "cell_texture": "cell_texture_contrast",
}

GRAPH_CONCEPT = "spatial_distance"

EXPECTED_TRAIN = 2875
EXPECTED_VAL = 719
EXPECTED_TEST = 899
