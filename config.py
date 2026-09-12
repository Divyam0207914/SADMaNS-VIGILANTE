"""
SADMaNS / VIGILANTE — Central Configuration
=============================================

All configurable parameters for the MVP are defined here.
Do not scatter configuration values throughout other files.
Import from this module: `from config import *` or `import config`.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "02-15-2018.csv")


# ─────────────────────────────────────────────
# TEMPORAL
# ─────────────────────────────────────────────
WINDOW_SIZE = "1min"

# Number of past windows used as input to the GRU.
SEQ_LEN = 5


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────
# For the MVP, nodes are destination ports (not hosts).
GRAPH_MODE = "DESTINATION_PORT"

# Keep only the top-K most frequent ports per window
# to keep graph sizes manageable.
TOP_K_PORTS = 50


# ─────────────────────────────────────────────
# NODE FEATURES (per-port, per-window)
# ─────────────────────────────────────────────
NODE_FEATURES = [
    "flow_count",
    "total_fwd_pkts",
    "total_bwd_pkts",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "mean_flow_duration",
    "mean_flow_pkts_per_s",
    "mean_flow_byts_per_s",
    "mean_pkt_len",
    "mean_flow_iat",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "fin_flag_count",
    "mean_fwd_pkts_per_s",
    "mean_bwd_pkts_per_s",
]

# Dimensionality shortcut (17 features).
NODE_FEATURE_DIM = len(NODE_FEATURES)


# ─────────────────────────────────────────────
# GLOBAL STATE (per-window aggregate)
# ─────────────────────────────────────────────
GLOBAL_STATE_FEATURES = [
    "flow_count",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_bytes",
    "mean_flow_duration",
    "mean_pkt_len",
    "mean_flow_iat",
    "syn_rate",
    "rst_rate",
    "psh_rate",
    "unique_dst_ports",
    "port_entropy",
    "protocol_ratio_tcp",
]

# Dimensionality shortcut (13 features).
GLOBAL_STATE_DIM = len(GLOBAL_STATE_FEATURES)


# ─────────────────────────────────────────────
# MODEL — GCN
# ─────────────────────────────────────────────
GCN_HIDDEN_DIM = 64
GCN_DROPOUT = 0.2

# Dimensionality of the graph embedding produced by
# GCN + Global Mean Pooling.
GCN_EMBED_DIM = GCN_HIDDEN_DIM  # 64


# ─────────────────────────────────────────────
# MODEL — GRU
# ─────────────────────────────────────────────
GRU_HIDDEN_DIM = 128
GRU_LAYERS = 1

# GRU input size = GCN embedding + global state.
GRU_INPUT_DIM = GCN_EMBED_DIM + GLOBAL_STATE_DIM  # 64 + 13 = 77


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 20
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# Phase 7 Multi-task loss weights
STATE_LOSS_WEIGHT = 1.0
RISK_LOSS_WEIGHT = 1.0
TYPE_LOSS_WEIGHT = 1.0

# Phase 7 Attack Type Mappings
ATTACK_TYPE_TO_ID = {
    "Benign": 0,
    "DoS attacks-GoldenEye": 1,
    "DoS attacks-Slowloris": 2
}
ID_TO_ATTACK_TYPE = {v: k for k, v in ATTACK_TYPE_TO_ID.items()}

# ─────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────
GNN_EXPLAINER_EPOCHS = 200


# ─────────────────────────────────────────────
# OUTPUT / CHECKPOINTS
# ─────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "world_model.pt")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "label_encoder.pkl")


# ─────────────────────────────────────────────
# DEVICE SELECTION
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority:
        1. CUDA  (NVIDIA GPU)
        2. MPS   (Apple Silicon GPU)
        3. CPU   (fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across numpy, torch, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may reduce performance slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# CONFIGURATION VALIDATION
# ─────────────────────────────────────────────
def validate_config() -> list[str]:
    """
    Validate critical configuration values.

    Returns a list of warning/error strings.
    An empty list means the configuration is valid.
    Does NOT load the CSV — only checks paths and param ranges.
    """
    issues: list[str] = []

    # Data path
    if not os.path.isfile(DATA_PATH):
        issues.append(f"DATA_PATH not found: {DATA_PATH}")

    # Positive-integer checks
    checks = {
        "TOP_K_PORTS": TOP_K_PORTS,
        "SEQ_LEN": SEQ_LEN,
        "GCN_HIDDEN_DIM": GCN_HIDDEN_DIM,
        "GRU_HIDDEN_DIM": GRU_HIDDEN_DIM,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "GRU_LAYERS": GRU_LAYERS,
        "GNN_EXPLAINER_EPOCHS": GNN_EXPLAINER_EPOCHS,
    }
    for name, value in checks.items():
        if not isinstance(value, int) or value <= 0:
            issues.append(f"{name} must be a positive integer, got {value!r}")

    # Positive-float checks
    float_checks = {
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "GCN_DROPOUT": GCN_DROPOUT,
    }
    for name, value in float_checks.items():
        if not isinstance(value, (int, float)) or value <= 0:
            issues.append(f"{name} must be a positive number, got {value!r}")

    # Dropout in [0, 1)
    if not (0.0 <= GCN_DROPOUT < 1.0):
        issues.append(f"GCN_DROPOUT must be in [0, 1), got {GCN_DROPOUT}")

    # Feature lists non-empty
    if len(NODE_FEATURES) == 0:
        issues.append("NODE_FEATURES list is empty")
    if len(GLOBAL_STATE_FEATURES) == 0:
        issues.append("GLOBAL_STATE_FEATURES list is empty")

    # Checkpoint directory
    if not os.path.isdir(CHECKPOINT_DIR):
        # Not an error — will be created during training.
        pass

    return issues


# ─────────────────────────────────────────────
# Immediate validation when config is imported
# (prints warnings but does not raise)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    issues = validate_config()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  ⚠  {issue}")
    else:
        print("Configuration OK.")
    print(f"\nDevice: {DEVICE}")
    print(f"Data:   {DATA_PATH}")
    print(f"Graph:  {GRAPH_MODE} (top-{TOP_K_PORTS} ports)")
    print(f"Window: {WINDOW_SIZE}, seq_len={SEQ_LEN}")
    print(f"GCN:    hidden={GCN_HIDDEN_DIM}, dropout={GCN_DROPOUT}")
    print(f"GRU:    hidden={GRU_HIDDEN_DIM}, layers={GRU_LAYERS}, input={GRU_INPUT_DIM}")
    print(f"Train:  lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}")
    print(f"Node features:   {NODE_FEATURE_DIM}")
    print(f"Global features: {GLOBAL_STATE_DIM}")
