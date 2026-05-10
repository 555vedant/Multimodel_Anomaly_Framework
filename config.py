"""
Configuration for Drift-Aware Relational Anomaly Detection System.
"""
import os
import json
from typing import Any

# ── Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EXPERIMENT_NAME = "default"

MODEL_PATH = os.path.join(OUTPUT_DIR, "autoencoder.pth")
BASELINE_CORR_PATH = os.path.join(OUTPUT_DIR, "baseline_corr.npy")
RELATION_MODELS_PATH = os.path.join(OUTPUT_DIR, "relation_models.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

# ── Data split
NUM_BATCHES = 10
TRAIN_BATCHES = [1, 2, 3, 4, 5]          # Batches used for training
TEST_BATCHES = [6, 7, 8, 9, 10]          # Batches used for testing
NUM_FEATURES = 128                         # 16 sensors × 8 features each

# ── Temporal autoencoder 
HIDDEN_DIM = 64
LATENT_DIM = 32
DROPOUT = 0.2
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10                  

# ── Relational module 
CORR_THRESHOLD = 0.7                      # |r| above this → "strongly correlated"
MAX_CORR_PAIRS = 500                      # cap to keep scoring fast

# ── Anomaly scoring 
ALPHA = 0.6                               # weight for temporal (reconstruction) error
BETA = 0.4                                # weight for relational error
ANOMALY_SIGMA = 2.0                       # threshold = mean + sigma * std
THRESHOLD_METHOD = "mean_std"            # "mean_std" or "percentile"
ANOMALY_PERCENTILE = 98.0                 # used when THRESHOLD_METHOD="percentile"
SCORE_NORM_METHOD = "minmax"             # "minmax" or "robust"
ROBUST_NORM_LOW_Q = 1.0                   # lower percentile for robust score scaling
ROBUST_NORM_HIGH_Q = 99.0                 # upper percentile for robust score scaling

SEED = 42


def _refresh_paths() -> None:
	"""Refresh all derived paths after OUTPUT_DIR or EXPERIMENT_NAME changes."""
	global MODEL_PATH, BASELINE_CORR_PATH, RELATION_MODELS_PATH, SCALER_PATH
	MODEL_PATH = os.path.join(OUTPUT_DIR, "autoencoder.pth")
	BASELINE_CORR_PATH = os.path.join(OUTPUT_DIR, "baseline_corr.npy")
	RELATION_MODELS_PATH = os.path.join(OUTPUT_DIR, "relation_models.pkl")
	SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")


def apply_overrides(overrides: dict[str, Any]) -> None:
	"""Apply a dictionary of config overrides to module globals."""
	global OUTPUT_DIR
	for key, value in overrides.items():
		if not hasattr(__import__(__name__), key):
			raise KeyError(f"Unknown config key: {key}")
		globals()[key] = value

	if "EXPERIMENT_NAME" in overrides and "OUTPUT_DIR" not in overrides:
		OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", str(EXPERIMENT_NAME))

	_refresh_paths()


def apply_overrides_from_json(json_path: str) -> None:
	"""Load and apply overrides from a JSON file."""
	with open(json_path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	if not isinstance(payload, dict):
		raise ValueError(f"Config file must contain a JSON object: {json_path}")
	apply_overrides(payload)
