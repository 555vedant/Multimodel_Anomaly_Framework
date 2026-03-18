"""
Configuration for Drift-Aware Relational Anomaly Detection System.
"""
import os

# ── Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

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

SEED = 42
