"""
Data loading, splitting, and scaling for the UCI Gas Sensor Array Drift Dataset.
Each .dat file is in libsvm format: label feat1:val1 feat2:val2 ...
128 features (16 sensors × 8 engineered features), 10 batches, 13 910 experiments.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

import config


# ── Loading

def load_batch(file_path: str, batch_id: int) -> pd.DataFrame:
    """Load a single batch .dat (libsvm) file and return a DataFrame."""
    X, y = load_svmlight_file(file_path, n_features=config.NUM_FEATURES)
    X = X.toarray()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["batch"] = batch_id
    df["label"] = y.astype(int)
    return df


def load_all_batches(folder: str | None = None) -> pd.DataFrame:
    """Load all 10 batch files, concatenate, and return a single DataFrame."""
    folder = folder or config.DATA_FOLDER
    frames = []
    for i in range(1, config.NUM_BATCHES + 1):
        path = os.path.join(folder, f"batch{i}.dat")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected batch file not found: {path}")
        frames.append(load_batch(path, i))
    df = pd.concat(frames, ignore_index=True)
    print(f"[preprocessing] Loaded {len(df)} samples across {config.NUM_BATCHES} batches.")
    return df


# ── Splitting 

def split_train_test(
    df: pd.DataFrame,
    train_batches: list[int] | None = None,
    test_batches: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by batch number. Returns (train_df, test_df)."""
    train_batches = train_batches or config.TRAIN_BATCHES
    test_batches = test_batches or config.TEST_BATCHES

    overlap = set(train_batches).intersection(test_batches)
    if overlap:
        raise ValueError(f"Train and test batches overlap: {sorted(overlap)}")

    train_df = df[df["batch"].isin(train_batches)].copy()
    test_df = df[df["batch"].isin(test_batches)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Empty train/test split. Check TRAIN_BATCHES and TEST_BATCHES in config."
        )

    print(f"[preprocessing] Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")
    return train_df, test_df


# ── Feature columns helper 

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric feature column names (exclude batch/label)."""
    return [c for c in df.columns if c not in ("batch", "label")]


# ── Scaling 

def scale_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on train features and transform both splits.
    Optionally persists the scaler to disk.
    """
    feats = feature_columns(train_df)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feats].values)
    test_scaled = scaler.transform(test_df[feats].values)
    if save_scaler:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        joblib.dump(scaler, config.SCALER_PATH)
        print(f"[preprocessing] Scaler saved → {config.SCALER_PATH}")
    return train_scaled, test_scaled, scaler
