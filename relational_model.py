"""
Relational Module – Cross-sensor consistency analysis.

For every pair of features whose |correlation| exceeds a threshold in the
training data, we fit a simple linear relationship  f_j ≈ slope·f_i + intercept.
At test time the average squared residual across all selected pairs gives the
per-sample *relational error*.  A high relational error means cross-sensor
relationships have drifted from the training baseline.
"""

import pickle
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import config


# ── Data structures 

@dataclass
class PairModel:
    """Stores a simple linear model for one feature pair."""
    i: int              # index of predictor feature
    j: int              # index of response feature
    slope: float
    intercept: float
    train_residual_std: float   # std of residuals on training data (for z-scoring)


# ── Fitting 

def compute_baseline_correlations(train_data: np.ndarray) -> np.ndarray:
    """Return the full correlation matrix of the training features."""
    return np.corrcoef(train_data, rowvar=False)


def select_correlated_pairs(
    corr_matrix: np.ndarray,
    threshold: float = config.CORR_THRESHOLD,
    max_pairs: int = config.MAX_CORR_PAIRS,
) -> list[tuple[int, int]]:
    """
    Select upper-triangle feature pairs whose |correlation| ≥ threshold.
    Returns at most *max_pairs* pairs (sorted by descending |r|).
    """
    n = corr_matrix.shape[0]
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = abs(corr_matrix[i, j])
            if not np.isfinite(r):
                continue
            if r >= threshold:
                pairs.append((i, j, r))
    # sort by strength, keep top-k
    pairs.sort(key=lambda x: x[2], reverse=True)
    selected = [(i, j) for i, j, _ in pairs[:max_pairs]]
    print(f"[relational] Selected {len(selected)} correlated pairs (threshold={threshold})")
    return selected


def fit_pair_models(
    train_data: np.ndarray,
    pairs: list[tuple[int, int]],
) -> list[PairModel]:
    """Fit a simple linear regression for each selected feature pair."""
    models: list[PairModel] = []
    for i, j in pairs:
        xi = train_data[:, i]
        xj = train_data[:, j]

        if np.std(xi) < 1e-12:
            # Predictor is near-constant: fallback to slope 0 and mean target.
            slope = 0.0
            intercept = float(np.mean(xj))
        else:
            # closed-form OLS:  xj = slope * xi + intercept
            slope, intercept = np.polyfit(xi, xj, deg=1)
            if not (np.isfinite(slope) and np.isfinite(intercept)):
                continue

        residuals = xj - (slope * xi + intercept)
        std = residuals.std() + 1e-8  # avoid division by zero
        models.append(PairModel(i=i, j=j, slope=slope, intercept=intercept, train_residual_std=std))
    return models


def build_relational_module(train_data: np.ndarray) -> tuple[np.ndarray, list[PairModel]]:
    """
    End-to-end: compute baseline correlations, select pairs, fit models.
    Returns (baseline_corr_matrix, list_of_PairModel).
    """
    corr = compute_baseline_correlations(train_data)
    pairs = select_correlated_pairs(corr)
    models = fit_pair_models(train_data, pairs)
    return corr, models


def save_relational_artifacts(
    baseline_corr: np.ndarray,
    pair_models: list[PairModel],
) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(config.BASELINE_CORR_PATH, baseline_corr)
    with open(config.RELATION_MODELS_PATH, "wb") as f:
        pickle.dump(pair_models, f)
    print(f"[relational] Artifacts saved → {config.OUTPUT_DIR}")


def load_relational_artifacts() -> tuple[np.ndarray, list[PairModel]]:
    baseline_corr = np.load(config.BASELINE_CORR_PATH)
    with open(config.RELATION_MODELS_PATH, "rb") as f:
        pair_models = pickle.load(f)
    return baseline_corr, pair_models


# ── Per-sample relational error 

def relational_error_single(sample: np.ndarray, pair_models: list[PairModel]) -> float:
    """
    For one sample (1-D array of shape [n_features]), compute the mean
    squared z-scored residual across all pair models.
    """
    if not pair_models:
        return 0.0
    total = 0.0
    for pm in pair_models:
        predicted_j = pm.slope * sample[pm.i] + pm.intercept
        residual = abs(sample[pm.j] - predicted_j) / pm.train_residual_std
        total += residual ** 2
    return total / len(pair_models)


def compute_relational_errors(
    data: np.ndarray,
    pair_models: list[PairModel],
    desc: str = "Relational errors",
) -> np.ndarray:
    """Vectorised relational error for an entire dataset (N × D)."""
    errors = np.empty(data.shape[0], dtype=np.float64)
    for idx in tqdm(range(data.shape[0]), desc=desc, leave=False):
        errors[idx] = relational_error_single(data[idx], pair_models)
    return errors