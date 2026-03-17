"""
Unified anomaly scoring module.

    S = α · E_temporal  +  β · E_relational

Both error components are min-max normalised to [0, 1] before combining so
that α and β directly control the relative importance regardless of scale.
"""

import numpy as np
import config


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def compute_combined_scores(
    temporal_errors: np.ndarray,
    relational_errors: np.ndarray,
    alpha: float = config.ALPHA,
    beta: float = config.BETA,
) -> np.ndarray:
    """
    Combine temporal and relational errors into a single anomaly score
    per sample.  Both inputs are 1-D arrays of length N.
    """
    t_norm = _minmax(temporal_errors)
    r_norm = _minmax(relational_errors)
    return alpha * t_norm + beta * r_norm


def compute_threshold(
    scores: np.ndarray,
    sigma: float = config.ANOMALY_SIGMA,
) -> float:
    """Threshold = mean + sigma × std of training/reference scores."""
    return float(scores.mean() + sigma * scores.std())


def flag_anomalies(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean array – True where sample is anomalous."""
    return scores > threshold
