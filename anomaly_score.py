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


def _robust_minmax(
    arr: np.ndarray,
    low_q: float = config.ROBUST_NORM_LOW_Q,
    high_q: float = config.ROBUST_NORM_HIGH_Q,
) -> np.ndarray:
    """
    Robust min-max using percentile clipping to reduce outlier dominance.
    Values are clipped to [low_q, high_q] before scaling to [0, 1].
    """
    lo = np.percentile(arr, low_q)
    hi = np.percentile(arr, high_q)
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo)


def _normalize(arr: np.ndarray, method: str = config.SCORE_NORM_METHOD) -> np.ndarray:
    if method == "minmax":
        return _minmax(arr)
    if method == "robust":
        return _robust_minmax(arr)
    raise ValueError(f"Unknown normalization method: {method}")


def compute_combined_scores(
    temporal_errors: np.ndarray,
    relational_errors: np.ndarray,
    alpha: float = config.ALPHA,
    beta: float = config.BETA,
    norm_method: str = config.SCORE_NORM_METHOD,
) -> np.ndarray:
    """
    Combine temporal and relational errors into a single anomaly score
    per sample.  Both inputs are 1-D arrays of length N.
    """
    t_norm = _normalize(temporal_errors, method=norm_method)
    r_norm = _normalize(relational_errors, method=norm_method)
    return alpha * t_norm + beta * r_norm


def compute_threshold(
    scores: np.ndarray,
    sigma: float = config.ANOMALY_SIGMA,
    method: str = config.THRESHOLD_METHOD,
    percentile: float = config.ANOMALY_PERCENTILE,
) -> float:
    """Compute threshold using either mean+sigma*std or percentile."""
    if method == "mean_std":
        return float(scores.mean() + sigma * scores.std())
    if method == "percentile":
        return float(np.percentile(scores, percentile))
    raise ValueError(f"Unknown threshold method: {method}")


def flag_anomalies(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean array – True where sample is anomalous."""
    return scores > threshold
