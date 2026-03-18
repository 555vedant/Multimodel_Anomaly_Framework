"""
Evaluation / Inference script – loads trained artifacts, scores all test
samples, detects anomalies, and produces visualisations.

Usage:
    python evaluate.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import config
from preprocessing import load_all_batches, split_train_test, scale_data
from temporal_model import Autoencoder
from relational_model import load_relational_artifacts, compute_relational_errors
from anomaly_score import compute_combined_scores, compute_threshold, flag_anomalies


def load_trained_model(input_dim: int) -> Autoencoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT,
    )
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_score_distribution(scores: np.ndarray, threshold: float, save_path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(scores, bins=80, alpha=0.7, color="steelblue", edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Anomaly Score Distribution (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved: {save_path}")


def plot_scores_by_batch(
    scores: np.ndarray,
    batch_ids: np.ndarray,
    threshold: float,
    save_path: str,
) -> None:
    plt.figure(figsize=(12, 5))
    unique_batches = sorted(np.unique(batch_ids))
    positions, data = [], []
    for b in unique_batches:
        data.append(scores[batch_ids == b])
        positions.append(b)
    bp = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold")
    plt.xlabel("Batch")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Scores by Batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved: {save_path}")


def plot_error_components(
    temporal: np.ndarray,
    relational: np.ndarray,
    anomalies: np.ndarray,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 8))
    normal = ~anomalies
    plt.scatter(temporal[normal], relational[normal], s=4, alpha=0.3, label="Normal", color="steelblue")
    if anomalies.any():
        plt.scatter(temporal[anomalies], relational[anomalies], s=12, alpha=0.7, label="Anomaly", color="red")
    plt.xlabel("Temporal Error (normalised)")
    plt.ylabel("Relational Error (normalised)")
    plt.title("Temporal vs Relational Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved: {save_path}")


def plot_anomaly_rate_by_batch(
    anomalies: np.ndarray,
    batch_ids: np.ndarray,
    save_path: str,
) -> None:
    unique_batches = sorted(np.unique(batch_ids))
    rates = [anomalies[batch_ids == b].mean() * 100 for b in unique_batches]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(unique_batches, rates, color="salmon", edgecolor="white")
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xlabel("Batch")
    plt.ylabel("Anomaly Rate (%)")
    plt.title("Anomaly Rate per Batch (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved: {save_path}")


# ── Main evaluation ──────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  EVALUATION – Drift-Aware Anomaly Detection")
    print("=" * 60)

    # ── 1. Load data 
    df = load_all_batches()
    train_df, test_df = split_train_test(df)
    train_scaled, test_scaled, scaler = scale_data(train_df, test_df, save_scaler=False)

    input_dim = train_scaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 2. Load trained artifacts 
    model = load_trained_model(input_dim)
    _, pair_models = load_relational_artifacts()

    # ── 3. Temporal errors 
    print("\n── Computing Temporal Errors ──")
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32).to(device)
    temporal_errors = model.reconstruction_error(test_tensor).cpu().numpy()
    print(f"  Temporal error  →  mean={temporal_errors.mean():.6f}  std={temporal_errors.std():.6f} Error : {temporal_errors}")

    # ── 4. relational errors 
    print("\n── Computing Relational Errors ──")
    relational_errors = compute_relational_errors(test_scaled, pair_models, desc="Test relational errors")
    print(f"  Relational error →  mean={relational_errors.mean():.6f}  std={relational_errors.std():.6f} Error : {relational_errors}")

    # ── 5. Combined scores 
    scores = compute_combined_scores(temporal_errors, relational_errors)

    # Compute threshold on training scores for a principled cutoff
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32).to(device)
    train_temporal = model.reconstruction_error(train_tensor).cpu().numpy()
    train_relational = compute_relational_errors(train_scaled, pair_models, desc="Train relational errors")
    train_scores = compute_combined_scores(train_temporal, train_relational)
    print(f"Combined score is {train_scores}")
    threshold = compute_threshold(train_scores, sigma=config.ANOMALY_SIGMA)

    anomalies = flag_anomalies(scores, threshold)

    # ── 6. Report
    batch_ids = test_df["batch"].values
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Threshold          : {threshold:.6f}")
    print(f"  Total test samples : {len(scores)}")
    print(f"  Anomalies detected : {anomalies.sum()}")
    print(f"  Anomaly rate       : {anomalies.mean() * 100:.2f}%")
    print()

    for b in sorted(np.unique(batch_ids)):
        mask = batch_ids == b
        n_total = mask.sum()
        n_anom = anomalies[mask].sum()
        rate = anomalies[mask].mean() * 100
        print(f"  Batch {b:>2}: {n_anom:>5} / {n_total:>5} anomalies  ({rate:.1f}%)")

    # ── 7. Visualisations
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # plot_score_distribution(scores, threshold, os.path.join(fig_dir, "score_distribution.png"))
    plot_scores_by_batch(scores, batch_ids, threshold, os.path.join(fig_dir, "scores_by_batch.png"))
    plot_error_components(
        temporal_errors / (temporal_errors.max() + 1e-12),
        relational_errors / (relational_errors.max() + 1e-12),
        anomalies,
        os.path.join(fig_dir, "temporal_vs_relational.png"),
    )
    plot_anomaly_rate_by_batch(anomalies, batch_ids, os.path.join(fig_dir, "anomaly_rate_by_batch.png"))

    # save raw scores
    np.savez(
        os.path.join(config.OUTPUT_DIR, "results.npz"),
        scores=scores,  
        temporal_errors=temporal_errors,
        relational_errors=relational_errors,
        anomalies=anomalies,
        batch_ids=batch_ids,
        threshold=threshold,
    )   
    print(f"\n[evaluate] Raw results saved → {os.path.join(config.OUTPUT_DIR, 'results.npz')}")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()