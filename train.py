"""
Training script – trains both the Temporal and Relational modules and
persists all artifacts to the outputs/ directory.

Usage:
    python train.py
    python train.py --config configs/exp_baseline.json
"""

import os
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from preprocessing import load_all_batches, split_train_test, scale_data
from temporal_model import Autoencoder
from relational_model import build_relational_module, save_relational_artifacts


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_autoencoder(
    train_data: np.ndarray,
    input_dim: int,
) -> tuple[Autoencoder, float, int]:
    """Train the temporal autoencoder with mini-batches and early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{config.EPOCHS}  loss={epoch_loss:.6f}  lr={current_lr:.2e}")

        # early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_epoch = epoch
            patience_counter = 0
            # save best weights
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best loss={best_loss:.6f})")
                break

    # reload best weights
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    print(f"[train] Best model saved -> {config.MODEL_PATH}")
    print(f"[train] Best loss={best_loss:.6f} at epoch {best_epoch}")
    return model, best_loss, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train drift-aware anomaly model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config overrides (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        config.apply_overrides_from_json(args.config)

    seed_everything()

    print("\n" + "=" * 70)
    print(f"TRAINING | experiment={config.EXPERIMENT_NAME} | output={config.OUTPUT_DIR}")
    print("=" * 70)
    df = load_all_batches()
    train_df, test_df = split_train_test(df)
    train_scaled, _, _ = scale_data(train_df, test_df)

    input_dim = train_scaled.shape[1]
    print(f"[train] Feature dimension: {input_dim}")

    # ── Temporal module ──────────────────────────────────────────────────
    print("\n── Training Temporal Autoencoder ──")
    _, best_loss, best_epoch = train_autoencoder(train_scaled, input_dim)

    # ── Relational module ────────────────────────────────────────────────
    print("\n── Building Relational Module ──")
    baseline_corr, pair_models = build_relational_module(train_scaled)
    save_relational_artifacts(baseline_corr, pair_models)

    summary = {
        "experiment": config.EXPERIMENT_NAME,
        "output_dir": config.OUTPUT_DIR,
        "best_train_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "num_pair_models": int(len(pair_models)),
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(config.OUTPUT_DIR, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Saved summary -> {summary_path}")

    print("\n[train] Training complete")


if __name__ == "__main__":
    main()