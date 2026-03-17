"""
Training script – trains both the Temporal and Relational modules and
persists all artifacts to the outputs/ directory.

Usage:
    python train.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
) -> Autoencoder:
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
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            # print("check1")
            recon = model(batch_x)
            # print("check2")

            loss = criterion(recon, batch_x)
            # print("check3")

            loss.backward()
            # print("check4")

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
    print(f"[train] Autoencoder saved → {config.MODEL_PATH}")
    return model


def main() -> None:
    seed_everything()

    # ── Data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  TRAINING – Drift-Aware Anomaly Detection")
    print("=" * 60)
    df = load_all_batches()
    train_df, test_df = split_train_test(df)
    train_scaled, test_scaled, scaler = scale_data(train_df, test_df)

    input_dim = train_scaled.shape[1]
    print(f"[train] Feature dimension: {input_dim}")

    # ── Temporal module ──────────────────────────────────────────────────
    print("\n── Training Temporal Autoencoder ──")
    model = train_autoencoder(train_scaled, input_dim)

    # ── Relational module ────────────────────────────────────────────────
    print("\n── Building Relational Module ──")
    baseline_corr, pair_models = build_relational_module(train_scaled)
    save_relational_artifacts(baseline_corr, pair_models)

    print("\n✓ Training complete. All artifacts saved to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()