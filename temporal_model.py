"""
Temporal Module – Fully-connected autoencoder for feature-level reconstruction.
Learns "normal" feature patterns from training batches; reconstruction error
on test batches signals feature-level anomalies.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Symmetric FC autoencoder:
        input_dim → hidden → latent → hidden → input_dim
    Includes BatchNorm and Dropout for robust training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (1-D tensor of length N)."""
        self.eval()
        recon = self.forward(x)
        mse = ((recon - x) ** 2).mean(dim=1)
        return mse