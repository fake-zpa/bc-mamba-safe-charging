"""Behavioral Cloning (BC) baseline for offline RL.

Simple supervised learning of actions from expert demonstrations.
Uses shared Mamba encoder for latent state representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BCPolicy(nn.Module):
    """Behavioral Cloning policy that maps latent state to action.

    Uses shared encoder's latent state z_t as input.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_action: float = 6.0,
    ):
        """Initialize BC policy.

        Args:
            latent_dim: Input latent dimension from encoder.
            action_dim: Output action dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of hidden layers.
            dropout: Dropout rate.
            max_action: Maximum action magnitude.
        """
        super().__init__()
        self.max_action = max_action

        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict action from latent state.

        Args:
            z: Latent state (B, latent_dim).

        Returns:
            Action (B, action_dim) scaled to [0, max_action].
        """
        raw = self.net(z)
        # Scale to [0, max_action] for charging current
        return torch.sigmoid(raw) * self.max_action

    def loss(
        self,
        z: torch.Tensor,
        target_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute BC loss (MSE between predicted and target actions).

        Args:
            z: Latent state (B, latent_dim).
            target_action: Target action (B, action_dim).

        Returns:
            Dict with 'bc_loss'.
        """
        pred = self.forward(z)
        return {"bc_loss": F.mse_loss(pred, target_action)}

    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        """Get action for deployment (no grad).

        Args:
            z: Latent state (B, latent_dim).

        Returns:
            Action (B, action_dim).
        """
        with torch.no_grad():
            return self.forward(z)
