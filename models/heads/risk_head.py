"""Risk prediction head for safety-aware battery charging control.

Predicts constraint violation probabilities from latent state and action.
"""
import torch
import torch.nn as nn
from typing import Dict


class RiskHead(nn.Module):
    """Predict constraint violation risks from latent state and action.

    Outputs per-constraint violation probabilities and overall risk score.
    Constraints: voltage, temperature, dT/dt, plating risk.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_constraints: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize RiskHead.

        Args:
            latent_dim: Input latent dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of hidden layers.
            n_constraints: Number of constraints.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_constraints = n_constraints

        layers = []
        in_dim = latent_dim + action_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Per-constraint violation probability
        self.constraint_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            for _ in range(n_constraints)
        ])

        # Overall risk score
        self.overall_risk = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict risk scores.

        Args:
            z: Latent state (B, latent_dim).
            action: Action (B, action_dim).

        Returns:
            Dictionary with per-constraint and overall risk scores.
        """
        x = torch.cat([z, action], dim=-1)
        h = self.backbone(x)

        constraint_probs = []
        for head in self.constraint_heads:
            constraint_probs.append(head(h))

        constraint_probs_cat = torch.cat(constraint_probs, dim=-1)  # (B, n_constraints)
        overall = self.overall_risk(h)  # (B, 1)

        return {
            "voltage_risk": constraint_probs[0],
            "temperature_risk": constraint_probs[1],
            "dT_dt_risk": constraint_probs[2],
            "plating_risk": constraint_probs[3] if self.n_constraints > 3 else torch.zeros_like(overall),
            "constraint_probs": constraint_probs_cat,
            "overall_risk": overall,
        }

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        violation_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute risk prediction loss (MSE, AMP-safe alternative to BCE).

        Args:
            predictions: Risk predictions dict.
            violation_labels: Binary violation labels (B, n_constraints).

        Returns:
            Scalar loss tensor.
        """
        pred = predictions["constraint_probs"]
        return nn.functional.mse_loss(pred, violation_labels)
