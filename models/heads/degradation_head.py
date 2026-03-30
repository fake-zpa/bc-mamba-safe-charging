"""Degradation prediction head operating on latent health state z_t.

Predicts battery health indicators from the shared Mamba latent state.
"""
import torch
import torch.nn as nn
from typing import Dict


class DegradationHead(nn.Module):
    """Predict degradation indicators from latent health state.

    Outputs:
        - soh_proxy: State of health proxy [0, 1]
        - capacity_fade_proxy: Capacity fade rate
        - resistance_growth_proxy: Resistance growth rate
        - cycle_life_proxy: Remaining useful life proxy
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize DegradationHead.

        Args:
            latent_dim: Input latent dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of hidden layers.
            dropout: Dropout rate.
        """
        super().__init__()
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

        self.backbone = nn.Sequential(*layers)
        self.soh_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.capacity_fade_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.resistance_growth_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.cycle_life_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict degradation indicators.

        Args:
            z: Latent health state (B, latent_dim).

        Returns:
            Dictionary of predictions, each (B, 1).
        """
        h = self.backbone(z)
        return {
            "soh_proxy": self.soh_head(h),
            "capacity_fade_proxy": self.capacity_fade_head(h),
            "resistance_growth_proxy": self.resistance_growth_head(h),
            "cycle_life_proxy": self.cycle_life_head(h),
        }

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute degradation prediction loss.

        Args:
            predictions: Model predictions dict.
            targets: Target values dict.

        Returns:
            Scalar loss tensor.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for key in predictions:
            if key in targets:
                total = total + nn.functional.mse_loss(predictions[key], targets[key])
        return total
