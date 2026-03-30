"""Dual-stream Mamba encoder for battery health state.

Processes voltage/current and thermal/degradation streams separately
before fusing into a unified latent health state.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .mamba_backend import MambaStack, check_mamba_available


class DualStreamMambaEncoder(nn.Module):
    """Dual-stream Mamba encoder with separate streams for:
    - Stream A: voltage, current, SOC, dV/dt (electrochemical)
    - Stream B: temperature, dT/dt, resistance, degradation (thermal/health)

    Both streams use official mamba-ssm and fuse into a shared latent state.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        d_model_a: int = 64,
        d_model_b: int = 64,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        fusion: str = "concat",
        use_checkpoint: bool = False,
    ):
        """Initialize DualStreamMambaEncoder.

        Args:
            obs_dim: Total observation dimension.
            d_model_a: Stream A model dimension.
            d_model_b: Stream B model dimension.
            n_layer: Number of Mamba layers per stream.
            d_state: SSM state dimension.
            d_conv: Convolution width.
            expand: Expansion factor.
            latent_dim: Output latent dimension.
            dropout: Dropout rate.
            fusion: Fusion strategy ('concat', 'attention', 'gate').
            use_checkpoint: Enable gradient checkpointing.
        """
        super().__init__()
        assert check_mamba_available(), "Official mamba-ssm required."

        self.fusion_type = fusion
        self.latent_dim = latent_dim

        # Stream A: electrochemical [voltage, current, soc, dV/dt, cycle, charge_throughput]
        self.stream_a_indices = [0, 1, 3, 4, 6, 7]  # 6 features
        self.stream_a_proj = nn.Sequential(
            nn.Linear(len(self.stream_a_indices), d_model_a),
            nn.LayerNorm(d_model_a),
            nn.GELU(),
        )
        self.stream_a_mamba = MambaStack(
            d_model=d_model_a, n_layer=n_layer, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # Stream B: thermal/health [temperature, dT/dt, resistance, degradation]
        self.stream_b_indices = [2, 5, 8, 9]  # 4 features
        self.stream_b_proj = nn.Sequential(
            nn.Linear(len(self.stream_b_indices), d_model_b),
            nn.LayerNorm(d_model_b),
            nn.GELU(),
        )
        self.stream_b_mamba = MambaStack(
            d_model=d_model_b, n_layer=n_layer, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # Fusion layer
        fused_dim = d_model_a + d_model_b
        if fusion == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(fused_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.Tanh(),
            )
        elif fusion == "gate":
            self.gate_a = nn.Linear(fused_dim, d_model_a)
            self.gate_b = nn.Linear(fused_dim, d_model_b)
            self.fuse = nn.Sequential(
                nn.Linear(fused_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.Tanh(),
            )
        elif fusion == "attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model_a, num_heads=4, batch_first=True, dropout=dropout,
            )
            self.fuse = nn.Sequential(
                nn.Linear(d_model_a + d_model_b, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

    def forward(
        self, obs: torch.Tensor, return_sequence: bool = False,
    ) -> torch.Tensor:
        """Encode windowed observation through dual streams.

        Args:
            obs: Input (B, L, obs_dim).
            return_sequence: Return full sequence (B, L, latent_dim).

        Returns:
            Latent state (B, latent_dim) or (B, L, latent_dim).
        """
        # Split streams
        x_a = obs[:, :, self.stream_a_indices]
        x_b = obs[:, :, self.stream_b_indices]

        # Encode each stream
        h_a = self.stream_a_mamba(self.stream_a_proj(x_a))  # (B, L, d_model_a)
        h_b = self.stream_b_mamba(self.stream_b_proj(x_b))  # (B, L, d_model_b)

        # Fuse
        if self.fusion_type == "gate":
            cat = torch.cat([h_a, h_b], dim=-1)
            g_a = torch.sigmoid(self.gate_a(cat))
            g_b = torch.sigmoid(self.gate_b(cat))
            fused = torch.cat([h_a * g_a, h_b * g_b], dim=-1)
        elif self.fusion_type == "attention":
            h_a_attn, _ = self.cross_attn(h_a, h_b, h_b)
            fused = torch.cat([h_a_attn, h_b], dim=-1)
        else:  # concat
            fused = torch.cat([h_a, h_b], dim=-1)

        if return_sequence:
            return self.fuse(fused)

        return self.fuse(fused[:, -1, :])
