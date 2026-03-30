"""Mamba-based latent health state encoder for battery time-series.

This encoder uses official mamba-ssm to map windowed battery observations
to a latent health state z_t that is shared across actor, critic,
world model, and risk head.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .mamba_backend import MambaStack, check_mamba_available


class MambaHealthEncoder(nn.Module):
    """Single-stream Mamba encoder for battery health state.

    Maps (B, L, obs_dim) -> (B, latent_dim) latent health state.
    Uses official mamba-ssm as the core sequence model.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        d_model: int = 128,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        """Initialize MambaHealthEncoder.

        Args:
            obs_dim: Input observation dimension per timestep.
            d_model: Mamba model dimension.
            n_layer: Number of Mamba layers.
            d_state: SSM state dimension.
            d_conv: Convolution width.
            expand: Expansion factor.
            latent_dim: Output latent state dimension.
            dropout: Dropout rate.
            use_checkpoint: Enable gradient checkpointing.
        """
        super().__init__()
        assert check_mamba_available(), (
            "Official mamba-ssm is required for MambaHealthEncoder. "
            "This project does not support custom Mamba reimplementations."
        )

        self.obs_dim = obs_dim
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Mamba stack (official mamba-ssm)
        self.mamba = MambaStack(
            d_model=d_model,
            n_layer=n_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # Latent projection: take last hidden state -> latent_dim
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Encode windowed observation to latent health state.

        Args:
            obs: Input tensor of shape (B, L, obs_dim).
            return_sequence: If True, return full sequence (B, L, latent_dim).

        Returns:
            Latent state z_t of shape (B, latent_dim) or (B, L, latent_dim).
        """
        # Project input to d_model
        x = self.input_proj(obs)  # (B, L, d_model)

        # Mamba sequence encoding
        h = self.mamba(x)  # (B, L, d_model)

        if return_sequence:
            return self.latent_proj(h)  # (B, L, latent_dim)

        # Take last timestep as latent state
        z = self.latent_proj(h[:, -1, :])  # (B, latent_dim)
        return z

    def encode_for_pretrain(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode for pretraining: return both sequence and final latent.

        Args:
            obs: Input tensor of shape (B, L, obs_dim).

        Returns:
            Tuple of (z_t: (B, latent_dim), h_seq: (B, L, d_model)).
        """
        x = self.input_proj(obs)
        h = self.mamba(x)
        z = self.latent_proj(h[:, -1, :])
        return z, h


class GRUHealthEncoder(nn.Module):
    """GRU-based encoder baseline for ablation comparison."""

    def __init__(
        self,
        obs_dim: int = 10,
        hidden_dim: int = 128,
        n_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        """Initialize GRU encoder.

        Args:
            obs_dim: Input observation dimension.
            hidden_dim: GRU hidden dimension.
            n_layers: Number of GRU layers.
            latent_dim: Output latent dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.gru = nn.GRU(
            obs_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Encode observation sequence.

        Args:
            obs: Input (B, L, obs_dim).
            return_sequence: Return full sequence output.

        Returns:
            Latent state (B, latent_dim) or (B, L, latent_dim).
        """
        h, _ = self.gru(obs)
        if return_sequence:
            return self.latent_proj(h)
        return self.latent_proj(h[:, -1, :])


class TransformerHealthEncoder(nn.Module):
    """Transformer-based encoder baseline for ablation comparison."""

    def __init__(
        self,
        obs_dim: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        """Initialize Transformer encoder.

        Args:
            obs_dim: Input observation dimension.
            d_model: Transformer model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            latent_dim: Output latent dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Encode observation sequence.

        Args:
            obs: Input (B, L, obs_dim).
            return_sequence: Return full sequence output.

        Returns:
            Latent state (B, latent_dim) or (B, L, latent_dim).
        """
        B, L, _ = obs.shape
        x = self.input_proj(obs) + self.pos_embed[:, :L, :]
        h = self.transformer(x)
        if return_sequence:
            return self.latent_proj(h)
        return self.latent_proj(h[:, -1, :])


class NoHistoryEncoder(nn.Module):
    """Baseline: no history encoding, just project current obs."""

    def __init__(self, obs_dim: int = 10, latent_dim: int = 64):
        """Initialize no-history encoder.

        Args:
            obs_dim: Input observation dimension.
            latent_dim: Output latent dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Encode only the last observation.

        Args:
            obs: Input (B, L, obs_dim).
            return_sequence: Return full sequence output.

        Returns:
            Latent state (B, latent_dim) or (B, L, latent_dim).
        """
        if return_sequence:
            return self.proj(obs)
        return self.proj(obs[:, -1, :])


def build_encoder(cfg: Dict) -> nn.Module:
    """Build encoder from configuration.

    Args:
        cfg: Encoder configuration dictionary.

    Returns:
        Encoder module.
    """
    enc_cfg = cfg.get("encoder", cfg)
    enc_type = enc_cfg.get("type", "mamba")

    if enc_type == "mamba":
        return MambaHealthEncoder(
            obs_dim=enc_cfg.get("obs_dim", 10),
            d_model=enc_cfg.get("d_model", 128),
            n_layer=enc_cfg.get("n_layer", 4),
            d_state=enc_cfg.get("d_state", 16),
            d_conv=enc_cfg.get("d_conv", 4),
            expand=enc_cfg.get("expand", 2),
            latent_dim=enc_cfg.get("latent_dim", 64),
            dropout=enc_cfg.get("dropout", 0.1),
            use_checkpoint=enc_cfg.get("use_checkpoint", False),
        )
    elif enc_type == "gru":
        gru_cfg = cfg.get("gru_encoder", enc_cfg)
        return GRUHealthEncoder(
            obs_dim=enc_cfg.get("obs_dim", 10),
            hidden_dim=gru_cfg.get("hidden_dim", 128),
            n_layers=gru_cfg.get("n_layers", 2),
            latent_dim=enc_cfg.get("latent_dim", 64),
            dropout=gru_cfg.get("dropout", 0.1),
        )
    elif enc_type == "transformer":
        tf_cfg = cfg.get("transformer_encoder", enc_cfg)
        return TransformerHealthEncoder(
            obs_dim=enc_cfg.get("obs_dim", 10),
            d_model=tf_cfg.get("d_model", 128),
            n_heads=tf_cfg.get("n_heads", 4),
            n_layers=tf_cfg.get("n_layers", 2),
            latent_dim=enc_cfg.get("latent_dim", 64),
            dropout=tf_cfg.get("dropout", 0.1),
        )
    elif enc_type == "none":
        return NoHistoryEncoder(
            obs_dim=enc_cfg.get("obs_dim", 10),
            latent_dim=enc_cfg.get("latent_dim", 64),
        )
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
