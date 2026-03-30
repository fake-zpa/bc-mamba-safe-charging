"""Mamba backend wrapper around official mamba-ssm package.

This module ONLY wraps the official mamba-ssm implementation.
It does NOT reimplement any core Mamba operators.
"""
import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None


def check_mamba_available() -> bool:
    """Check if official mamba-ssm is available.

    Returns:
        True if mamba-ssm is importable.
    """
    return MAMBA_AVAILABLE


class MambaBlock(nn.Module):
    """Single Mamba block wrapping official mamba-ssm.Mamba.

    This is a thin wrapper that handles:
    - Input/output dimension adaptation
    - Configuration injection
    - Residual connection + LayerNorm
    - Gradient checkpointing compatibility

    Core SSM computation is entirely delegated to official mamba-ssm.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        """Initialize MambaBlock.

        Args:
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_conv: Convolution width.
            expand: Expansion factor.
            dropout: Dropout rate.
            use_checkpoint: Enable gradient checkpointing.

        Raises:
            ImportError: If official mamba-ssm is not installed.
        """
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "Official mamba-ssm package is required but not installed. "
                "Install with: pip install mamba-ssm. "
                "This project does NOT support custom Mamba reimplementations."
            )

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Output tensor of shape (B, L, D).
        """
        residual = x
        x = self.norm(x)

        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self.mamba, x, use_reentrant=False)
        else:
            x = self.mamba(x)

        x = self.dropout(x)
        return x + residual


class MambaStack(nn.Module):
    """Stack of MambaBlocks forming the core sequence model.

    Uses official mamba-ssm for all SSM computations.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        """Initialize MambaStack.

        Args:
            d_model: Model dimension.
            n_layer: Number of Mamba layers.
            d_state: SSM state dimension.
            d_conv: Convolution width.
            expand: Expansion factor.
            dropout: Dropout rate.
            use_checkpoint: Enable gradient checkpointing.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all Mamba layers.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Output tensor of shape (B, L, D).
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
