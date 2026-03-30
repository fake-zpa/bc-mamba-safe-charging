"""Safety layer for action projection in battery charging control.

Projects actor actions to satisfy safety constraints based on
risk predictions and world model uncertainty.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class SafetyLayer(nn.Module):
    """Project actions to satisfy safety constraints.

    Supports multiple modes:
    - none: no safety projection (passthrough)
    - reward_penalty: add penalty to reward (no action modification)
    - hard_projection: clip action based on risk threshold
    - uncertainty_aware: adaptive projection using world model uncertainty
    """

    def __init__(
        self,
        mode: str = "uncertainty_aware",
        risk_threshold: float = 0.5,
        max_action: float = 6.0,
        uncertainty_scale: float = 0.1,
        projection_lr: float = 0.01,
    ):
        """Initialize SafetyLayer.

        Args:
            mode: Safety mode ('none', 'reward_penalty', 'hard_projection', 'uncertainty_aware').
            risk_threshold: Risk threshold for triggering projection.
            max_action: Maximum action value.
            uncertainty_scale: Scale factor for uncertainty in projection.
            projection_lr: Learning rate for iterative projection.
        """
        super().__init__()
        self.mode = mode
        self.risk_threshold = risk_threshold
        self.max_action = max_action
        self.uncertainty_scale = uncertainty_scale
        self.projection_lr = projection_lr

    def forward(
        self,
        action: torch.Tensor,
        risk_info: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project action through safety layer.

        Args:
            action: Raw actor action (B, action_dim).
            risk_info: Risk prediction dict from RiskHead.
            uncertainty: World model uncertainty (B, 1).

        Returns:
            Safe action (B, action_dim).
        """
        if self.mode == "none":
            return action

        if self.mode == "reward_penalty":
            # No action modification; penalty is applied in loss
            return action

        if self.mode == "hard_projection":
            return self._hard_projection(action, risk_info)

        if self.mode == "uncertainty_aware":
            return self._uncertainty_aware_projection(action, risk_info, uncertainty)

        return action

    def _hard_projection(
        self,
        action: torch.Tensor,
        risk_info: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Hard projection: reduce action when risk exceeds threshold.

        Args:
            action: Raw action (B, action_dim).
            risk_info: Risk predictions.

        Returns:
            Projected action.
        """
        overall_risk = risk_info["overall_risk"]  # (B, 1)
        # Scale down action proportionally when risk is high
        risk_exceeded = (overall_risk > self.risk_threshold).float()
        scale = torch.where(
            risk_exceeded.bool(),
            self.risk_threshold / (overall_risk + 1e-6),
            torch.ones_like(overall_risk),
        )
        safe_action = action * scale.clamp(0.0, 1.0)
        return safe_action.clamp(0.0, self.max_action)

    def _uncertainty_aware_projection(
        self,
        action: torch.Tensor,
        risk_info: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Uncertainty-aware projection: combine risk and model uncertainty.

        Uses smooth sigmoid scaling instead of hard cutoff to avoid
        over-conservative action suppression.

        Args:
            action: Raw action (B, action_dim).
            risk_info: Risk predictions.
            uncertainty: Model uncertainty (B, 1).

        Returns:
            Projected action.
        """
        overall_risk = risk_info["overall_risk"]  # (B, 1)

        if uncertainty is not None:
            combined_risk = overall_risk + self.uncertainty_scale * uncertainty
        else:
            combined_risk = overall_risk

        # Smooth sigmoid scaling: gradually reduce action as risk increases
        # scale = sigmoid(-(combined_risk - threshold) * sharpness)
        # At risk=0: scale≈1.0, at risk=threshold: scale≈0.5, at risk>>threshold: scale→0
        sharpness = 5.0
        scale = torch.sigmoid(-(combined_risk - self.risk_threshold) * sharpness)
        # Ensure minimum scale of 0.5 (never suppress action below 50%)
        scale = scale * 0.5 + 0.5

        safe_action = action * scale
        return safe_action.clamp(0.0, self.max_action)

    def compute_penalty(
        self,
        risk_info: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward penalty for reward_penalty mode.

        Args:
            risk_info: Risk predictions.
            uncertainty: Model uncertainty.

        Returns:
            Penalty scalar (B, 1).
        """
        overall_risk = risk_info["overall_risk"]
        penalty = torch.relu(overall_risk - self.risk_threshold)
        if uncertainty is not None:
            penalty = penalty + self.uncertainty_scale * uncertainty
        return penalty
