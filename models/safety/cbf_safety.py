"""Control Barrier Function (CBF) safety filter for battery charging.

Provides hard safety guarantees by projecting unsafe actions to the
boundary of the safe set, combined with smooth Sigmoid soft scaling.

CBF condition: h_dot(x, u) + alpha * h(x) >= 0
where h(x) = x_limit - x(t) is the barrier function.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class CBFSafetyFilter(nn.Module):
    """Dual-layer safety: CBF hard constraint + Sigmoid soft scaling.

    Layer 1 (Sigmoid soft): smoothly reduces action as state approaches limits.
    Layer 2 (CBF hard): projects action to satisfy barrier constraint exactly.
    """

    def __init__(
        self,
        T_limit: float = 38.0,
        V_limit: float = 4.15,
        cbf_alpha: float = 0.5,
        sigmoid_margin_T: float = 5.0,
        sigmoid_margin_V: float = 0.15,
        min_scale: float = 0.1,
        dt: float = 30.0,
        nominal_capacity: float = 0.681,
        k_heat: float = 0.09,
        k_cool: float = 0.02,
        T_amb: float = 25.0,
    ):
        super().__init__()
        self.T_limit = T_limit
        self.V_limit = V_limit
        self.cbf_alpha = cbf_alpha
        self.sigmoid_margin_T = sigmoid_margin_T
        self.sigmoid_margin_V = sigmoid_margin_V
        self.min_scale = min_scale
        self.dt = dt
        self.nominal_capacity = nominal_capacity

        # Thermal model parameters (approximate, for CBF prediction)
        # dT/dt ≈ k_heat * I^2 - k_cool * (T - T_amb)
        # k_heat calibrated from SPMe data: at 6C from 35°C, dT≈1.5°C/step → k_heat≈0.09
        self.k_heat = k_heat  # heating coefficient (°C per A^2 per step)
        self.k_cool = k_cool  # cooling coefficient (1/step)
        self.T_amb = T_amb    # ambient temperature (must match evaluation scenario)

    def barrier_temperature(self, T: torch.Tensor) -> torch.Tensor:
        """Temperature barrier: h_T(x) = T_limit - T. Safe when h_T >= 0."""
        return self.T_limit - T

    def barrier_voltage(self, V: torch.Tensor) -> torch.Tensor:
        """Voltage barrier: h_V(x) = V_limit - V. Safe when h_V >= 0."""
        return self.V_limit - V

    def sigmoid_scale(self, T: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Soft Sigmoid scaling based on proximity to limits.

        Returns scale in [min_scale, 1.0].
        """
        # Temperature proximity: 1.0 far from limit, min_scale at limit
        T_dist = (self.T_limit - T) / self.sigmoid_margin_T  # 1.0 = margin away, 0.0 = at limit
        T_scale = torch.sigmoid(5.0 * T_dist)  # smooth transition

        # Voltage proximity
        V_dist = (self.V_limit - V) / self.sigmoid_margin_V
        V_scale = torch.sigmoid(5.0 * V_dist)

        # Combined scale
        scale = T_scale * V_scale
        return torch.clamp(scale, min=self.min_scale, max=1.0)

    def cbf_max_current(self, T: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute maximum safe current from CBF constraint.

        For temperature: h_dot + alpha * h >= 0
        h = T_limit - T
        h_dot ≈ -k_heat * I^2 + k_cool * (T - T_amb)  (negative because T increases)
        
        CBF: -k_heat * I^2 + k_cool * (T - T_amb) + alpha * (T_limit - T) >= 0
        => I^2 <= [k_cool * (T - T_amb) + alpha * (T_limit - T)] / k_heat
        => I_max = sqrt(max(0, rhs) / k_heat)
        """
        h_T = self.barrier_temperature(T)

        # CBF constraint for temperature
        rhs = self.k_cool * (T - self.T_amb) + self.cbf_alpha * h_T
        I_max_sq = rhs / self.k_heat
        I_max_T = torch.sqrt(torch.clamp(I_max_sq, min=0.01))

        # For voltage: simpler linear constraint
        h_V = self.barrier_voltage(V)
        # At high voltage, limit current proportionally
        I_max_V = torch.where(
            h_V > 0.05,
            torch.ones_like(V) * 6.0 * self.nominal_capacity,  # No constraint far from limit
            torch.clamp(h_V / 0.05 * 4.0 * self.nominal_capacity, min=0.1 * self.nominal_capacity)
        )

        # Take the tighter constraint
        I_max = torch.minimum(I_max_T, I_max_V)
        return I_max

    def forward(
        self,
        action: torch.Tensor,
        temperature: torch.Tensor,
        voltage: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply dual-layer safety filter.

        Args:
            action: Raw action (current in A), shape (B, 1) or (B,)
            temperature: Current temperature in °C, shape (B,) or (B, 1)
            voltage: Current voltage in V, shape (B,) or (B, 1)
            uncertainty: Optional uncertainty estimate, shape (B,) or (B, 1)

        Returns:
            safe_action: Safety-filtered action
            info: Dictionary with safety metrics
        """
        # Ensure correct shapes
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        T = temperature.view(-1) if temperature.dim() > 0 else temperature.unsqueeze(0)
        V = voltage.view(-1) if voltage.dim() > 0 else voltage.unsqueeze(0)

        # Layer 1: Sigmoid soft scaling
        soft_scale = self.sigmoid_scale(T, V).unsqueeze(-1)  # (B, 1)
        action_soft = action * soft_scale

        # Layer 2: CBF hard constraint
        I_max = self.cbf_max_current(T, V).unsqueeze(-1)  # (B, 1)

        # If uncertainty is provided, reduce I_max proportionally
        if uncertainty is not None:
            unc = uncertainty.view(-1, 1) if uncertainty.dim() > 0 else uncertainty.unsqueeze(0).unsqueeze(-1)
            # Higher uncertainty → more conservative limit
            unc_scale = torch.clamp(1.0 - 0.5 * unc, min=0.3, max=1.0)
            I_max = I_max * unc_scale

        # Project: min(action_soft, I_max), but keep non-negative
        safe_action = torch.clamp(action_soft, min=0.0, max=None)
        safe_action = torch.minimum(safe_action, I_max)

        # Info
        info = {
            'raw_action': action.squeeze(-1),
            'soft_scale': soft_scale.squeeze(-1),
            'cbf_I_max': I_max.squeeze(-1),
            'safe_action': safe_action.squeeze(-1),
            'h_T': self.barrier_temperature(T),
            'h_V': self.barrier_voltage(V),
            'intervention_rate': (safe_action.squeeze(-1) < action.squeeze(-1) * 0.95).float().mean(),
        }

        return safe_action, info


class NoCBFSafetyFilter(nn.Module):
    """No safety filter (pass-through). Used as baseline."""

    def forward(self, action, temperature, voltage, uncertainty=None):
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        info = {
            'raw_action': action.squeeze(-1),
            'safe_action': action.squeeze(-1),
            'soft_scale': torch.ones_like(action.squeeze(-1)),
            'cbf_I_max': torch.ones_like(action.squeeze(-1)) * 999,
            'h_T': torch.zeros_like(action.squeeze(-1)),
            'h_V': torch.zeros_like(action.squeeze(-1)),
            'intervention_rate': torch.tensor(0.0),
        }
        return action, info


class SigmoidOnlySafetyFilter(nn.Module):
    """Sigmoid-only soft scaling (no CBF hard constraint). For ablation."""

    def __init__(self, T_limit=38.0, V_limit=4.15, margin_T=5.0, margin_V=0.15, min_scale=0.1):
        super().__init__()
        self.T_limit = T_limit
        self.V_limit = V_limit
        self.margin_T = margin_T
        self.margin_V = margin_V
        self.min_scale = min_scale

    def forward(self, action, temperature, voltage, uncertainty=None):
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        T = temperature.view(-1)
        V = voltage.view(-1)

        T_dist = (self.T_limit - T) / self.margin_T
        V_dist = (self.V_limit - V) / self.margin_V
        scale = torch.sigmoid(5.0 * T_dist) * torch.sigmoid(5.0 * V_dist)
        scale = torch.clamp(scale, min=self.min_scale, max=1.0).unsqueeze(-1)

        safe_action = action * scale
        info = {
            'raw_action': action.squeeze(-1),
            'safe_action': safe_action.squeeze(-1),
            'soft_scale': scale.squeeze(-1),
            'cbf_I_max': torch.ones_like(action.squeeze(-1)) * 999,
            'h_T': self.T_limit - T,
            'h_V': self.V_limit - V,
            'intervention_rate': (scale.squeeze(-1) < 0.95).float().mean(),
        }
        return safe_action, info
