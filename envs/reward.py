"""Reward function for battery charging environment."""
import numpy as np
from typing import Dict


class ChargingReward:
    """Compute reward for battery charging control.

    Reward = charge_speed_bonus - safety_penalty - degradation_penalty + target_bonus
    """

    def __init__(
        self,
        charge_speed_weight: float = 1.0,
        safety_penalty_weight: float = 5.0,
        degradation_penalty_weight: float = 2.0,
        target_bonus: float = 10.0,
        target_soc: float = 0.8,
        v_limit: float = 4.2,
    ):
        """Initialize reward function.

        Args:
            charge_speed_weight: Weight for charging speed reward.
            safety_penalty_weight: Weight for safety violation penalty.
            degradation_penalty_weight: Weight for degradation penalty.
            target_bonus: Bonus for reaching target SOC.
            target_soc: Target SOC value.
            v_limit: Voltage limit for proximity penalty.
        """
        self.charge_speed_weight = charge_speed_weight
        self.safety_penalty_weight = safety_penalty_weight
        self.degradation_penalty_weight = degradation_penalty_weight
        self.target_bonus = target_bonus
        self.target_soc = target_soc
        self.v_limit = v_limit

    def compute(
        self,
        soc: float,
        soc_prev: float,
        voltage: float,
        temperature: float,
        dT_dt: float,
        current: float,
        violation_vector: np.ndarray,
        done: bool = False,
    ) -> Dict[str, float]:
        """Compute reward components and total reward.

        Args:
            soc: Current SOC.
            soc_prev: Previous SOC.
            voltage: Current voltage.
            temperature: Current temperature.
            dT_dt: Temperature rate of change.
            current: Applied charging current.
            violation_vector: Constraint violation magnitudes.
            done: Whether episode is done.

        Returns:
            Dictionary with reward components and total.
        """
        # Charging speed: reward for SOC increase
        delta_soc = soc - soc_prev
        charge_speed = self.charge_speed_weight * delta_soc * 100.0

        # Safety penalty: penalize constraint violations
        safety_penalty = self.safety_penalty_weight * np.sum(violation_vector)

        # Voltage proximity penalty: encourage reducing current near voltage limit
        # This teaches the policy CC-CV behavior
        v_margin = max(0.0, voltage - (self.v_limit - 0.1))  # penalty starts 0.1V below limit
        voltage_proximity = 3.0 * v_margin ** 2  # quadratic penalty near limit

        # Degradation penalty: penalize high current at high temperature
        T_ref = 25.0
        deg_stress = (current / 6.0) ** 2 * (1.0 + 0.05 * max(0, temperature - T_ref))
        degradation = self.degradation_penalty_weight * deg_stress * 0.01

        # Target bonus (given at any termination if SOC >= target)
        bonus = 0.0
        if soc >= self.target_soc:
            bonus = self.target_bonus

        total = charge_speed - safety_penalty - voltage_proximity - degradation + bonus

        return {
            "total": float(total),
            "charge_speed": float(charge_speed),
            "safety_penalty": float(safety_penalty),
            "voltage_proximity": float(voltage_proximity),
            "degradation": float(degradation),
            "target_bonus": float(bonus),
        }
