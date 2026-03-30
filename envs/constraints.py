"""Safety constraints for battery charging environment."""
import numpy as np
from typing import Dict, Tuple


class ChargingConstraints:
    """Define and evaluate safety constraints for battery charging.

    Constraints include voltage, temperature, temperature rate,
    and lithium plating risk limits.
    """

    def __init__(
        self,
        voltage_limit: float = 4.2,
        temperature_limit: float = 45.0,
        dT_dt_limit: float = 1.0,
        plating_risk_limit: float = 0.3,
    ):
        """Initialize constraints.

        Args:
            voltage_limit: Maximum voltage in V.
            temperature_limit: Maximum temperature in °C.
            dT_dt_limit: Maximum temperature rate in °C/step.
            plating_risk_limit: Maximum plating risk score.
        """
        self.voltage_limit = voltage_limit
        self.temperature_limit = temperature_limit
        self.dT_dt_limit = dT_dt_limit
        self.plating_risk_limit = plating_risk_limit

    def check(
        self,
        voltage: float,
        temperature: float,
        dT_dt: float,
        plating_risk: float = 0.0,
    ) -> Dict[str, bool]:
        """Check all constraints and return violation flags.

        Args:
            voltage: Current voltage.
            temperature: Current temperature.
            dT_dt: Temperature rate of change.
            plating_risk: Plating risk proxy.

        Returns:
            Dictionary mapping constraint name to violation flag.
        """
        return {
            "voltage": voltage > self.voltage_limit,
            "temperature": temperature > self.temperature_limit,
            "dT_dt": abs(dT_dt) > self.dT_dt_limit,
            "plating_risk": plating_risk > self.plating_risk_limit,
        }

    def any_violated(
        self,
        voltage: float,
        temperature: float,
        dT_dt: float,
        plating_risk: float = 0.0,
    ) -> bool:
        """Check if any constraint is violated.

        Args:
            voltage: Current voltage.
            temperature: Current temperature.
            dT_dt: Temperature rate of change.
            plating_risk: Plating risk proxy.

        Returns:
            True if any constraint is violated.
        """
        violations = self.check(voltage, temperature, dT_dt, plating_risk)
        return any(violations.values())

    def hard_violated(
        self,
        voltage: float,
        temperature: float,
    ) -> bool:
        """Check if hard safety constraints (voltage, temperature) are violated.

        Args:
            voltage: Current voltage.
            temperature: Current temperature.

        Returns:
            True if hard constraint violated.
        """
        return voltage > self.voltage_limit or temperature > self.temperature_limit

    def violation_vector(
        self,
        voltage: float,
        temperature: float,
        dT_dt: float,
        plating_risk: float = 0.0,
    ) -> np.ndarray:
        """Get violation magnitudes as a vector.

        Args:
            voltage: Current voltage.
            temperature: Current temperature.
            dT_dt: Temperature rate of change.
            plating_risk: Plating risk proxy.

        Returns:
            Array of violation magnitudes (0 if not violated).
        """
        return np.array([
            max(0.0, voltage - self.voltage_limit),
            max(0.0, temperature - self.temperature_limit),
            max(0.0, abs(dT_dt) - self.dT_dt_limit),
            max(0.0, plating_risk - self.plating_risk_limit),
        ], dtype=np.float32)
