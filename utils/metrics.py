"""Metrics computation utilities for battery charging evaluation."""
import numpy as np
from typing import Dict, List, Optional


def charging_time(soc_trajectory: np.ndarray, dt: float = 1.0) -> float:
    """Compute total charging time to reach target SOC.

    Args:
        soc_trajectory: SOC values over time.
        dt: Time step in seconds.

    Returns:
        Total charging time in seconds.
    """
    return len(soc_trajectory) * dt


def energy_efficiency(
    voltage: np.ndarray,
    current: np.ndarray,
    soc_start: float,
    soc_end: float,
    nominal_capacity: float = 1.1,
    nominal_voltage: float = 3.6,
    dt: float = 1.0,
) -> float:
    """Compute energy efficiency of charging process.

    Args:
        voltage: Voltage trajectory.
        current: Current trajectory.
        soc_start: Starting SOC.
        soc_end: Ending SOC.
        nominal_capacity: Nominal capacity in Ah.
        nominal_voltage: Nominal voltage in V.
        dt: Time step in seconds.

    Returns:
        Energy efficiency ratio.
    """
    energy_in = np.sum(voltage * current * dt) / 3600.0  # Wh
    energy_stored = nominal_capacity * nominal_voltage * (soc_end - soc_start)
    if energy_in <= 0:
        return 0.0
    return float(energy_stored / energy_in)


def constraint_violation_rate(
    violations: np.ndarray,
) -> float:
    """Compute fraction of steps with constraint violations.

    Args:
        violations: Boolean array of violations per step.

    Returns:
        Violation rate in [0, 1].
    """
    if len(violations) == 0:
        return 0.0
    return float(np.mean(violations))


def max_temperature_rise(temperature: np.ndarray) -> float:
    """Compute maximum temperature rise during charging.

    Args:
        temperature: Temperature trajectory in Celsius.

    Returns:
        Maximum temperature rise.
    """
    if len(temperature) < 2:
        return 0.0
    return float(np.max(temperature) - temperature[0])


def capacity_fade_proxy(
    current: np.ndarray,
    temperature: np.ndarray,
    dt: float = 1.0,
) -> float:
    """Compute a proxy for capacity fade based on aggressive charging.

    Higher currents and temperatures accelerate degradation.

    Args:
        current: Current trajectory.
        temperature: Temperature trajectory in Celsius.
        dt: Time step in seconds.

    Returns:
        Degradation proxy score (lower is better).
    """
    # Simplified Arrhenius-like degradation proxy
    T_ref = 25.0
    stress = np.abs(current) ** 2 * np.exp(0.01 * (temperature - T_ref))
    return float(np.sum(stress * dt))


def compute_episode_metrics(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    soc: np.ndarray,
    rewards: np.ndarray,
    risks: np.ndarray,
    dt: float = 1.0,
    v_max: float = 4.2,
    t_max: float = 45.0,
) -> Dict[str, float]:
    """Compute all episode-level metrics.

    Args:
        voltage: Voltage trajectory.
        current: Current trajectory.
        temperature: Temperature trajectory.
        soc: SOC trajectory.
        rewards: Reward trajectory.
        risks: Risk score trajectory.
        dt: Time step.
        v_max: Maximum voltage constraint.
        t_max: Maximum temperature constraint.

    Returns:
        Dictionary of metrics.
    """
    v_violations = voltage > v_max
    t_violations = temperature > t_max

    # Soft safety margins (0=at limit, positive=safe, negative=violated)
    voltage_margin = float(v_max - np.max(voltage)) if len(voltage) > 0 else v_max
    temp_margin = float(t_max - np.max(temperature)) if len(temperature) > 0 else t_max
    # Voltage utilization: how close to limit (higher = more aggressive = less safe)
    voltage_utilization = float(np.max(voltage) / v_max * 100) if len(voltage) > 0 else 0.0
    # Soft violation rate: fraction of steps where voltage > v_max - 0.1V (within 0.1V of limit)
    soft_v_viol = float(np.mean(voltage > (v_max - 0.1))) if len(voltage) > 0 else 0.0

    return {
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "charging_time_s": charging_time(soc, dt),
        "final_soc": float(soc[-1]) if len(soc) > 0 else 0.0,
        "max_temp_rise": max_temperature_rise(temperature),
        "max_voltage": float(np.max(voltage)) if len(voltage) > 0 else 0.0,
        "max_temperature": float(np.max(temperature)) if len(temperature) > 0 else 0.0,
        "voltage_violation_rate": constraint_violation_rate(v_violations),
        "temperature_violation_rate": constraint_violation_rate(t_violations),
        "voltage_margin": voltage_margin,
        "temp_margin": temp_margin,
        "voltage_utilization": voltage_utilization,
        "soft_voltage_violation_rate": soft_v_viol,
        "mean_risk": float(np.mean(risks)) if len(risks) > 0 else 0.0,
        "max_risk": float(np.max(risks)) if len(risks) > 0 else 0.0,
        "degradation_proxy": capacity_fade_proxy(current, temperature, dt),
        "mean_current": float(np.mean(current)) if len(current) > 0 else 0.0,
    }
