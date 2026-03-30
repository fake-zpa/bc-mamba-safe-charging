"""Feature engineering utilities for battery time-series data."""
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_derivatives(
    signal: np.ndarray, dt: float = 1.0
) -> np.ndarray:
    """Compute first derivative of a signal using central differences.

    Args:
        signal: 1D signal array.
        dt: Time step.

    Returns:
        Derivative array of same length.
    """
    deriv = np.zeros_like(signal)
    if len(signal) < 2:
        return deriv
    deriv[1:-1] = (signal[2:] - signal[:-2]) / (2.0 * dt)
    deriv[0] = (signal[1] - signal[0]) / dt if len(signal) > 1 else 0.0
    deriv[-1] = (signal[-1] - signal[-2]) / dt if len(signal) > 1 else 0.0
    return deriv


def compute_internal_resistance_proxy(
    voltage: np.ndarray, current: np.ndarray
) -> np.ndarray:
    """Estimate internal resistance from voltage and current.

    Uses simple V/I ratio with smoothing.

    Args:
        voltage: Voltage time series.
        current: Current time series.

    Returns:
        Internal resistance proxy array.
    """
    eps = 1e-6
    r_proxy = np.zeros_like(voltage)
    mask = np.abs(current) > eps
    r_proxy[mask] = voltage[mask] / (current[mask] + eps)
    # Smooth with running mean
    if len(r_proxy) > 5:
        kernel = np.ones(5) / 5.0
        r_proxy = np.convolve(r_proxy, kernel, mode="same")
    return np.clip(r_proxy, 0.0, 1.0)


def compute_degradation_proxy(
    current: np.ndarray,
    temperature: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Compute cumulative degradation proxy.

    Args:
        current: Current time series.
        temperature: Temperature time series.
        dt: Time step.

    Returns:
        Cumulative degradation proxy.
    """
    T_ref = 25.0
    stress = np.abs(current) ** 2 * np.exp(0.01 * (temperature - T_ref))
    return np.cumsum(stress * dt * 1e-4)


def build_feature_matrix(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    soc: np.ndarray,
    cycle_index: int = 0,
    dt: float = 1.0,
) -> np.ndarray:
    """Build full feature matrix from raw signals.

    Features: [voltage, current, temperature, soc, dV/dt, dT/dt,
               cycle_index, charge_throughput, resistance_proxy, degradation_proxy]

    Args:
        voltage: Voltage time series.
        current: Current time series.
        temperature: Temperature time series.
        soc: SOC time series.
        cycle_index: Cycle index.
        dt: Time step.

    Returns:
        Feature matrix of shape (T, 10).
    """
    T = len(voltage)
    dV_dt = compute_derivatives(voltage, dt)
    dT_dt = compute_derivatives(temperature, dt)
    charge_throughput = np.cumsum(np.abs(current) * dt / 3600.0)
    r_proxy = compute_internal_resistance_proxy(voltage, current)
    deg_proxy = compute_degradation_proxy(current, temperature, dt)
    cycle_arr = np.full(T, cycle_index, dtype=np.float32)

    features = np.stack([
        voltage, current, temperature, soc,
        dV_dt, dT_dt, cycle_arr, charge_throughput,
        r_proxy, deg_proxy,
    ], axis=-1).astype(np.float32)

    return features


def normalize_features(
    features: np.ndarray,
    stats: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize features using z-score normalization.

    Args:
        features: Feature matrix of shape (..., n_features).
        stats: Optional precomputed mean and std.

    Returns:
        Tuple of (normalized features, stats dict).
    """
    if stats is None:
        mean = np.mean(features, axis=tuple(range(features.ndim - 1)), keepdims=True)
        std = np.std(features, axis=tuple(range(features.ndim - 1)), keepdims=True) + 1e-8
        stats = {"mean": mean.squeeze(), "std": std.squeeze()}
    else:
        mean = stats["mean"]
        std = stats["std"]

    normalized = (features - mean) / std
    return normalized, stats


def create_windows(
    features: np.ndarray,
    window_length: int = 64,
    stride: int = 1,
) -> np.ndarray:
    """Create sliding windows from feature matrix.

    Args:
        features: Feature matrix of shape (T, n_features).
        window_length: Window length.
        stride: Stride between windows.

    Returns:
        Array of shape (n_windows, window_length, n_features).
    """
    T, D = features.shape
    if T < window_length:
        # Pad with zeros
        padded = np.zeros((window_length, D), dtype=features.dtype)
        padded[-T:] = features
        return padded[np.newaxis]

    n_windows = (T - window_length) // stride + 1
    windows = np.zeros((n_windows, window_length, D), dtype=features.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = features[start:start + window_length]
    return windows
