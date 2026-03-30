"""Battery fast-charging environment compatible with Gymnasium API.

Simulates lithium-ion battery charging with simplified electrochemical
dynamics. Supports both online interaction and offline dataset generation.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from .constraints import ChargingConstraints
from .reward import ChargingReward


class BatteryChargingEnv(gym.Env):
    """Gymnasium environment for lithium-ion battery fast charging.

    Observation: last L steps of [voltage, current, temperature, soc,
                 dV/dt, dT/dt, cycle_index, charge_throughput,
                 internal_resistance_proxy, degradation_proxy]
    Action: continuous charging current (scalar)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        nominal_capacity_ah: float = 1.1,
        nominal_voltage: float = 3.6,
        max_voltage: float = 4.2,
        min_voltage: float = 2.5,
        max_temperature: float = 45.0,
        max_dT_dt: float = 1.0,
        ambient_temperature: float = 25.0,
        initial_soc: float = 0.0,
        target_soc: float = 0.8,
        min_current: float = 0.0,
        max_current: float = 6.0,
        max_steps: int = 500,
        dt: float = 10.0,
        window_length: int = 64,
        reward_cfg: Optional[Dict] = None,
        constraint_cfg: Optional[Dict] = None,
        cycle_index: int = 0,
        initial_soh: float = 1.0,
        obs_noise_std: float = 0.0,
    ):
        """Initialize battery charging environment.

        Args:
            nominal_capacity_ah: Nominal capacity in Ah.
            nominal_voltage: Nominal voltage in V.
            max_voltage: Maximum voltage limit.
            min_voltage: Minimum voltage.
            max_temperature: Maximum temperature limit.
            max_dT_dt: Maximum temperature rate.
            ambient_temperature: Ambient temperature in °C.
            initial_soc: Starting SOC.
            target_soc: Target SOC for episode termination.
            min_current: Minimum charging current.
            max_current: Maximum charging current.
            max_steps: Maximum steps per episode.
            dt: Time step in seconds.
            window_length: History window length for observations.
            reward_cfg: Reward function configuration.
            constraint_cfg: Constraint configuration.
            cycle_index: Current cycle index for degradation tracking.
            initial_soh: Initial state of health (1.0 = fresh).
            obs_noise_std: Observation noise standard deviation.
        """
        super().__init__()

        self.nominal_capacity = nominal_capacity_ah
        self.nominal_voltage = nominal_voltage
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.max_temperature = max_temperature
        self.max_dT_dt = max_dT_dt
        self.ambient_temp = ambient_temperature
        self.initial_soc = initial_soc
        self.target_soc = target_soc
        self.min_current = min_current
        self.max_current = max_current
        self.max_steps = max_steps
        self.dt = dt
        self.window_length = window_length
        self.cycle_index = cycle_index
        self.initial_soh = initial_soh
        self.obs_noise_std = obs_noise_std

        # Obs: (window_length, 10 features)
        self.obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_length, self.obs_dim),
            dtype=np.float32,
        )
        # Action: single continuous current
        self.action_space = spaces.Box(
            low=np.array([min_current], dtype=np.float32),
            high=np.array([max_current], dtype=np.float32),
            shape=(1,),
        )

        # Constraints and reward
        c_cfg = constraint_cfg or {}
        self.constraints = ChargingConstraints(
            voltage_limit=c_cfg.get("voltage_limit", max_voltage),
            temperature_limit=c_cfg.get("temperature_limit", max_temperature),
            dT_dt_limit=c_cfg.get("dT_dt_limit", max_dT_dt),
            plating_risk_limit=c_cfg.get("plating_risk_limit", 0.3),
        )
        r_cfg = reward_cfg or {}
        self.reward_fn = ChargingReward(
            charge_speed_weight=r_cfg.get("charge_speed_weight", 1.0),
            safety_penalty_weight=r_cfg.get("safety_penalty_weight", 5.0),
            degradation_penalty_weight=r_cfg.get("degradation_penalty_weight", 2.0),
            target_bonus=r_cfg.get("target_bonus", 10.0),
            target_soc=self.target_soc,
        )

        # State variables
        self.soc = self.initial_soc
        self.voltage = self._soc_to_ocv(self.initial_soc)
        self.temperature = self.ambient_temp
        self.current = 0.0
        self.step_count = 0
        self.charge_throughput = 0.0
        self.soh = self.initial_soh
        self.resistance = self._base_resistance()
        self.dV_dt = 0.0
        self.dT_dt = 0.0
        self.degradation_proxy = 0.0
        self.history = []

    def _soc_to_ocv(self, soc: float) -> float:
        """Simplified SOC-to-OCV mapping for LFP/NMC-like cell.

        Args:
            soc: State of charge [0, 1].

        Returns:
            Open circuit voltage.
        """
        soc = np.clip(soc, 0.0, 1.0)
        # Polynomial approximation
        ocv = (
            3.0
            + 1.2 * soc
            - 0.5 * soc ** 2
            + 0.3 * soc ** 3
            + 0.1 * np.log(soc + 0.01)
            - 0.1 * np.log(1.01 - soc)
        )
        return float(np.clip(ocv, self.min_voltage, self.max_voltage + 0.1))

    def _base_resistance(self) -> float:
        """Compute base internal resistance based on SOH.

        Returns:
            Internal resistance in Ohms.
        """
        r0 = 0.05  # Fresh cell resistance
        return r0 * (2.0 - self.soh)

    def _plating_risk(self, current: float, temperature: float, soc: float) -> float:
        """Estimate lithium plating risk proxy.

        Higher current, lower temperature, higher SOC → higher risk.

        Args:
            current: Charging current.
            temperature: Cell temperature.
            soc: State of charge.

        Returns:
            Plating risk score [0, 1].
        """
        c_factor = current / self.max_current
        t_factor = max(0.0, (30.0 - temperature) / 30.0)
        s_factor = soc ** 2
        risk = 0.3 * c_factor + 0.4 * t_factor + 0.3 * s_factor
        return float(np.clip(risk, 0.0, 1.0))

    def _step_dynamics(self, current: float):
        """Update battery state using simplified dynamics.

        Args:
            current: Applied charging current in A.
        """
        prev_voltage = self.voltage
        prev_temperature = self.temperature

        # SOC update
        delta_soc = current * self.dt / (self.nominal_capacity * 3600.0)
        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)

        # Voltage = OCV + I*R
        ocv = self._soc_to_ocv(self.soc)
        self.resistance = self._base_resistance() * (1.0 + 0.3 * self.soc)
        self.voltage = ocv + current * self.resistance

        # Temperature dynamics (lumped thermal model)
        q_gen = current ** 2 * self.resistance  # Joule heating
        q_cool = 0.5 * (self.temperature - self.ambient_temp)  # Cooling
        thermal_mass = 50.0  # J/K
        delta_T = (q_gen - q_cool) * self.dt / thermal_mass
        self.temperature += delta_T

        # Rates
        self.dV_dt = (self.voltage - prev_voltage) / self.dt
        self.dT_dt = (self.temperature - prev_temperature) / self.dt

        # Charge throughput
        self.charge_throughput += abs(current) * self.dt / 3600.0

        # Degradation proxy update
        T_ref = 25.0
        stress = (current / self.max_current) ** 2 * np.exp(
            0.01 * (self.temperature - T_ref)
        )
        self.degradation_proxy += stress * self.dt * 1e-4

        # SOH degradation (slow)
        self.soh = max(0.5, self.soh - stress * self.dt * 1e-7)

        self.current = current
        self.step_count += 1

    def _get_obs_vector(self) -> np.ndarray:
        """Get current observation vector (single step).

        Returns:
            Array of shape (10,).
        """
        obs = np.array([
            self.voltage,
            self.current,
            self.temperature,
            self.soc,
            self.dV_dt,
            self.dT_dt,
            self.cycle_index,
            self.charge_throughput,
            self.resistance,
            self.degradation_proxy,
        ], dtype=np.float32)
        if self.obs_noise_std > 0:
            obs += np.random.randn(self.obs_dim).astype(np.float32) * self.obs_noise_std
        return obs

    def _get_windowed_obs(self) -> np.ndarray:
        """Get windowed observation from history.

        Returns:
            Array of shape (window_length, obs_dim).
        """
        obs = np.zeros((self.window_length, self.obs_dim), dtype=np.float32)
        hist_len = len(self.history)
        if hist_len > 0:
            start = max(0, hist_len - self.window_length)
            window = self.history[start:]
            obs[-len(window):] = np.array(window, dtype=np.float32)
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Optional reset options (initial_soc, initial_soh, ambient_temp, cycle_index).

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)

        opts = options or {}
        self.initial_soc = opts.get("initial_soc", self.initial_soc)
        self.initial_soh = opts.get("initial_soh", self.initial_soh)
        self.ambient_temp = opts.get("ambient_temp", self.ambient_temp)
        self.cycle_index = opts.get("cycle_index", self.cycle_index)

        self.soc = self.initial_soc
        self.soh = self.initial_soh
        self.voltage = self._soc_to_ocv(self.soc)
        self.temperature = self.ambient_temp
        self.current = 0.0
        self.step_count = 0
        self.charge_throughput = 0.0
        self.resistance = self._base_resistance()
        self.dV_dt = 0.0
        self.dT_dt = 0.0
        self.degradation_proxy = 0.0

        # Initialize history with zeros
        self.history = []
        init_obs = self._get_obs_vector()
        for _ in range(self.window_length):
            self.history.append(init_obs.copy())

        obs = self._get_windowed_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Charging current array of shape (1,).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        current = float(np.clip(action[0], self.min_current, self.max_current))
        soc_prev = self.soc

        # Step dynamics
        self._step_dynamics(current)

        # Record observation
        self.history.append(self._get_obs_vector())

        # Check constraints
        plating = self._plating_risk(current, self.temperature, self.soc)
        violation_vec = self.constraints.violation_vector(
            self.voltage, self.temperature, self.dT_dt, plating
        )

        # Check termination
        terminated = self.soc >= self.target_soc
        truncated = self.step_count >= self.max_steps
        # Only terminate on SEVERE violations (well beyond limits)
        # This allows policy to learn CV-phase behavior near voltage limit
        severe_violation = (self.voltage > self.max_voltage + 0.1) or (self.temperature > self.max_temperature + 5.0)
        if severe_violation:
            terminated = True

        # Compute reward
        reward_dict = self.reward_fn.compute(
            soc=self.soc,
            soc_prev=soc_prev,
            voltage=self.voltage,
            temperature=self.temperature,
            dT_dt=self.dT_dt,
            current=current,
            violation_vector=violation_vec,
            done=terminated,
        )

        obs = self._get_windowed_obs()
        info = self._get_info()
        info["reward_components"] = reward_dict
        info["violations"] = self.constraints.check(
            self.voltage, self.temperature, self.dT_dt, plating
        )
        info["plating_risk"] = plating

        return obs, reward_dict["total"], terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get current state information dict.

        Returns:
            Info dictionary.
        """
        return {
            "soc": self.soc,
            "voltage": self.voltage,
            "temperature": self.temperature,
            "current": self.current,
            "dV_dt": self.dV_dt,
            "dT_dt": self.dT_dt,
            "soh": self.soh,
            "resistance": self.resistance,
            "charge_throughput": self.charge_throughput,
            "degradation_proxy": self.degradation_proxy,
            "step": self.step_count,
        }


def make_env(cfg: Dict) -> BatteryChargingEnv:
    """Create battery charging environment from config.

    Args:
        cfg: Environment configuration dictionary.

    Returns:
        BatteryChargingEnv instance.
    """
    env_cfg = cfg.get("env", cfg)
    return BatteryChargingEnv(
        nominal_capacity_ah=env_cfg.get("nominal_capacity_ah", 1.1),
        nominal_voltage=env_cfg.get("nominal_voltage", 3.6),
        max_voltage=env_cfg.get("max_voltage", 4.2),
        min_voltage=env_cfg.get("min_voltage", 2.5),
        max_temperature=env_cfg.get("max_temperature", 45.0),
        max_dT_dt=env_cfg.get("max_dT_dt", 1.0),
        ambient_temperature=env_cfg.get("ambient_temperature", 25.0),
        initial_soc=env_cfg.get("initial_soc", 0.0),
        target_soc=env_cfg.get("target_soc", 0.8),
        min_current=env_cfg.get("min_current", 0.0),
        max_current=env_cfg.get("max_current", 6.0),
        max_steps=env_cfg.get("max_steps", 500),
        dt=env_cfg.get("dt", 10.0),
        window_length=env_cfg.get("window_length", 64),
        reward_cfg=env_cfg.get("reward", {}),
        constraint_cfg=env_cfg.get("constraints", {}),
    )
