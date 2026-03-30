"""PyBaMM SPMe-based battery fast-charging Gymnasium environment.

Uses Single Particle Model with electrolyte (SPMe) for physically
realistic charging simulation with electrochemical accuracy.
"""
import os
os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"

import numpy as np
import pybamm
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from .constraints import ChargingConstraints
from .reward import ChargingReward


class PyBaMMChargingEnv(gym.Env):
    """Gymnasium env using PyBaMM SPMe for Li-ion fast charging.

    Action: continuous charging current in Amperes (scalar).
    Obs: window of [V, I, T, SOC, dV/dt, dT/dt, cycle, throughput, R_proxy, deg_proxy].
    Each step runs PyBaMM for dt seconds at the chosen current.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_voltage: float = 4.15,
        max_temperature: float = 38.0,
        max_c_rate: float = 6.0,
        min_c_rate: float = 0.0,
        target_soc: float = 0.8,
        max_steps: int = 200,
        dt: float = 30.0,
        window_length: int = 64,
        ambient_temp: float = 25.0,
        cycle_index: int = 0,
    ):
        super().__init__()
        self.max_voltage = max_voltage
        self.max_temperature = max_temperature
        self.max_c_rate = max_c_rate
        self.min_c_rate = min_c_rate
        self.target_soc = target_soc
        self.max_steps = max_steps
        self.dt = dt
        self.window_length = window_length
        self.ambient_temp = ambient_temp
        self.cycle_index = cycle_index

        # Build PyBaMM model: SPMe + lumped thermal + Marquis2019
        # Reduced heat transfer coeff to simulate realistic pack-level cooling
        self.model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        self.param = pybamm.ParameterValues("Marquis2019")
        self.param["Total heat transfer coefficient [W.m-2.K-1]"] = 0.5
        self.nominal_capacity = self.param["Nominal cell capacity [A.h]"]
        self.max_current = max_c_rate * self.nominal_capacity

        self.obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_length, self.obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.max_current], dtype=np.float32),
        )

        self.constraints = ChargingConstraints(
            voltage_limit=max_voltage, temperature_limit=max_temperature,
            dT_dt_limit=1.0, plating_risk_limit=0.3,
        )
        self.reward_fn = ChargingReward(
            charge_speed_weight=1.5, safety_penalty_weight=8.0,
            degradation_penalty_weight=2.0, target_bonus=20.0,
            target_soc=target_soc, v_limit=max_voltage,
        )

        # Runtime state
        self.step_count = 0
        self.total_time = 0.0
        self.soc = 0.0
        self.voltage = 3.0
        self.current = 0.0
        self.temperature = ambient_temp
        self.prev_voltage = 3.0
        self.prev_temperature = ambient_temp
        self.charge_throughput = 0.0
        self.degradation_proxy = 0.0
        self.history = []

    def _run_pybamm_step(self, current_A: float) -> Dict[str, float]:
        """Run PyBaMM for dt seconds at given current."""
        try:
            param = self.param.copy()
            param["Current function [A]"] = -abs(current_A)  # negative = charge in PyBaMM
            param["Ambient temperature [K]"] = 273.15 + self.ambient_temp
            param["Initial temperature [K]"] = 273.15 + self.temperature

            # Use pybamm's built-in initial SOC setting
            soc_init = float(np.clip(self.soc, 0.005, 0.995))

            model_fresh = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
            sim = pybamm.Simulation(model_fresh, parameter_values=param)
            sol = sim.solve(initial_soc=soc_init, t_eval=[0, self.dt])

            V = np.array(sol["Terminal voltage [V]"].entries).flatten()
            I = np.array(sol["Current [A]"].entries).flatten()
            try:
                T_K = np.array(sol["Volume-averaged cell temperature [K]"].entries).flatten()
                temp = float(T_K[-1]) - 273.15  # Convert K to C
            except Exception:
                try:
                    T_C = np.array(sol["Volume-averaged cell temperature [C]"].entries).flatten()
                    temp = float(T_C[-1]) if len(T_C) > 1 else self.ambient_temp
                except Exception:
                    temp = self.ambient_temp
            try:
                Q = np.array(sol["Throughput capacity [A.h]"].entries).flatten()
                throughput = float(Q[-1]) if len(Q) > 1 else 0.0
            except Exception:
                throughput = abs(current_A) * self.dt / 3600.0
            try:
                plating = np.array(sol["Loss of capacity to negative lithium plating [A.h]"].entries).flatten()
                plating_val = float(plating[-1]) if len(plating) > 0 else 0.0
            except Exception:
                plating_val = 0.0

            # Get real SOC from PyBaMM: discharge_capacity / nominal_capacity
            # PyBaMM "Discharge capacity" is negative during charge, so use abs
            try:
                dc = np.array(sol["Discharge capacity [A.h]"].entries).flatten()
                # SOC at end = initial_soc + abs(charged capacity) / nominal_capacity
                real_soc = soc_init + abs(float(dc[-1])) / self.nominal_capacity
                real_soc = float(np.clip(real_soc, 0.0, 1.0))
            except Exception:
                real_soc = None  # fallback to simple integration

            return {
                "voltage": float(V[-1]),
                "current": abs(current_A),
                "temperature": temp,
                "throughput": throughput,
                "plating": plating_val,
                "real_soc": real_soc,
                "ok": True,
            }
        except Exception:
            # Fallback if PyBaMM fails (e.g., infeasible conditions)
            return {
                "voltage": min(self.voltage + 0.02 * current_A, self.max_voltage + 0.2),
                "current": abs(current_A),
                "temperature": self.temperature + 0.01 * current_A ** 2,
                "throughput": abs(current_A) * self.dt / 3600.0,
                "plating": 0.0,
                "ok": False,
            }

    def _obs_vector(self) -> np.ndarray:
        dV = (self.voltage - self.prev_voltage) / max(self.dt, 1.0)
        dT = (self.temperature - self.prev_temperature) / max(self.dt, 1.0)
        R = self.voltage / max(self.current, 0.01)
        R = np.clip(R, 0, 10.0)
        return np.array([
            self.voltage, self.current, self.temperature, self.soc,
            dV, dT, float(self.cycle_index), self.charge_throughput,
            R, self.degradation_proxy,
        ], dtype=np.float32)

    def _windowed_obs(self) -> np.ndarray:
        obs = np.zeros((self.window_length, self.obs_dim), dtype=np.float32)
        if self.history:
            start = max(0, len(self.history) - self.window_length)
            w = self.history[start:]
            obs[-len(w):] = np.array(w, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self.soc = opts.get("initial_soc", 0.0)
        self.ambient_temp = opts.get("ambient_temp", self.ambient_temp)
        self.cycle_index = opts.get("cycle_index", self.cycle_index)

        self.step_count = 0
        self.total_time = 0.0
        self.voltage = 3.0 + self.soc * 1.0
        self.current = 0.0
        self.temperature = self.ambient_temp
        self.prev_voltage = self.voltage
        self.prev_temperature = self.temperature
        self.charge_throughput = 0.0
        self.degradation_proxy = 0.0

        self.history = []
        vec = self._obs_vector()
        for _ in range(self.window_length):
            self.history.append(vec.copy())

        return self._windowed_obs(), self._get_info()

    def step(self, action):
        current_A = float(np.clip(action[0], 0.0, self.max_current))
        soc_prev = self.soc
        self.prev_voltage = self.voltage
        self.prev_temperature = self.temperature

        # PyBaMM simulation
        result = self._run_pybamm_step(current_A)
        self.voltage = result["voltage"]
        self.current = result["current"]
        self.temperature = result["temperature"]
        self.charge_throughput += result["throughput"]
        # Instantaneous Arrhenius-based degradation proxy (matches dataset generation)
        c_rate = current_A / self.nominal_capacity
        self.degradation_proxy = c_rate**2 * np.exp(0.01 * (self.temperature - 25.0))

        # SOC update: use PyBaMM real SOC if available, else fallback to integration
        if result.get("real_soc") is not None:
            self.soc = result["real_soc"]
        else:
            delta_soc = current_A * self.dt / (self.nominal_capacity * 3600.0)
            self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))
        self.step_count += 1
        self.total_time += self.dt
        self.history.append(self._obs_vector())

        # Constraints
        dV = (self.voltage - self.prev_voltage) / self.dt
        dT = (self.temperature - self.prev_temperature) / self.dt
        plating_risk = min(1.0, self.degradation_proxy * 10.0)
        viol_vec = self.constraints.violation_vector(self.voltage, self.temperature, dT, plating_risk)

        # Termination
        terminated = self.soc >= self.target_soc
        truncated = self.step_count >= self.max_steps
        severe = (self.voltage > self.max_voltage + 0.05) or (self.temperature > self.max_temperature + 3.0)
        if severe:
            terminated = True

        # Reward
        reward_dict = self.reward_fn.compute(
            soc=self.soc, soc_prev=soc_prev, voltage=self.voltage,
            temperature=self.temperature, dT_dt=dT, current=current_A,
            violation_vector=viol_vec, done=terminated,
        )

        info = self._get_info()
        info["reward_components"] = reward_dict
        info["pybamm_ok"] = result["ok"]
        info["violations"] = self.constraints.check(self.voltage, self.temperature, dT, plating_risk)
        info["plating_risk"] = plating_risk
        info["total_time_min"] = self.total_time / 60.0

        return self._windowed_obs(), reward_dict["total"], terminated, truncated, info

    def _get_info(self):
        return {
            "soc": self.soc, "voltage": self.voltage,
            "temperature": self.temperature, "current": self.current,
            "soh": 1.0, "resistance": self.voltage / max(self.current, 0.01),
            "charge_throughput": self.charge_throughput,
            "degradation_proxy": self.degradation_proxy,
            "step": self.step_count, "total_time_s": self.total_time,
            "dV_dt": (self.voltage - self.prev_voltage) / max(self.dt, 1.0),
            "dT_dt": (self.temperature - self.prev_temperature) / max(self.dt, 1.0),
        }


def make_pybamm_env(cfg=None):
    """Create PyBaMM charging env from config."""
    cfg = cfg or {}
    c = cfg.get("env", cfg)
    return PyBaMMChargingEnv(
        max_voltage=c.get("max_voltage", 4.2),
        max_temperature=c.get("max_temperature", 45.0),
        max_c_rate=c.get("max_c_rate", 6.0),
        target_soc=c.get("target_soc", 0.8),
        max_steps=c.get("max_steps", 200),
        dt=c.get("dt", 30.0),
        window_length=c.get("window_length", 64),
    )
