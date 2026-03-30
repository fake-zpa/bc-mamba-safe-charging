"""Fast PyBaMM dataset generation using Experiment API.

Instead of stepping PyBaMM one-dt-at-a-time (slow), we run a full
charging protocol in one solve() call, then slice the solution into
RL (s,a,r,s',done) transitions. This is 50-100x faster.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pybamm
from battery_mamba_safe_rl.datasets.feature_utils import normalize_features


def run_pybamm_protocol(c_rates, durations_s, initial_soc=0.005):
    """Run a multi-stage CC protocol via PyBaMM Experiment API.

    Args:
        c_rates: list of C-rates for each stage.
        durations_s: list of durations (seconds) per stage.
        initial_soc: initial state of charge.

    Returns:
        Dict with time-series arrays, or None if failed.
    """
    model = pybamm.lithium_ion.SPM()
    param = pybamm.ParameterValues("Marquis2019")
    capacity = param["Nominal cell capacity [A.h]"]

    # Build experiment steps
    steps = []
    for c_rate, dur in zip(c_rates, durations_s):
        current_A = c_rate * capacity
        dur_min = dur / 60.0
        steps.append("Charge at %.3f A for %.1f minutes" % (current_A, dur_min))

    try:
        exp = pybamm.Experiment(steps, period="10 seconds")  # 10s resolution
        sim = pybamm.Simulation(model, experiment=exp)
        sol = sim.solve(initial_soc=max(0.005, min(initial_soc, 0.99)))

        t = np.array(sol.t).flatten()
        V = np.array(sol["Terminal voltage [V]"].entries).flatten()
        I = np.array(sol["Current [A]"].entries).flatten()

        try:
            T = np.array(sol["Volume-averaged cell temperature [C]"].entries).flatten()
        except Exception:
            T = np.full_like(t, 25.0)

        try:
            Q = np.array(sol["Discharge capacity [A.h]"].entries).flatten()
        except Exception:
            Q = np.cumsum(np.abs(I) * np.diff(t, prepend=t[0]) / 3600.0)

        try:
            plating = np.array(sol["Loss of capacity to negative lithium plating [A.h]"].entries).flatten()
        except Exception:
            plating = np.zeros_like(t)

        # SOC from discharge capacity (negative = charging)
        soc = initial_soc + np.abs(Q) / capacity

        return {
            "t": t, "V": V, "I": np.abs(I), "T": T,
            "soc": np.clip(soc, 0, 1), "Q": Q,
            "plating": plating, "capacity": capacity,
        }
    except Exception as e:
        return None


def protocol_to_trajectory(sol_data, protocol_name, dt=30.0, cell_id=0, cycle_idx=0):
    """Convert a PyBaMM solution into RL trajectory with features.

    Resample to uniform dt steps and compute observation features.

    Args:
        sol_data: dict from run_pybamm_protocol.
        protocol_name: string name of protocol.
        dt: target time step for RL transitions.
        cell_id: cell identifier.
        cycle_idx: cycle identifier.

    Returns:
        Trajectory dict compatible with OfflineRLDataset.
    """
    t = sol_data["t"]
    V = sol_data["V"]
    I = sol_data["I"]
    T = sol_data["T"]
    soc = sol_data["soc"]
    plating = sol_data["plating"]

    # Resample to uniform dt
    t_uniform = np.arange(0, t[-1], dt)
    if len(t_uniform) < 5:
        return None

    V_u = np.interp(t_uniform, t, V)
    I_u = np.interp(t_uniform, t, I)
    T_u = np.interp(t_uniform, t, T) if len(T) == len(t) else np.full_like(t_uniform, 25.0)
    soc_u = np.interp(t_uniform, t, soc)
    plat_u = np.interp(t_uniform, t, plating) if len(plating) == len(t) else np.zeros_like(t_uniform)

    L = len(t_uniform)

    # Compute derivatives
    dV_dt = np.gradient(V_u, dt)
    dT_dt = np.gradient(T_u, dt)

    # Charge throughput
    charge_throughput = np.cumsum(I_u * dt / 3600.0)

    # Resistance proxy
    R_proxy = np.where(I_u > 0.01, V_u / I_u, 0.05)
    R_proxy = np.clip(R_proxy, 0, 10.0)

    # Degradation proxy
    deg_proxy = np.cumsum(np.abs(plat_u))

    # Build feature matrix (L, 10)
    features = np.stack([
        V_u, I_u, T_u, soc_u, dV_dt, dT_dt,
        np.full(L, float(cycle_idx)),
        charge_throughput, R_proxy, deg_proxy,
    ], axis=-1).astype(np.float32)

    # Rewards: strongly incentivize FAST charging
    # NO voltage proximity penalty in offline data (that goes in eval env only)
    # The RL agent should learn to use high C-rates from data
    delta_soc = np.diff(soc_u, prepend=soc_u[0])

    # Base: SOC increase (the ONLY positive reward per step)
    rewards = delta_soc * 200.0

    # Time penalty: -1.0 per step (strongly encourages fewer steps)
    rewards -= 1.0

    # Mild degradation penalty (don't dominate)
    rewards -= (I_u / 5.0) ** 2 * 0.005

    # Large one-time bonus for reaching target SOC
    target_reached = soc_u >= 0.8
    if np.any(target_reached):
        first_target_step = np.argmax(target_reached)
        rewards[first_target_step] += 80.0

    actions = I_u[:, np.newaxis].astype(np.float32)

    return {
        "features": features,
        "actions": actions,
        "rewards": rewards.astype(np.float32),
        "voltage": V_u,
        "current": I_u,
        "temperature": T_u,
        "soc": soc_u,
        "cell_id": cell_id,
        "cycle_index": cycle_idx,
        "soh": 1.0,
        "protocol_type": protocol_name,
        "charging_time_min": t_uniform[-1] / 60.0,
        "final_soc": float(soc_u[-1]),
    }


def generate_all_protocols():
    """Generate diverse charging protocol configs. Focus on feasible C-rates (<=4C)."""
    protocols = []

    # CC at various C-rates (extended range 0.5-5C for more aggressive data)
    for c_rate in np.arange(0.5, 5.1, 0.25):
        dur = 0.85 / c_rate * 3600 * 1.3
        for init_soc in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15]:
            protocols.append({
                "name": "CC_%.2fC_soc%.0f" % (c_rate, init_soc * 100),
                "c_rates": [c_rate],
                "durations_s": [dur],
                "initial_soc": init_soc,
            })

    # Multi-stage CC: extended to 5C first stage for aggressive fast charging
    multi_configs = [
        ([4.0, 3.0, 2.0, 1.0], [200, 300, 300, 600]),
        ([3.5, 2.5, 1.5], [300, 400, 600]),
        ([3.0, 2.0, 1.0], [400, 400, 600]),
        ([3.5, 2.0, 1.0], [300, 400, 600]),
        ([4.0, 2.5, 1.0], [250, 350, 600]),
        ([3.0, 1.5, 0.5], [400, 400, 800]),
        ([2.5, 2.0, 1.5, 1.0], [300, 300, 300, 600]),
        ([3.5, 3.0, 2.0, 1.0, 0.5], [150, 200, 250, 300, 500]),
        ([2.0, 1.5, 1.0], [500, 400, 600]),
        ([4.0, 3.0, 1.5, 0.5], [200, 250, 350, 600]),
        ([3.0, 2.5, 2.0, 1.5, 1.0], [200, 200, 250, 300, 500]),
        ([2.5, 1.5, 0.8], [500, 500, 800]),
        ([3.5, 2.0, 0.5], [300, 500, 800]),
        ([4.0, 2.0, 1.0, 0.5], [200, 300, 400, 600]),
        ([3.0, 2.0, 1.5], [400, 400, 600]),
        ([2.0, 1.0], [600, 900]),
        ([3.5, 1.5], [400, 800]),
        ([4.0, 1.0], [300, 900]),
        # New aggressive 4.5C-5C first stage protocols
        ([5.0, 3.0, 1.5], [180, 300, 600]),
        ([5.0, 4.0, 2.0, 1.0], [150, 200, 300, 500]),
        ([4.5, 3.5, 2.0], [200, 300, 600]),
        ([4.5, 3.0, 1.0], [200, 350, 600]),
        ([5.0, 2.0, 1.0], [200, 400, 600]),
        ([4.5, 2.5, 1.5], [250, 350, 600]),
        ([5.0, 4.0, 3.0, 1.0], [120, 180, 250, 500]),
        ([4.5, 4.0, 3.0, 2.0], [150, 200, 250, 500]),
    ]
    for i, (c_rates, durs) in enumerate(multi_configs):
        name = "-".join(["%.1fC" % c for c in c_rates])
        for init_soc in [0.0, 0.03, 0.06, 0.1]:
            protocols.append({
                "name": "MS_%s_soc%.0f" % (name, init_soc * 100),
                "c_rates": c_rates,
                "durations_s": durs,
                "initial_soc": init_soc,
            })

    # Random protocols (constrained to feasible range) — expanded for diversity
    rng = np.random.RandomState(42)
    for i in range(400):
        n_stages = rng.randint(2, 6)
        c_rates = sorted(rng.uniform(0.3, 5.0, n_stages).tolist(), reverse=True)
        durs = rng.uniform(100, 900, n_stages).tolist()
        protocols.append({
            "name": "RAND_%d" % i,
            "c_rates": c_rates,
            "durations_s": durs,
            "initial_soc": rng.uniform(0.0, 0.15),
        })

    # Additional CC protocols at finer C-rate granularity with wider init SOC
    for c_rate in [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]:
        for init_soc in [0.0, 0.03, 0.07, 0.12, 0.18, 0.25]:
            protocols.append({
                "name": "CC_fine_%.2fC_soc%.0f" % (c_rate, init_soc * 100),
                "c_rates": [c_rate],
                "durations_s": [0.85 / c_rate * 3600 * 1.3],
                "initial_soc": init_soc,
            })

    return protocols


def save_trajectories(trajectories, output_dir, prefix):
    """Save trajectories as npz + metadata json."""
    os.makedirs(output_dir, exist_ok=True)
    features_list, actions_list, rewards_list, meta = [], [], [], []
    for traj in trajectories:
        features_list.append(traj["features"])
        actions_list.append(traj["actions"])
        rewards_list.append(traj["rewards"])
        meta.append({
            "cell_id": int(traj.get("cell_id", 0)),
            "cycle_index": int(traj.get("cycle_index", 0)),
            "soh": float(traj.get("soh", 1.0)),
            "protocol_type": str(traj.get("protocol_type", "")),
            "length": len(traj["features"]),
            "charging_time_min": float(traj.get("charging_time_min", 0)),
            "final_soc": float(traj.get("final_soc", 0)),
        })
    np.savez_compressed(
        os.path.join(output_dir, "%s_trajectories.npz" % prefix),
        features=np.array(features_list, dtype=object),
        actions=np.array(actions_list, dtype=object),
        rewards=np.array(rewards_list, dtype=object),
    )
    with open(os.path.join(output_dir, "%s_metadata.json" % prefix), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved %d trajectories -> %s/%s_*" % (len(trajectories), output_dir, prefix))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed_pybamm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    protocols = generate_all_protocols()
    print("Total protocols: %d" % len(protocols))

    trajectories = []
    t0 = time.time()

    for i, proto in enumerate(protocols):
        sol = run_pybamm_protocol(
            proto["c_rates"], proto["durations_s"], proto["initial_soc"]
        )
        if sol is not None:
            traj = protocol_to_trajectory(
                sol, proto["name"], dt=30.0,
                cell_id=i % 50, cycle_idx=i // 50,
            )
            if traj is not None:
                trajectories.append(traj)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(protocols) - i - 1) / rate
            print("  %d/%d (%.1f/s, ETA %.0fs) | valid: %d" % (
                i + 1, len(protocols), rate, eta, len(trajectories)))

    elapsed = time.time() - t0
    print("\nGenerated %d trajectories in %.1f min (%.1f/s)" % (
        len(trajectories), elapsed / 60, len(protocols) / elapsed))

    # Split
    rng = np.random.RandomState(args.seed)
    rng.shuffle(trajectories)
    n = len(trajectories)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = trajectories[:n_train]
    val = trajectories[n_train:n_train + n_val]
    test = trajectories[n_train + n_val:]
    print("Split: train=%d val=%d test=%d" % (len(train), len(val), len(test)))

    save_trajectories(train, args.output_dir, "train")
    save_trajectories(val, args.output_dir, "val")
    save_trajectories(test, args.output_dir, "test")

    # Norm stats
    all_f = np.concatenate([t["features"] for t in train], axis=0)
    _, stats = normalize_features(all_f)
    np.savez(os.path.join(args.output_dir, "norm_stats.npz"),
             mean=stats["mean"], std=stats["std"])

    # Summary stats
    times = [t["charging_time_min"] for t in trajectories]
    socs = [t["final_soc"] for t in trajectories]
    print("\nSummary:")
    print("  Charging time: %.1f +/- %.1f min (range: %.1f - %.1f)" % (
        np.mean(times), np.std(times), np.min(times), np.max(times)))
    print("  Final SOC: %.3f +/- %.3f" % (np.mean(socs), np.std(socs)))
    print("Done!")


if __name__ == "__main__":
    main()
