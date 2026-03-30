"""Generate offline RL dataset from PyBaMM SPMe environment.

Runs various charging protocols (CC, multi-stage CC, random) to create
a diverse offline dataset for training. Uses parallel CPU workers.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from battery_mamba_safe_rl.envs.pybamm_env import PyBaMMChargingEnv
from battery_mamba_safe_rl.datasets.feature_utils import normalize_features


def run_episode(protocol, env_kwargs, seed):
    """Run one charging episode with given protocol.

    Args:
        protocol: dict with 'type' and parameters.
        env_kwargs: kwargs for PyBaMMChargingEnv.
        seed: random seed.

    Returns:
        Trajectory dict or None if failed.
    """
    np.random.seed(seed)
    try:
        env = PyBaMMChargingEnv(**env_kwargs)
        initial_soc = protocol.get("initial_soc", np.random.uniform(0.0, 0.1))
        obs, info = env.reset(options={"initial_soc": initial_soc})

        voltages, currents, temperatures, socs = [], [], [], []
        rewards_list, actions_list = [], []
        features_list = []

        done = False
        truncated = False
        while not (done or truncated):
            # Determine current based on protocol
            ptype = protocol["type"]
            soc = info["soc"]

            if ptype == "cc":
                current = protocol["c_rate"] * env.nominal_capacity
            elif ptype == "multi_stage":
                stages = protocol["stages"]  # list of (soc_threshold, c_rate)
                current = stages[-1][1] * env.nominal_capacity
                for threshold, c_rate in stages:
                    if soc < threshold:
                        current = c_rate * env.nominal_capacity
                        break
            elif ptype == "random":
                current = np.random.uniform(0.5, protocol.get("max_c", 4.0)) * env.nominal_capacity
            elif ptype == "cc_cv":
                if info["voltage"] < protocol.get("cv_voltage", 4.15):
                    current = protocol["c_rate"] * env.nominal_capacity
                else:
                    # CV phase: reduce current to hold voltage
                    current = max(0.1 * env.nominal_capacity,
                                  protocol["c_rate"] * env.nominal_capacity * (1.0 - soc) * 2)
            else:
                current = 1.0 * env.nominal_capacity

            action = np.array([current], dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)

            voltages.append(info["voltage"])
            currents.append(info["current"])
            temperatures.append(info["temperature"])
            socs.append(info["soc"])
            rewards_list.append(reward)
            actions_list.append(current)
            features_list.append(obs[-1].copy())  # last row of windowed obs

        if len(voltages) < 5:
            return None

        features = np.array(features_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.float32)[:, np.newaxis]
        rewards = np.array(rewards_list, dtype=np.float32)

        return {
            "features": features,
            "actions": actions,
            "rewards": rewards,
            "voltage": np.array(voltages),
            "current": np.array(currents),
            "temperature": np.array(temperatures),
            "soc": np.array(socs),
            "protocol_type": ptype,
            "charging_time_min": info["total_time_s"] / 60.0,
            "final_soc": info["soc"],
            "cell_id": seed % 100,
            "cycle_index": seed // 100,
        }
    except Exception as e:
        print("Episode failed (seed=%d): %s" % (seed, e))
        return None


def generate_protocols(n_total=500):
    """Generate diverse charging protocols."""
    protocols = []

    # CC at various C-rates
    for c_rate in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        for _ in range(n_total // 50):
            protocols.append({
                "type": "cc",
                "c_rate": c_rate,
                "initial_soc": np.random.uniform(0.0, 0.1),
            })

    # Multi-stage CC
    stage_configs = [
        [(0.3, 4.0), (0.6, 3.0), (0.8, 2.0), (1.0, 1.0)],
        [(0.5, 5.0), (0.7, 3.0), (1.0, 1.5)],
        [(0.4, 3.5), (0.7, 2.5), (1.0, 1.0)],
        [(0.2, 5.0), (0.5, 4.0), (0.7, 2.0), (1.0, 0.5)],
        [(0.6, 4.5), (0.8, 2.0), (1.0, 1.0)],
    ]
    for stages in stage_configs:
        for _ in range(n_total // 25):
            protocols.append({
                "type": "multi_stage",
                "stages": stages,
                "initial_soc": np.random.uniform(0.0, 0.1),
            })

    # CC-CV
    for c_rate in [1.0, 2.0, 3.0, 4.0]:
        for cv_v in [4.1, 4.15, 4.2]:
            for _ in range(n_total // 60):
                protocols.append({
                    "type": "cc_cv",
                    "c_rate": c_rate,
                    "cv_voltage": cv_v,
                    "initial_soc": np.random.uniform(0.0, 0.1),
                })

    # Random exploration
    for _ in range(n_total // 5):
        protocols.append({
            "type": "random",
            "max_c": np.random.uniform(2.0, 5.0),
            "initial_soc": np.random.uniform(0.0, 0.15),
        })

    np.random.shuffle(protocols)
    return protocols[:n_total]


def save_trajectories_pybamm(trajectories, output_dir, prefix):
    """Save trajectory list as numpy arrays."""
    os.makedirs(output_dir, exist_ok=True)
    all_features, all_actions, all_rewards, metadata = [], [], [], []

    for traj in trajectories:
        all_features.append(traj["features"])
        all_actions.append(traj["actions"])
        all_rewards.append(traj["rewards"])
        metadata.append({
            "cell_id": int(traj.get("cell_id", 0)),
            "cycle_index": int(traj.get("cycle_index", 0)),
            "soh": 1.0,
            "protocol_type": traj.get("protocol_type", "unknown"),
            "length": len(traj["features"]),
            "charging_time_min": float(traj.get("charging_time_min", 0)),
            "final_soc": float(traj.get("final_soc", 0)),
        })

    np.savez_compressed(
        os.path.join(output_dir, "%s_trajectories.npz" % prefix),
        features=np.array(all_features, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
    )
    with open(os.path.join(output_dir, "%s_metadata.json" % prefix), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved %d trajectories to %s/%s_*" % (len(trajectories), output_dir, prefix))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed_pybamm")
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env_kwargs = {
        "max_voltage": 4.2,
        "max_temperature": 45.0,
        "max_c_rate": 6.0,
        "target_soc": 0.8,
        "max_steps": 200,
        "dt": 30.0,
        "window_length": 16,  # small for dataset generation
    }

    print("Generating %d charging protocols..." % args.n_episodes)
    protocols = generate_protocols(args.n_episodes)
    print("Protocol types: %s" % {p["type"] for p in protocols})

    print("Running %d episodes with %d workers..." % (len(protocols), args.n_workers))
    t0 = time.time()

    trajectories = []
    # Run sequentially to avoid PyBaMM multiprocessing issues
    for i, proto in enumerate(protocols):
        traj = run_episode(proto, env_kwargs, args.seed + i)
        if traj is not None:
            trajectories.append(traj)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(protocols) - i - 1) / rate
            print("  %d/%d done (%.1f ep/s, ETA %.0fs) | valid: %d" % (
                i + 1, len(protocols), rate, eta, len(trajectories)))

    elapsed = time.time() - t0
    print("\nGenerated %d valid trajectories in %.1f min" % (len(trajectories), elapsed / 60))

    # Split
    np.random.seed(args.seed)
    np.random.shuffle(trajectories)
    n = len(trajectories)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train = trajectories[:n_train]
    val = trajectories[n_train:n_train + n_val]
    test = trajectories[n_train + n_val:]

    print("Split: train=%d, val=%d, test=%d" % (len(train), len(val), len(test)))

    save_trajectories_pybamm(train, args.output_dir, "train")
    save_trajectories_pybamm(val, args.output_dir, "val")
    save_trajectories_pybamm(test, args.output_dir, "test")

    # Norm stats
    all_feat = np.concatenate([t["features"] for t in train], axis=0)
    _, stats = normalize_features(all_feat)
    np.savez(os.path.join(args.output_dir, "norm_stats.npz"),
             mean=stats["mean"], std=stats["std"])

    # Summary
    times = [t["charging_time_min"] for t in trajectories]
    socs = [t["final_soc"] for t in trajectories]
    print("\nDataset summary:")
    print("  Charging time: %.1f +/- %.1f min" % (np.mean(times), np.std(times)))
    print("  Final SOC: %.3f +/- %.3f" % (np.mean(socs), np.std(socs)))
    print("Done!")


if __name__ == "__main__":
    main()
