"""Preprocess MATR battery dataset for offline RL training.

Handles loading raw MATR data (from BatteryML or .mat files),
extracting charging cycles, computing features, and saving processed data.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .feature_utils import build_feature_matrix, normalize_features, create_windows


def load_processed_cells(processed_dir: str) -> List[Dict]:
    """Load processed cell data from BatteryML output directory.

    Args:
        processed_dir: Path to processed directory.

    Returns:
        List of cell data dictionaries.
    """
    cells = []
    pdir = Path(processed_dir)
    if not pdir.exists():
        return cells

    for f in sorted(pdir.glob("*.json")):
        try:
            with open(f, "r") as fp:
                cell = json.load(fp)
            cells.append(cell)
        except Exception:
            continue

    # Also check for .npz files
    for f in sorted(pdir.glob("*.npz")):
        try:
            data = dict(np.load(f, allow_pickle=True))
            cells.append(data)
        except Exception:
            continue

    return cells


def extract_charging_trajectories_from_synthetic(
    n_cells: int = 50,
    n_cycles_per_cell: int = 20,
    steps_per_cycle: int = 200,
    dt: float = 10.0,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Generate synthetic charging trajectories for development/testing.

    Simulates simplified battery charging with varying protocols
    and degradation.

    Args:
        n_cells: Number of cells to simulate.
        n_cycles_per_cell: Cycles per cell.
        steps_per_cycle: Steps per charging cycle.
        dt: Time step in seconds.
        seed: Random seed.

    Returns:
        List of trajectory dictionaries.
    """
    rng = np.random.RandomState(seed)
    trajectories = []

    for cell_idx in range(n_cells):
        soh = 1.0 - 0.001 * cell_idx  # Slight degradation per cell
        ambient = 25.0 + rng.randn() * 2.0

        for cycle_idx in range(n_cycles_per_cell):
            # Random charging protocol
            protocol_type = rng.choice(["cc", "multi_stage", "random"])
            T = steps_per_cycle

            voltage = np.zeros(T)
            current = np.zeros(T)
            temperature = np.zeros(T)
            soc = np.zeros(T)

            # Initial conditions
            soc[0] = rng.uniform(0.0, 0.1)
            voltage[0] = 3.0 + 1.2 * soc[0]
            temperature[0] = ambient
            r0 = 0.05 * (2.0 - soh)

            for t in range(1, T):
                # Determine current based on protocol
                if protocol_type == "cc":
                    c_rate = rng.uniform(1.0, 4.0)
                    I = c_rate * 1.1
                elif protocol_type == "multi_stage":
                    if soc[t - 1] < 0.3:
                        I = rng.uniform(3.0, 5.0)
                    elif soc[t - 1] < 0.6:
                        I = rng.uniform(2.0, 3.0)
                    else:
                        I = rng.uniform(0.5, 2.0)
                else:
                    I = rng.uniform(0.5, 5.5)

                # SOC update
                delta_soc = I * dt / (1.1 * 3600.0)
                soc[t] = min(1.0, soc[t - 1] + delta_soc)

                # Voltage
                ocv = 3.0 + 1.2 * soc[t] - 0.5 * soc[t] ** 2 + 0.3 * soc[t] ** 3
                r = r0 * (1.0 + 0.3 * soc[t])
                voltage[t] = ocv + I * r

                # Temperature
                q_gen = I ** 2 * r
                q_cool = 0.5 * (temperature[t - 1] - ambient)
                temperature[t] = temperature[t - 1] + (q_gen - q_cool) * dt / 50.0

                current[t] = I

                # Stop if voltage too high or SOC reached
                if voltage[t] > 4.25 or soc[t] >= 0.95:
                    voltage[t:] = voltage[t]
                    current[t:] = 0.0
                    temperature[t:] = temperature[t]
                    soc[t:] = soc[t]
                    break

            # Build feature matrix
            features = build_feature_matrix(
                voltage, current, temperature, soc,
                cycle_index=cycle_idx, dt=dt,
            )

            # Simple reward: charge speed - degradation stress
            rewards = np.diff(soc, prepend=soc[0]) * 100.0 - 0.01 * current ** 2 * 0.001
            # Actions
            actions = current[:, np.newaxis]

            trajectories.append({
                "cell_id": cell_idx,
                "cycle_index": cycle_idx,
                "features": features,
                "actions": actions,
                "rewards": rewards,
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "soc": soc,
                "soh": soh,
                "protocol_type": protocol_type,
            })

            # Degrade SOH
            soh -= rng.uniform(0.0001, 0.001)

    return trajectories


def save_trajectories(
    trajectories: List[Dict],
    output_dir: str,
    prefix: str = "train",
):
    """Save trajectory list as numpy arrays.

    Args:
        trajectories: List of trajectory dicts.
        output_dir: Output directory.
        prefix: File prefix (train/val/test).
    """
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_actions = []
    all_rewards = []
    metadata = []

    for traj in trajectories:
        all_features.append(traj["features"])
        all_actions.append(traj["actions"])
        all_rewards.append(traj["rewards"])
        metadata.append({
            "cell_id": int(traj.get("cell_id", 0)),
            "cycle_index": int(traj.get("cycle_index", 0)),
            "soh": float(traj.get("soh", 1.0)),
            "protocol_type": str(traj.get("protocol_type", "unknown")),
            "length": len(traj["features"]),
        })

    np.savez_compressed(
        os.path.join(output_dir, f"{prefix}_trajectories.npz"),
        features=np.array(all_features, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
    )

    with open(os.path.join(output_dir, f"{prefix}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(trajectories)} trajectories to {output_dir}/{prefix}_*")


def preprocess_matr_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    window_length: int = 64,
    stride: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    use_synthetic: bool = False,
    n_synthetic_cells: int = 50,
):
    """Full preprocessing pipeline: load data, extract features, split, save.

    Args:
        raw_dir: Raw data directory.
        processed_dir: Processed data directory.
        window_length: Window length for trajectories.
        stride: Stride for windowing.
        train_ratio: Train split ratio.
        val_ratio: Validation split ratio.
        seed: Random seed.
        use_synthetic: Force synthetic data generation.
        n_synthetic_cells: Number of synthetic cells.
    """
    rng = np.random.RandomState(seed)
    os.makedirs(processed_dir, exist_ok=True)

    # Try to load real data
    trajectories = []
    if not use_synthetic:
        cells = load_processed_cells(processed_dir)
        if cells:
            print(f"Loaded {len(cells)} processed cells from {processed_dir}")
            # TODO: Extract charging trajectories from real data
        else:
            print(f"No processed cells found in {processed_dir}, using synthetic data")
            use_synthetic = True

    if use_synthetic:
        print(f"Generating synthetic trajectories ({n_synthetic_cells} cells)...")
        trajectories = extract_charging_trajectories_from_synthetic(
            n_cells=n_synthetic_cells,
            n_cycles_per_cell=20,
            steps_per_cycle=200,
            seed=seed,
        )

    if not trajectories:
        print("WARNING: No trajectories available")
        return

    # Shuffle and split
    rng.shuffle(trajectories)
    n = len(trajectories)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_traj = trajectories[:n_train]
    val_traj = trajectories[n_train:n_train + n_val]
    test_traj = trajectories[n_train + n_val:]

    print(f"Split: train={len(train_traj)}, val={len(val_traj)}, test={len(test_traj)}")

    # Save splits
    save_trajectories(train_traj, processed_dir, "train")
    save_trajectories(val_traj, processed_dir, "val")
    save_trajectories(test_traj, processed_dir, "test")

    # Compute and save normalization stats from training data
    all_features = np.concatenate([t["features"] for t in train_traj], axis=0)
    _, stats = normalize_features(all_features)
    np.savez(
        os.path.join(processed_dir, "norm_stats.npz"),
        mean=stats["mean"],
        std=stats["std"],
    )
    print(f"Saved normalization stats to {processed_dir}/norm_stats.npz")
    print("Preprocessing complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess_matr_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        use_synthetic=args.synthetic,
        n_synthetic_cells=args.n_cells,
        seed=args.seed,
    )
