"""Parse MATR batch .mat files (HDF5/v7.3) into charging trajectories.

MATR dataset structure (Matlab v7.3 = HDF5):
  batch.cells[i].cycles[j].{V, I, t, T, Qc, Qd, ...}
  batch.cells[i].summary.{cycle, QDischarge, IR, Tmax, Tavg, Tmin, chargetime}
"""
import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from battery_mamba_safe_rl.datasets.feature_utils import (
    build_feature_matrix, normalize_features
)


def load_matr_batch(mat_path: str, max_cells: int = 999) -> List[Dict]:
    """Load a MATR batch .mat file and extract cell data.

    Args:
        mat_path: Path to .mat file.
        max_cells: Maximum number of cells to load.

    Returns:
        List of cell data dictionaries.
    """
    print(f"Loading {mat_path}...")
    cells = []

    with h5py.File(mat_path, "r") as f:
        # The top-level key is 'batch'; skip HDF5 internal keys like #refs#
        batch_key = "batch"
        if batch_key not in f:
            for k in f.keys():
                if not k.startswith("#"):
                    batch_key = k
                    break
        batch = f[batch_key]
        print(f"  Top-level key: {batch_key}")
        print(f"  Batch keys: {list(batch.keys())}")

        # Get references to individual cells
        # In MATR HDF5, batch.cycles is an array of object references
        if "summary" in batch:
            summary_refs = batch["summary"]
        else:
            summary_refs = None

        if "cycles" in batch:
            cycles_refs = batch["cycles"]
        else:
            cycles_refs = None

        # Try to get the number of cells
        # The structure varies by batch; try common patterns
        n_cells = 0
        if "barcode" in batch:
            n_cells = batch["barcode"].shape[0]
        elif "policy" in batch:
            n_cells = batch["policy"].shape[0]
        elif summary_refs is not None:
            n_cells = summary_refs.shape[0]

        if n_cells == 0:
            # Try to infer from first available array
            for k in batch.keys():
                if hasattr(batch[k], "shape") and len(batch[k].shape) > 0:
                    n_cells = batch[k].shape[0]
                    break

        n_cells = min(n_cells, max_cells)
        print(f"  Number of cells: {n_cells}")

        for ci in range(n_cells):
            try:
                cell_data = _extract_cell(f, batch, ci)
                if cell_data is not None and len(cell_data.get("charge_cycles", [])) > 0:
                    cells.append(cell_data)
                    if len(cells) % 10 == 0:
                        print(f"    Loaded {len(cells)} cells...")
            except Exception as e:
                print(f"    Error loading cell {ci}: {e}")
                continue

    print(f"  Total valid cells loaded: {len(cells)}")
    return cells


def _deref(f, ref):
    """Dereference an HDF5 object reference."""
    return f[ref]


def _extract_cell(f: h5py.File, batch, ci: int) -> Optional[Dict]:
    """Extract data for a single cell from the MATR batch.

    Args:
        f: HDF5 file handle.
        batch: Batch group.
        ci: Cell index.

    Returns:
        Cell data dictionary or None.
    """
    cell = {}

    # Extract summary data
    if "summary" in batch:
        summary_ref = batch["summary"][ci, 0]
        summary = f[summary_ref]

        # Extract available summary fields
        for key in ["cycle", "QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]:
            if key in summary:
                try:
                    cell[f"summary_{key}"] = np.array(summary[key]).flatten()
                except Exception:
                    pass

    # Extract cycle_life directly from batch
    if "cycle_life" in batch:
        try:
            cl_ref = batch["cycle_life"][ci, 0]
            cell["cycle_life"] = int(np.array(f[cl_ref]).flatten()[0])
        except Exception:
            cell["cycle_life"] = 500

    # Extract cycle-level data
    if "cycles" in batch:
        cycles_ref = batch["cycles"][ci, 0]
        cycles_group = f[cycles_ref]

        charge_cycles = []
        n_cycles = 0

        # Determine number of cycles
        for key in ["V", "I", "t", "T", "Qc"]:
            if key in cycles_group:
                n_cycles = max(n_cycles, cycles_group[key].shape[0])
                break

        # Skip cycle 0 (empty/dummy in MATR), limit to 500 cycles
        for cyc_idx in range(1, min(n_cycles, 500)):
            try:
                cyc_data = {}
                for key, out_name in [("V", "voltage"), ("I", "current"), ("t", "time"),
                                       ("T", "temperature"), ("Qc", "charge_capacity"),
                                       ("Qd", "discharge_capacity")]:
                    if key in cycles_group:
                        ref = cycles_group[key][cyc_idx, 0]
                        arr = np.array(f[ref]).flatten()
                        cyc_data[out_name] = arr

                # Extract charging portion (positive current)
                if "current" in cyc_data and len(cyc_data["current"]) > 10:
                    I = cyc_data["current"]
                    charge_mask = I > 0.01
                    n_charge = int(np.sum(charge_mask))
                    if n_charge > 10:
                        for k in list(cyc_data.keys()):
                            if len(cyc_data[k]) == len(I):
                                cyc_data[k] = cyc_data[k][charge_mask]
                        cyc_data["cycle_index"] = cyc_idx
                        charge_cycles.append(cyc_data)
            except Exception:
                continue

        cell["charge_cycles"] = charge_cycles
        cell["n_charge_cycles"] = len(charge_cycles)

    # Fallback cycle_life from summary
    if "cycle_life" not in cell and "summary_QDischarge" in cell:
        qdis = cell["summary_QDischarge"]
        if len(qdis) > 1 and qdis[1] > 0:
            fade = qdis / qdis[1]
            eol = np.where(fade < 0.8)[0]
            cell["cycle_life"] = int(eol[0]) if len(eol) > 0 else len(qdis)
        else:
            cell["cycle_life"] = len(qdis)

    # Extract policy if available
    if "policy_readable" in batch:
        try:
            ref = batch["policy_readable"][ci, 0]
            policy_arr = np.array(f[ref]).flatten()
            cell["policy"] = "".join(chr(int(c)) for c in policy_arr if 32 <= c < 127)
        except Exception:
            cell["policy"] = "unknown"
    elif "policy" in batch:
        try:
            ref = batch["policy"][ci, 0]
            policy_arr = np.array(f[ref]).flatten()
            cell["policy"] = "".join(chr(int(c)) for c in policy_arr if 32 <= c < 127)
        except Exception:
            cell["policy"] = "unknown"

    cell["cell_index"] = ci
    return cell


def convert_to_trajectories(
    cells: List[Dict],
    dt: float = 10.0,
    nominal_capacity: float = 1.1,
) -> List[Dict[str, np.ndarray]]:
    """Convert parsed cell data into RL-ready trajectories.

    Args:
        cells: List of parsed cell data.
        dt: Target time step for resampling (seconds).
        nominal_capacity: Nominal capacity in Ah.

    Returns:
        List of trajectory dictionaries.
    """
    trajectories = []

    for cell in cells:
        ci = cell.get("cell_index", 0)
        cycle_life = cell.get("cycle_life", 500)
        soh_base = 1.0

        for cyc in cell.get("charge_cycles", []):
            cyc_idx = cyc.get("cycle_index", 0)

            V = cyc.get("voltage", np.array([]))
            I = cyc.get("current", np.array([]))
            T = cyc.get("temperature", np.array([]))
            t = cyc.get("time", np.array([]))
            Qc = cyc.get("charge_capacity", np.array([]))

            if len(V) < 20 or len(I) < 20:
                continue

            # Ensure arrays same length
            L = min(len(V), len(I), len(T) if len(T) > 0 else len(V))
            V = V[:L]
            I = I[:L]
            if len(T) >= L:
                T = T[:L]
            else:
                T = np.full(L, 25.0)  # Default temperature

            # Compute SOC from charge capacity
            if len(Qc) >= L:
                Qc = Qc[:L]
                soc = np.clip(Qc / nominal_capacity, 0, 1)
            else:
                # Estimate SOC from current integration
                if len(t) >= L:
                    dt_arr = np.diff(t[:L], prepend=t[0])
                    soc = np.cumsum(np.abs(I) * dt_arr / 3600.0) / nominal_capacity
                    soc = np.clip(soc, 0, 1)
                else:
                    soc = np.linspace(0, 0.8, L)

            # Resample to uniform time steps if needed (simple decimation)
            if len(V) > 500:
                step = max(1, len(V) // 500)
                V = V[::step]
                I = I[::step]
                T = T[::step]
                soc = soc[::step]

            # SOH degradation based on cycle index / cycle life
            soh = max(0.5, soh_base - cyc_idx * 0.2 / max(cycle_life, 1))

            # Build feature matrix
            features = build_feature_matrix(V, I, T, soc, cycle_index=cyc_idx, dt=dt)

            # Rewards: SOC increase - current stress
            rewards = np.diff(soc, prepend=soc[0]) * 100.0 - 0.001 * I ** 2
            actions = I[:, np.newaxis]

            trajectories.append({
                "cell_id": ci,
                "cycle_index": cyc_idx,
                "features": features,
                "actions": actions,
                "rewards": rewards,
                "voltage": V,
                "current": I,
                "temperature": T,
                "soc": soc,
                "soh": soh,
                "protocol_type": cell.get("policy", "unknown"),
            })

    return trajectories


def build_matr_dataset(
    raw_dir: str,
    processed_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_cells: int = 999,
):
    """Full pipeline: parse .mat -> trajectories -> train/val/test split.

    Args:
        raw_dir: Directory containing .mat files.
        processed_dir: Output directory.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        seed: Random seed.
        max_cells: Max cells to load per batch.
    """
    from battery_mamba_safe_rl.datasets.preprocess_matr import save_trajectories

    rng = np.random.RandomState(seed)
    os.makedirs(processed_dir, exist_ok=True)

    all_trajectories = []

    # Find all .mat files
    mat_files = sorted(Path(raw_dir).glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files in {raw_dir}")

    for mat_path in mat_files:
        print(f"\nProcessing: {mat_path.name}")
        cells = load_matr_batch(str(mat_path), max_cells=max_cells)
        trajs = convert_to_trajectories(cells)
        print(f"  Extracted {len(trajs)} charging trajectories")
        all_trajectories.extend(trajs)

    if not all_trajectories:
        print("ERROR: No trajectories extracted!")
        return

    print(f"\nTotal trajectories: {len(all_trajectories)}")

    # Shuffle and split
    rng.shuffle(all_trajectories)
    n = len(all_trajectories)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_traj = all_trajectories[:n_train]
    val_traj = all_trajectories[n_train:n_train + n_val]
    test_traj = all_trajectories[n_train + n_val:]

    print(f"Split: train={len(train_traj)}, val={len(val_traj)}, test={len(test_traj)}")

    save_trajectories(train_traj, processed_dir, "train")
    save_trajectories(val_traj, processed_dir, "val")
    save_trajectories(test_traj, processed_dir, "test")

    # Normalization stats from training data
    all_features = np.concatenate([t["features"] for t in train_traj], axis=0)
    _, stats = normalize_features(all_features)
    np.savez(
        os.path.join(processed_dir, "norm_stats.npz"),
        mean=stats["mean"],
        std=stats["std"],
    )
    print(f"\nDataset saved to {processed_dir}")
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--max_cells", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_matr_dataset(args.raw_dir, args.processed_dir, seed=args.seed, max_cells=args.max_cells)
