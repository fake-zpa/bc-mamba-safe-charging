"""Build offline RL dataset from preprocessed trajectories.

Converts trajectory data into (s, a, r, s', done) tuples suitable
for offline RL training with windowed observations.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .feature_utils import normalize_features, create_windows


class OfflineRLDataset(Dataset):
    """PyTorch Dataset for offline RL with windowed battery observations.

    Each sample contains:
        - obs: (window_length, obs_dim) current observation window
        - action: (action_dim,) action taken
        - reward: scalar reward
        - next_obs: (window_length, obs_dim) next observation window
        - done: bool terminal flag
        - info: dict with metadata
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_length: int = 64,
        normalize: bool = True,
        norm_stats_path: Optional[str] = None,
        lazy_load: bool = False,
    ):
        """Initialize offline RL dataset.

        Args:
            data_dir: Path to processed data directory.
            split: Data split (train/val/test).
            window_length: Observation window length.
            normalize: Whether to normalize features.
            norm_stats_path: Path to normalization stats.
            lazy_load: Whether to lazy-load data.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_length = window_length
        self.normalize = normalize
        self.lazy_load = lazy_load

        # Load trajectories
        traj_path = self.data_dir / f"{split}_trajectories.npz"
        if traj_path.exists():
            data = np.load(traj_path, allow_pickle=True)
            self.features_list = list(data["features"])
            self.actions_list = list(data["actions"])
            self.rewards_list = list(data["rewards"])
        else:
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

        # Load metadata
        meta_path = self.data_dir / f"{split}_metadata.json"
        self.metadata = []
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        # Load normalization stats
        self.norm_stats = None
        if normalize:
            stats_path = norm_stats_path or str(self.data_dir / "norm_stats.npz")
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                std = stats["std"].copy()
                std = np.maximum(std, 0.01)  # Clamp std to prevent division by near-zero
                self.norm_stats = {"mean": stats["mean"], "std": std}

        # Build index: (traj_idx, step_idx) pairs
        self.index = []
        for traj_idx, features in enumerate(self.features_list):
            T = len(features)
            for step_idx in range(max(1, T - 1)):
                self.index.append((traj_idx, step_idx))

    def __len__(self) -> int:
        return len(self.index)

    def _get_window(self, features: np.ndarray, step: int) -> np.ndarray:
        """Extract observation window ending at step.

        Args:
            features: Full trajectory features (T, obs_dim).
            step: Current step index.

        Returns:
            Window array of shape (window_length, obs_dim).
        """
        T, D = features.shape
        start = max(0, step - self.window_length + 1)
        window = features[start:step + 1]

        # Pad if needed
        if len(window) < self.window_length:
            pad = np.zeros((self.window_length - len(window), D), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        if self.norm_stats is not None:
            window = (window - self.norm_stats["mean"]) / self.norm_stats["std"]

        return window.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, step_idx = self.index[idx]
        features = self.features_list[traj_idx]
        actions = self.actions_list[traj_idx]
        rewards = self.rewards_list[traj_idx]
        T = len(features)

        obs = self._get_window(features, step_idx)
        action = actions[step_idx].astype(np.float32)
        if action.ndim == 0:
            action = np.array([action], dtype=np.float32)
        reward = float(rewards[step_idx])

        next_step = min(step_idx + 1, T - 1)
        next_obs = self._get_window(features, next_step)
        done = float(next_step >= T - 1)

        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": torch.from_numpy(next_obs),
            "done": torch.tensor(done, dtype=torch.float32),
        }


def build_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 256,
    window_length: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    normalize: bool = True,
) -> DataLoader:
    """Build DataLoader for offline RL training.

    Args:
        data_dir: Path to processed data directory.
        split: Data split.
        batch_size: Batch size.
        window_length: Observation window length.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for CUDA transfers.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Prefetch factor per worker.
        normalize: Whether to normalize features.

    Returns:
        DataLoader instance.
    """
    dataset = OfflineRLDataset(
        data_dir=data_dir,
        split=split,
        window_length=window_length,
        normalize=normalize,
    )

    shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=shuffle,
    )

    return loader


def generate_synthetic_offline_dataset(
    output_dir: str,
    n_cells: int = 50,
    n_cycles: int = 20,
    steps: int = 200,
    seed: int = 42,
):
    """Generate and save a synthetic offline RL dataset.

    Args:
        output_dir: Output directory.
        n_cells: Number of cells.
        n_cycles: Cycles per cell.
        steps: Steps per cycle.
        seed: Random seed.
    """
    from .preprocess_matr import (
        extract_charging_trajectories_from_synthetic,
        save_trajectories,
    )

    print(f"Generating synthetic dataset: {n_cells} cells x {n_cycles} cycles...")
    trajectories = extract_charging_trajectories_from_synthetic(
        n_cells=n_cells,
        n_cycles_per_cell=n_cycles,
        steps_per_cycle=steps,
        seed=seed,
    )

    # Split
    rng = np.random.RandomState(seed)
    rng.shuffle(trajectories)
    n = len(trajectories)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    save_trajectories(trajectories[:n_train], output_dir, "train")
    save_trajectories(trajectories[n_train:n_train + n_val], output_dir, "val")
    save_trajectories(trajectories[n_train + n_val:], output_dir, "test")

    # Normalization stats
    all_features = np.concatenate(
        [t["features"] for t in trajectories[:n_train]], axis=0
    )
    _, stats = normalize_features(all_features)
    np.savez(
        os.path.join(output_dir, "norm_stats.npz"),
        mean=stats["mean"],
        std=stats["std"],
    )
    print(f"Synthetic dataset saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic_offline_dataset(
        output_dir=args.output_dir,
        n_cells=args.n_cells,
        seed=args.seed,
    )
