"""Latent world model for battery charging dynamics prediction.

Predicts next latent state, observations, and rewards from current
latent state and action. Supports ensemble uncertainty estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SingleDynamicsModel(nn.Module):
    """Single deterministic dynamics model: (z_t, a_t) -> z_{t+1}, obs_{t+1}."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        obs_dim: int = 10,
        predict_reward: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize single dynamics model.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            obs_dim: Observation dimension for prediction.
            predict_reward: Whether to predict reward.
            dropout: Dropout rate.
        """
        super().__init__()
        in_dim = latent_dim + action_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Next latent prediction
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

        # Next observation prediction
        self.obs_head = nn.Linear(hidden_dim, obs_dim)

        # Reward prediction
        self.predict_reward = predict_reward
        if predict_reward:
            self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict next state.

        Args:
            z: Current latent state (B, latent_dim).
            action: Current action (B, action_dim).

        Returns:
            Dict with 'next_latent', 'next_obs', optionally 'reward'.
        """
        x = torch.cat([z, action], dim=-1)
        h = self.backbone(x)

        result = {
            "next_latent": self.latent_head(h),
            "next_obs": self.obs_head(h),
        }
        if self.predict_reward:
            result["reward"] = self.reward_head(h)

        return result


class LatentWorldModel(nn.Module):
    """Ensemble latent world model with uncertainty estimation.

    Uses an ensemble of dynamics models to provide epistemic
    uncertainty estimates for safe planning.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        obs_dim: int = 10,
        n_ensemble: int = 5,
        predict_reward: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize ensemble world model.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            obs_dim: Observation dimension.
            n_ensemble: Number of ensemble members.
            predict_reward: Whether to predict reward.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_ensemble = n_ensemble
        self.latent_dim = latent_dim

        self.models = nn.ModuleList([
            SingleDynamicsModel(
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                obs_dim=obs_dim,
                predict_reward=predict_reward,
                dropout=dropout,
            )
            for _ in range(n_ensemble)
        ])

    def predict_next_latent(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state with uncertainty.

        Args:
            z: Current latent (B, latent_dim).
            action: Action (B, action_dim).

        Returns:
            Tuple of (mean_next_latent, std_next_latent), each (B, latent_dim).
        """
        preds = []
        for model in self.models:
            out = model(z, action)
            preds.append(out["next_latent"])

        preds = torch.stack(preds, dim=0)  # (E, B, latent_dim)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std

    def predict_next_observation(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next observation with uncertainty.

        Args:
            z: Current latent (B, latent_dim).
            action: Action (B, action_dim).

        Returns:
            Tuple of (mean_next_obs, std_next_obs).
        """
        preds = []
        for model in self.models:
            out = model(z, action)
            preds.append(out["next_obs"])

        preds = torch.stack(preds, dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std

    def rollout(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Multi-step rollout using mean ensemble prediction.

        Args:
            z_init: Initial latent state (B, latent_dim).
            actions: Action sequence (B, H, action_dim).

        Returns:
            Dict with 'latent_traj' (B, H+1, latent_dim),
            'obs_traj' (B, H, obs_dim),
            'uncertainty_traj' (B, H, latent_dim).
        """
        B, H, _ = actions.shape
        z = z_init
        latent_traj = [z]
        obs_traj = []
        uncertainty_traj = []

        for t in range(H):
            a = actions[:, t, :]
            z_next_mean, z_next_std = self.predict_next_latent(z, a)
            obs_mean, _ = self.predict_next_observation(z, a)

            latent_traj.append(z_next_mean)
            obs_traj.append(obs_mean)
            uncertainty_traj.append(z_next_std)

            z = z_next_mean

        return {
            "latent_traj": torch.stack(latent_traj, dim=1),
            "obs_traj": torch.stack(obs_traj, dim=1),
            "uncertainty_traj": torch.stack(uncertainty_traj, dim=1),
        }

    def uncertainty_score(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar uncertainty score for state-action pair.

        Args:
            z: Latent state (B, latent_dim).
            action: Action (B, action_dim).

        Returns:
            Uncertainty score (B, 1).
        """
        _, std = self.predict_next_latent(z, action)
        return std.mean(dim=-1, keepdim=True)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor, model_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through a specific ensemble member.

        Args:
            z: Latent state (B, latent_dim).
            action: Action (B, action_dim).
            model_idx: Ensemble member index.

        Returns:
            Prediction dictionary.
        """
        return self.models[model_idx](z, action)

    def loss(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        target_z: torch.Tensor,
        target_obs: Optional[torch.Tensor] = None,
        target_reward: Optional[torch.Tensor] = None,
        latent_weight: float = 1.0,
        obs_weight: float = 0.5,
        reward_weight: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """Compute ensemble training loss.

        Each member is trained independently on the full batch.

        Args:
            z: Current latent (B, latent_dim).
            action: Action (B, action_dim).
            target_z: Target next latent (B, latent_dim).
            target_obs: Target next observation (B, obs_dim).
            target_reward: Target reward (B, 1).
            latent_weight: Weight for latent prediction loss.
            obs_weight: Weight for observation prediction loss.
            reward_weight: Weight for reward prediction loss.

        Returns:
            Dict with 'total_loss' and per-component losses.
        """
        total = torch.tensor(0.0, device=z.device)
        losses = {}

        for i, model in enumerate(self.models):
            pred = model(z, action)
            l_latent = F.mse_loss(pred["next_latent"], target_z)
            loss_i = latent_weight * l_latent

            if target_obs is not None:
                l_obs = F.mse_loss(pred["next_obs"], target_obs)
                loss_i = loss_i + obs_weight * l_obs

            if target_reward is not None and "reward" in pred:
                l_reward = F.mse_loss(pred["reward"], target_reward)
                loss_i = loss_i + reward_weight * l_reward

            total = total + loss_i

        total = total / self.n_ensemble
        losses["total_loss"] = total
        return losses
