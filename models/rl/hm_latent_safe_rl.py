"""HM-LatentSafeRL: Health-Mamba Latent Safe Offline Reinforcement Learning.

Main method combining Mamba encoder, world model, risk head, degradation head,
and safety-constrained actor-critic into a unified framework.

Key design:
- Mamba encoder produces latent health state z_t shared by ALL components
- Actor, critic, world model, risk head all operate on z_t
- Joint loss: RL + health prediction + latent dynamics + risk prediction
- Safety layer projects actions based on risk/world model predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..dynamics.latent_world_model import LatentWorldModel
from ..heads.degradation_head import DegradationHead
from ..heads.risk_head import RiskHead
from ..safety.action_projection import SafetyLayer


class SafeActor(nn.Module):
    """Tanh-squashed Gaussian actor with safety layer integration.

    Outputs continuous charging current, then passes through safety layer.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        max_action: float = 6.0,
    ):
        """Initialize SafeActor.

        Args:
            latent_dim: Input latent dimension.
            action_dim: Output action dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of hidden layers.
            dropout: Dropout rate.
            log_std_min: Min log std.
            log_std_max: Max log std.
            max_action: Maximum action value.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action

        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, z: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            z: Latent state (B, latent_dim).
            deterministic: Use mean action.

        Returns:
            Tuple of (action, log_prob).
        """
        h = self.backbone(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            raw = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()

        action = torch.sigmoid(raw) * self.max_action
        log_prob = torch.distributions.Normal(mean, std).log_prob(raw)
        log_prob = log_prob - torch.log(
            self.max_action * torch.sigmoid(raw) * (1 - torch.sigmoid(raw)) + 1e-6
        )
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def forward_log_prob(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of a given action under current policy.

        Args:
            z: Latent state (B, latent_dim).
            action: Action to evaluate (B, action_dim).

        Returns:
            Log probability (B, 1).
        """
        h = self.backbone(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        # Inverse sigmoid to recover raw action
        action_clipped = action.clamp(1e-6, self.max_action - 1e-6) / self.max_action
        raw = torch.log(action_clipped / (1.0 - action_clipped + 1e-6) + 1e-6)
        log_prob = torch.distributions.Normal(mean, std).log_prob(raw)
        log_prob = log_prob - torch.log(
            self.max_action * torch.sigmoid(raw) * (1 - torch.sigmoid(raw)) + 1e-6
        )
        return log_prob.sum(-1, keepdim=True)

    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        """Get deterministic action."""
        with torch.no_grad():
            action, _ = self.forward(z, deterministic=True)
        return action


class ConservativeCritic(nn.Module):
    """Double Q critic with conservative regularization support."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize ConservativeCritic."""
        super().__init__()

        def build_q():
            layers = []
            in_dim = latent_dim + action_dim
            for _ in range(n_layers):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.q1 = build_q()
        self.q2 = build_q()

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(z, action)
        return torch.min(q1, q2)


class HMLatentSafeRL(nn.Module):
    """HM-LatentSafeRL: unified framework for safe offline RL with Mamba encoder.

    Components (all share latent state z_t from Mamba encoder):
    - encoder: Mamba-based health state encoder
    - actor: safe policy with safety layer
    - critic: conservative double-Q critic
    - world_model: ensemble latent dynamics model
    - degradation_head: health/degradation predictor
    - risk_head: constraint violation predictor
    - safety_layer: action projection for safety
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        obs_dim: int = 10,
        n_ensemble: int = 5,
        max_action: float = 6.0,
        safety_mode: str = "uncertainty_aware",
        risk_threshold: float = 0.3,
        gamma: float = 0.99,
        tau: float = 0.005,
        conservative_weight: float = 1.0,
        device: str = "cuda",
    ):
        """Initialize HM-LatentSafeRL.

        Args:
            encoder: Pretrained or trainable Mamba encoder.
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden dimension for all heads.
            obs_dim: Observation dimension.
            n_ensemble: World model ensemble size.
            max_action: Maximum action value.
            safety_mode: Safety layer mode.
            risk_threshold: Risk threshold for safety.
            gamma: Discount factor.
            tau: Target network update rate.
            conservative_weight: Weight for conservative Q penalty.
            device: Compute device.
        """
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.conservative_weight = conservative_weight
        self.max_action = max_action
        self.device = device

        self.actor = SafeActor(
            latent_dim=latent_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, max_action=max_action,
        )
        self.critic = ConservativeCritic(
            latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.critic_target = ConservativeCritic(
            latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # IQL Value network V(z) for expectile regression
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.expectile = 0.9  # Higher expectile focuses V(z) on upper tail of Q distribution
        self.iql_temperature = 50.0  # Higher temp = less conservative, more willing to select high C-rate actions

        self.world_model = LatentWorldModel(
            latent_dim=latent_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, obs_dim=obs_dim, n_ensemble=n_ensemble,
        )
        self.degradation_head = DegradationHead(
            latent_dim=latent_dim, hidden_dim=hidden_dim,
        )
        self.risk_head = RiskHead(
            latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim // 2,
        )
        self.safety_layer = SafetyLayer(
            mode=safety_mode,
            risk_threshold=risk_threshold,
            max_action=max_action,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation through Mamba to get latent state z_t.

        Args:
            obs: Windowed observation (B, L, obs_dim).

        Returns:
            Latent state z_t (B, latent_dim).
        """
        return self.encoder(obs)

    def get_safe_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get safety-projected action for deployment.

        Args:
            obs: Windowed observation (B, L, obs_dim).
            deterministic: Use deterministic policy.

        Returns:
            Tuple of (safe_action, info_dict).
        """
        z = self.encode(obs)
        raw_action, log_prob = self.actor(z, deterministic=deterministic)

        # Risk assessment
        risk_info = self.risk_head(z, raw_action)

        # World model uncertainty
        uncertainty = self.world_model.uncertainty_score(z, raw_action)

        # Safety projection
        safe_action = self.safety_layer(
            raw_action, risk_info, uncertainty,
        )

        info = {
            "raw_action": raw_action,
            "risk": risk_info["overall_risk"],
            "uncertainty": uncertainty,
            "z": z,
        }
        return safe_action, info

    def compute_losses(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        health_weight: float = 0.5,
        dynamics_weight: float = 0.3,
        risk_weight: float = 0.5,
        n_random_actions: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses for joint training.

        Args:
            obs: Current observation (B, L, obs_dim).
            action: Action (B, action_dim).
            reward: Reward (B, 1).
            next_obs: Next observation (B, L, obs_dim).
            done: Done flag (B, 1).
            health_weight: Weight for health prediction loss.
            dynamics_weight: Weight for dynamics loss.
            risk_weight: Weight for risk loss.
            n_random_actions: Random actions for conservative penalty.

        Returns:
            Dict of all loss components.
        """
        B = obs.shape[0]

        # Encode current and next states through shared Mamba encoder
        z = self.encode(obs)
        with torch.no_grad():
            next_z = self.encode(next_obs)

        # ---- IQL: Value loss (expectile regression) ----
        # V(z) trained via expectile regression on Q(z,a) - V(z)
        with torch.no_grad():
            q1_data, q2_data = self.critic_target(z, action)
            q_data = torch.min(q1_data, q2_data)
        v = self.value_net(z)
        diff = q_data - v
        expectile_weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        value_loss = (expectile_weight * diff ** 2).mean()

        # ---- IQL: Critic loss ----
        # Q(z,a) trained on r + gamma * V(z') (no actor in target!)
        with torch.no_grad():
            next_v = self.value_net(next_z)  # (B, 1)
            # Ensure reward and done have correct shape for broadcasting
            r = reward.view(-1, 1) if reward.dim() == 1 else reward
            d = done.view(-1, 1) if done.dim() == 1 else done
            target_q = r + self.gamma * (1 - d) * next_v
        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # ---- IQL: Actor loss (advantage-weighted regression) ----
        with torch.no_grad():
            advantage = q_data - v
            weights = torch.exp(advantage * self.iql_temperature).clamp(max=100.0)
        new_action, log_prob = self.actor(z.detach())
        # Advantage-weighted BC: maximize log_prob of data actions weighted by advantage
        actor_log_prob = self.actor.forward_log_prob(z.detach(), action)
        actor_loss = -(weights * actor_log_prob).mean()

        # Risk penalty for actor
        risk_info = self.risk_head(z, new_action)
        actor_risk_penalty = risk_info["overall_risk"].mean()
        actor_loss = actor_loss + risk_weight * actor_risk_penalty

        # ---- World model loss ----
        wm_losses = self.world_model.loss(
            z.detach(), action, next_z,
            target_obs=next_obs[:, -1, :] if next_obs.dim() == 3 else None,
        )
        dynamics_loss = wm_losses["total_loss"]

        # ---- Health/degradation loss (self-supervised) ----
        deg_preds = self.degradation_head(z)
        # Use SOH proxy from observations as pseudo-label
        if obs.dim() == 3:
            soh_target = obs[:, -1, 9:10]  # degradation_proxy
        else:
            soh_target = torch.zeros(B, 1, device=z.device)
        health_loss = F.mse_loss(deg_preds["soh_proxy"], torch.ones(B, 1, device=z.device) * 0.9) + \
                      F.mse_loss(deg_preds["capacity_fade_proxy"], soh_target.abs())

        # ---- Risk prediction loss (self-supervised) ----
        # Use simple heuristic: high voltage/temp observations indicate violations
        if obs.dim() == 3:
            v_obs = obs[:, -1, 0]  # voltage
            t_obs = obs[:, -1, 2]  # temperature
            pseudo_labels = torch.stack([
                (v_obs > 4.1).float(),
                (t_obs > 40.0).float(),
                torch.zeros(B, device=z.device),
                torch.zeros(B, device=z.device),
            ], dim=-1)
        else:
            pseudo_labels = torch.zeros(B, 4, device=z.device)
        risk_loss = self.risk_head.loss(risk_info, pseudo_labels)

        # ---- Total loss ----
        total_loss = (
            value_loss
            + critic_loss
            + actor_loss
            + dynamics_weight * dynamics_loss
            + health_weight * health_loss
            + risk_weight * risk_loss
        )

        return {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "dynamics_loss": dynamics_loss,
            "health_loss": health_loss,
            "risk_loss": risk_loss,
            "q1_mean": q1.mean().detach(),
            "q2_mean": q2.mean().detach(),
            "v_mean": v.mean().detach(),
            "advantage_mean": advantage.mean().detach(),
            "risk_mean": risk_info["overall_risk"].mean().detach(),
        }

    def soft_update_target(self):
        """Soft update target critic network."""
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def get_all_parameters(self) -> list:
        """Get all trainable parameters for optimizer."""
        params = []
        params += list(self.encoder.parameters())
        params += list(self.actor.parameters())
        params += list(self.critic.parameters())
        params += list(self.value_net.parameters())
        params += list(self.world_model.parameters())
        params += list(self.degradation_head.parameters())
        params += list(self.risk_head.parameters())
        return params
