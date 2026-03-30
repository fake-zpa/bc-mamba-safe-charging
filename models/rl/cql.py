"""Conservative Q-Learning (CQL) for offline RL battery charging.

Implements CQL with shared Mamba encoder latent state.
Penalizes Q-values for out-of-distribution actions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class CQLCritic(nn.Module):
    """Double Q-network critic for CQL."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize CQL critic.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of hidden layers.
            dropout: Dropout rate.
        """
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

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both critics.

        Args:
            z: Latent state (B, latent_dim).
            action: Action (B, action_dim).

        Returns:
            Tuple of (Q1, Q2), each (B, 1).
        """
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute minimum Q-value across critics.

        Args:
            z: Latent state (B, latent_dim).
            action: Action (B, action_dim).

        Returns:
            Minimum Q-value (B, 1).
        """
        q1, q2 = self.forward(z, action)
        return torch.min(q1, q2)


class CQLActor(nn.Module):
    """Tanh-squashed Gaussian policy for CQL."""

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
        """Initialize CQL actor.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of hidden layers.
            dropout: Dropout rate.
            log_std_min: Minimum log standard deviation.
            log_std_max: Maximum log standard deviation.
            max_action: Maximum action value.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action
        self.action_dim = action_dim

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
        """Sample action and compute log probability.

        Args:
            z: Latent state (B, latent_dim).
            deterministic: Use mean action without sampling.

        Returns:
            Tuple of (action (B, action_dim), log_prob (B, 1)).
        """
        h = self.backbone(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            raw_action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()

        # Tanh squashing -> [0, max_action]
        action = torch.sigmoid(raw_action) * self.max_action

        # Log probability with tanh correction
        log_prob = torch.distributions.Normal(mean, std).log_prob(raw_action)
        log_prob = log_prob - torch.log(
            self.max_action * torch.sigmoid(raw_action) * (1 - torch.sigmoid(raw_action)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        """Get deterministic action for evaluation.

        Args:
            z: Latent state (B, latent_dim).

        Returns:
            Action (B, action_dim).
        """
        with torch.no_grad():
            action, _ = self.forward(z, deterministic=True)
        return action


class CQL:
    """CQL algorithm combining actor and critic with conservative regularization."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 1,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 5.0,
        n_action_samples: int = 10,
        max_action: float = 6.0,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        device: str = "cuda",
    ):
        """Initialize CQL algorithm.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            gamma: Discount factor.
            tau: Target network soft update rate.
            alpha: CQL conservative penalty weight.
            n_action_samples: Number of random actions for CQL penalty.
            max_action: Maximum action value.
            lr_actor: Actor learning rate.
            lr_critic: Critic learning rate.
            device: Compute device.
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.n_action_samples = n_action_samples
        self.max_action = max_action
        self.device = device

        self.actor = CQLActor(
            latent_dim, action_dim, hidden_dim, max_action=max_action,
        ).to(device)
        self.critic = CQLCritic(latent_dim, action_dim, hidden_dim).to(device)
        self.critic_target = CQLCritic(latent_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def update(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_z: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one CQL update step.

        Args:
            z: Current latent (B, latent_dim).
            action: Action (B, action_dim).
            reward: Reward (B, 1).
            next_z: Next latent (B, latent_dim).
            done: Done flag (B, 1).

        Returns:
            Dict of training metrics.
        """
        B = z.shape[0]

        # Target Q
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_z)
            q1_target, q2_target = self.critic_target(next_z, next_action)
            q_target = torch.min(q1_target, q2_target)
            target_q = reward + self.gamma * (1 - done) * q_target

        # Critic loss
        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL penalty: logsumexp of Q on random actions - Q on data actions
        random_actions = torch.rand(B, self.n_action_samples, action.shape[-1]).to(z.device) * self.max_action
        z_rep = z.unsqueeze(1).expand(-1, self.n_action_samples, -1).reshape(-1, z.shape[-1])
        random_actions_flat = random_actions.reshape(-1, action.shape[-1])

        q1_rand, q2_rand = self.critic(z_rep, random_actions_flat)
        q1_rand = q1_rand.reshape(B, self.n_action_samples)
        q2_rand = q2_rand.reshape(B, self.n_action_samples)

        cql_penalty = (
            torch.logsumexp(q1_rand, dim=1).mean()
            + torch.logsumexp(q2_rand, dim=1).mean()
            - q1.mean() - q2.mean()
        )

        total_critic_loss = critic_loss + self.alpha * cql_penalty

        self.critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor loss
        new_action, log_prob = self.actor(z)
        q1_new, q2_new = self.critic(z, new_action)
        actor_loss = -torch.min(q1_new, q2_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }
