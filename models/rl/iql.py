"""Implicit Q-Learning (IQL) for offline RL battery charging.

Implements IQL with expectile regression and advantage-weighted policy.
Uses shared Mamba encoder latent state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class IQLValueNet(nn.Module):
    """State value network V(z) for IQL."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class IQLCritic(nn.Module):
    """Double Q-network for IQL."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 1, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        def build_q():
            layers = []
            in_dim = latent_dim + action_dim
            for _ in range(n_layers):
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
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


class IQLActor(nn.Module):
    """Advantage-weighted policy for IQL."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 1, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1, max_action: float = 6.0):
        super().__init__()
        self.max_action = max_action
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        if deterministic:
            action = torch.sigmoid(mean) * self.max_action
            return action, torch.zeros_like(action[:, :1])
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.sigmoid(raw) * self.max_action
        log_prob = dist.log_prob(raw) - torch.log(self.max_action * torch.sigmoid(raw) * (1 - torch.sigmoid(raw)) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def log_prob(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given action."""
        h = self.backbone(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        # Inverse sigmoid to get raw action
        action_clipped = action.clamp(1e-6, self.max_action - 1e-6) / self.max_action
        raw = torch.log(action_clipped / (1 - action_clipped + 1e-6) + 1e-6)
        dist = torch.distributions.Normal(mean, std)
        log_p = dist.log_prob(raw) - torch.log(self.max_action * torch.sigmoid(raw) * (1 - torch.sigmoid(raw)) + 1e-6)
        return log_p.sum(-1, keepdim=True)

    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.forward(z, deterministic=True)
        return action


class IQL:
    """IQL algorithm with expectile regression."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 1, hidden_dim: int = 256,
                 gamma: float = 0.99, tau: float = 0.005, expectile: float = 0.7,
                 temperature: float = 3.0, max_action: float = 6.0,
                 lr_actor: float = 3e-4, lr_critic: float = 3e-4, lr_value: float = 3e-4,
                 device: str = "cuda"):
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.device = device

        self.actor = IQLActor(latent_dim, action_dim, hidden_dim, max_action=max_action).to(device)
        self.critic = IQLCritic(latent_dim, action_dim, hidden_dim).to(device)
        self.critic_target = IQLCritic(latent_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.value = IQLValueNet(latent_dim, hidden_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr_value)

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff ** 2).mean()

    def update(self, z: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
               next_z: torch.Tensor, done: torch.Tensor) -> Dict[str, float]:
        # Value loss (expectile regression on Q - V)
        with torch.no_grad():
            q1, q2 = self.critic_target(z, action)
            q = torch.min(q1, q2)
        v = self.value(z)
        value_loss = self._expectile_loss(q - v)

        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_opt.step()

        # Critic loss
        with torch.no_grad():
            next_v = self.value(next_z)
            target_q = reward + self.gamma * (1 - done) * next_v
        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor loss (advantage-weighted)
        with torch.no_grad():
            v_val = self.value(z)
            q_val = torch.min(*self.critic_target(z, action))
            advantage = q_val - v_val
            weights = torch.exp(advantage * self.temperature).clamp(max=100.0)

        log_prob = self.actor.log_prob(z, action)
        actor_loss = -(weights * log_prob).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": q.mean().item(),
            "v_mean": v.mean().item(),
            "advantage_mean": advantage.mean().item(),
        }
