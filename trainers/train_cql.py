"""CQL trainer for offline RL battery charging baseline.

Stage 4 baseline: Conservative Q-Learning with shared Mamba encoder.
"""
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict

from ..utils.device import get_device, get_amp_dtype
from ..utils.logger import setup_logger, MetricsLogger
from ..utils.profiling import Profiler
from ..models.encoders.mamba_encoder import build_encoder
from ..models.rl.cql import CQL


class CQLTrainer:
    """Train CQL with shared Mamba encoder."""

    def __init__(self, cfg: Dict, run_dir: str, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device

        enc_cfg = cfg.get("encoder", {})
        rl_cfg = cfg.get("rl", {})
        cql_cfg = rl_cfg.get("cql", {})

        self.encoder = build_encoder(cfg).to(device)
        latent_dim = enc_cfg.get("latent_dim", 64)

        self.cql = CQL(
            latent_dim=latent_dim,
            action_dim=1,
            gamma=rl_cfg.get("gamma", 0.99),
            tau=rl_cfg.get("tau", 0.005),
            alpha=cql_cfg.get("alpha", 5.0),
            n_action_samples=cql_cfg.get("n_action_samples", 10),
            max_action=cfg.get("env", {}).get("max_current", 6.0),
            lr_actor=rl_cfg.get("lr_actor", 3e-4),
            lr_critic=rl_cfg.get("lr_critic", 3e-4),
            device=str(device),
        )

        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=rl_cfg.get("lr_encoder", 1e-4),
        )

        self.use_amp = rl_cfg.get("use_amp", True)
        self.amp_dtype = get_amp_dtype()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.logger = setup_logger("cql", run_dir)
        self.metrics = MetricsLogger(run_dir)
        self.profiler = Profiler(run_dir)

        self.epochs = rl_cfg.get("epochs", 200)
        self.steps_per_epoch = rl_cfg.get("steps_per_epoch", 1000)
        self.save_every = rl_cfg.get("save_every", 20)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.encoder.train()
        agg = {}
        n = 0

        for batch in dataloader:
            obs = batch["obs"].to(self.device)
            action = batch["action"].to(self.device)
            reward = batch["reward"].to(self.device).unsqueeze(-1)
            next_obs = batch["next_obs"].to(self.device)
            done = batch["done"].to(self.device).unsqueeze(-1)

            z = self.encoder(obs)
            with torch.no_grad():
                next_z = self.encoder(next_obs)

            step_metrics = self.cql.update(z, action, reward, next_z, done)

            # Update encoder via critic gradients
            self.encoder_opt.zero_grad()
            z_new = self.encoder(obs)
            q_val = self.cql.critic.q_min(z_new, action)
            enc_loss = -q_val.mean() * 0.01
            enc_loss.backward()
            self.encoder_opt.step()

            for k, v in step_metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

            if n >= self.steps_per_epoch:
                break

        return {k: v / max(n, 1) for k, v in agg.items()}

    def train(self, train_loader):
        self.logger.info(f"Starting CQL training for {self.epochs} epochs")
        best_q = -float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(train_loader)
            elapsed = time.time() - t0
            metrics["epoch"] = epoch
            metrics["time_s"] = elapsed
            self.metrics.log(epoch, metrics)
            self.profiler.log_all(epoch)

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"critic={metrics.get('critic_loss', 0):.4f} "
                f"actor={metrics.get('actor_loss', 0):.4f} "
                f"q1={metrics.get('q1_mean', 0):.3f} | {elapsed:.1f}s"
            )

            if metrics.get("q1_mean", -999) > best_q:
                best_q = metrics["q1_mean"]
                self._save("best.pt", epoch, metrics)

            if epoch % self.save_every == 0:
                self._save("latest.pt", epoch, metrics)

        self._save("latest.pt", self.epochs, metrics)

    def _save(self, name, epoch, metrics):
        path = os.path.join(self.run_dir, "checkpoints", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "actor_state_dict": self.cql.actor.state_dict(),
            "critic_state_dict": self.cql.critic.state_dict(),
            "metrics": metrics,
        }, path)
