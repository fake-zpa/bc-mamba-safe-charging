"""Behavioral Cloning trainer for baseline comparison.

Stage 4 baseline: Train BC policy using shared Mamba encoder.
"""
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional

from ..utils.seed import set_seed
from ..utils.device import get_device, get_amp_dtype
from ..utils.logger import setup_logger, MetricsLogger
from ..utils.profiling import Profiler
from ..models.encoders.mamba_encoder import build_encoder
from ..models.rl.bc import BCPolicy


class BCTrainer:
    """Train BC policy with shared Mamba encoder."""

    def __init__(self, cfg: Dict, run_dir: str, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device

        enc_cfg = cfg.get("encoder", {})
        rl_cfg = cfg.get("rl", {})

        self.encoder = build_encoder(cfg).to(device)
        latent_dim = enc_cfg.get("latent_dim", 64)

        self.policy = BCPolicy(
            latent_dim=latent_dim,
            action_dim=1,
            hidden_dim=rl_cfg.get("hidden_dim", 256) if isinstance(rl_cfg.get("hidden_dim"), int) else 256,
            max_action=cfg.get("env", {}).get("max_current", 6.0),
        ).to(device)

        params = list(self.encoder.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=rl_cfg.get("lr_actor", 3e-4))

        self.use_amp = rl_cfg.get("use_amp", True)
        self.amp_dtype = get_amp_dtype()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.logger = setup_logger("bc", run_dir)
        self.metrics = MetricsLogger(run_dir)
        self.profiler = Profiler(run_dir)

        self.epochs = rl_cfg.get("epochs", 200)
        self.grad_clip = rl_cfg.get("grad_clip", 1.0)
        self.save_every = rl_cfg.get("save_every", 20)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.encoder.train()
        self.policy.train()
        total_loss = 0.0
        n = 0

        for batch in dataloader:
            obs = batch["obs"].to(self.device)
            action = batch["action"].to(self.device)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                z = self.encoder(obs)
                losses = self.policy.loss(z, action)
                loss = losses["bc_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.policy.parameters()), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n += 1

        return {"bc_loss": total_loss / max(n, 1)}

    def train(self, train_loader, val_loader=None):
        self.logger.info(f"Starting BC training for {self.epochs} epochs")
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(train_loader)
            elapsed = time.time() - t0
            metrics["epoch"] = epoch
            metrics["time_s"] = elapsed
            self.metrics.log(epoch, metrics)

            self.logger.info(f"Epoch {epoch}/{self.epochs} | bc_loss={metrics['bc_loss']:.4f} | {elapsed:.1f}s")

            if metrics["bc_loss"] < best_loss:
                best_loss = metrics["bc_loss"]
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
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
