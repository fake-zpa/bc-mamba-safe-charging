"""World model training on latent state transitions.

Stage 3: Train the ensemble latent world model to predict next latent state,
observations, and rewards from current state-action pairs.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional

from ..utils.seed import set_seed
from ..utils.device import get_device, get_amp_dtype
from ..utils.logger import setup_logger, MetricsLogger
from ..utils.config import load_config, setup_run_dir, generate_run_name
from ..utils.profiling import Profiler
from ..models.encoders.mamba_encoder import build_encoder
from ..models.dynamics.latent_world_model import LatentWorldModel


class WorldModelTrainer:
    """Train ensemble world model on latent transitions."""

    def __init__(self, cfg: Dict, run_dir: str, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device

        enc_cfg = cfg.get("encoder", {})
        wm_cfg = cfg.get("world_model", {})
        train_cfg = cfg.get("world_model_train", {})

        # Build encoder (load pretrained if available)
        self.encoder = build_encoder(cfg).to(device)
        self.encoder.eval()  # Freeze during world model training

        latent_dim = enc_cfg.get("latent_dim", 64)
        self.world_model = LatentWorldModel(
            latent_dim=latent_dim,
            action_dim=wm_cfg.get("action_dim", 1),
            hidden_dim=wm_cfg.get("hidden_dim", 256),
            obs_dim=wm_cfg.get("obs_dim", 10),
            n_ensemble=wm_cfg.get("n_ensemble", 5),
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=train_cfg.get("lr", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        self.use_amp = train_cfg.get("use_amp", True)
        self.amp_dtype = get_amp_dtype()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.logger = setup_logger("world_model", run_dir)
        self.metrics = MetricsLogger(run_dir)
        self.profiler = Profiler(run_dir)

        self.epochs = train_cfg.get("epochs", 100)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.save_every = train_cfg.get("save_every", 10)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.world_model.train()
        total_loss = 0.0
        n = 0

        for batch in dataloader:
            obs = batch["obs"].to(self.device)
            action = batch["action"].to(self.device)
            next_obs = batch["next_obs"].to(self.device)

            self.profiler.start_step()

            with torch.no_grad():
                z = self.encoder(obs)
                next_z = self.encoder(next_obs)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self.world_model.loss(
                    z, action, next_z,
                    target_obs=next_obs[:, -1, :] if next_obs.dim() == 3 else None,
                )
                loss = losses["total_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.profiler.end_step(n, obs.shape[0])
            total_loss += loss.item()
            n += 1

        return {"loss": total_loss / max(n, 1)}

    def train(self, train_loader, val_loader=None):
        self.logger.info(f"Starting world model training for {self.epochs} epochs")
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(train_loader)
            elapsed = time.time() - t0
            metrics["epoch"] = epoch
            metrics["time_s"] = elapsed
            self.metrics.log(epoch, metrics)
            self.profiler.log_all(epoch)

            self.logger.info(f"Epoch {epoch}/{self.epochs} | loss={metrics['loss']:.4f} | {elapsed:.1f}s")

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self._save_checkpoint("best.pt", epoch, metrics)

            if epoch % self.save_every == 0:
                self._save_checkpoint("latest.pt", epoch, metrics)

        self._save_checkpoint("latest.pt", self.epochs, metrics)

    def _save_checkpoint(self, name, epoch, metrics):
        path = os.path.join(self.run_dir, "checkpoints", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "world_model_state_dict": self.world_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
