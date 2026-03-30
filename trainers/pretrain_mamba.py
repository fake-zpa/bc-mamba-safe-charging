"""Mamba encoder pretraining via masked reconstruction and health prediction.

Stage 2: Pretrain the Mamba encoder on battery time-series data before
using it as the shared state encoder for RL training.
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional

from ..utils.seed import set_seed
from ..utils.device import get_device, get_amp_dtype, print_device_info
from ..utils.logger import setup_logger, MetricsLogger
from ..utils.config import load_config, setup_run_dir, generate_run_name
from ..utils.profiling import Profiler
from ..models.encoders.mamba_encoder import build_encoder


class MambaPretrainer:
    """Pretrain Mamba encoder with reconstruction and health prediction losses."""

    def __init__(self, cfg: Dict, run_dir: str, device: torch.device):
        """Initialize pretrainer.

        Args:
            cfg: Full merged configuration.
            run_dir: Output directory for this run.
            device: Compute device.
        """
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device

        pretrain_cfg = cfg.get("pretrain", {})
        enc_cfg = cfg.get("encoder", {})

        # Build encoder
        self.encoder = build_encoder(cfg).to(device)
        obs_dim = enc_cfg.get("obs_dim", 10)
        d_model = enc_cfg.get("d_model", 128)

        # Reconstruction head
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, obs_dim),
        ).to(device)

        # Health prediction head
        latent_dim = enc_cfg.get("latent_dim", 64)
        self.health_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # soh, capacity_fade, resistance_growth, cycle_life
        ).to(device)

        # Optimizer
        all_params = (
            list(self.encoder.parameters())
            + list(self.recon_head.parameters())
            + list(self.health_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=pretrain_cfg.get("lr", 3e-4),
            weight_decay=pretrain_cfg.get("weight_decay", 1e-4),
        )

        # AMP
        self.use_amp = pretrain_cfg.get("use_amp", True)
        self.amp_dtype = get_amp_dtype()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Logging
        self.logger = setup_logger("pretrain", run_dir)
        self.metrics = MetricsLogger(run_dir)
        self.profiler = Profiler(run_dir)

        # Config
        self.epochs = pretrain_cfg.get("epochs", 100)
        self.grad_clip = pretrain_cfg.get("grad_clip", 1.0)
        self.recon_weight = pretrain_cfg.get("recon_weight", 1.0)
        self.health_weight = pretrain_cfg.get("health_pred_weight", 0.5)
        self.save_every = pretrain_cfg.get("save_every", 10)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dict of epoch metrics.
        """
        self.encoder.train()
        self.recon_head.train()
        self.health_head.train()

        total_loss = 0.0
        total_recon = 0.0
        total_health = 0.0
        n_batches = 0

        for batch in dataloader:
            obs = batch["obs"].to(self.device)  # (B, L, obs_dim)
            B, L, D = obs.shape

            self.profiler.start_step()

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # Encode: get sequence output for reconstruction
                z, h_seq = self.encoder.encode_for_pretrain(obs)

                # Reconstruction loss: predict original observations from hidden states
                recon = self.recon_head(h_seq)  # (B, L, obs_dim)
                recon_loss = F.mse_loss(recon, obs)

                # Health prediction from latent state
                health_pred = self.health_head(z)  # (B, 4)
                # Pseudo targets from data
                health_target = torch.stack([
                    torch.ones(B, device=self.device) * 0.9,  # soh proxy
                    obs[:, -1, 9].abs(),  # degradation proxy
                    obs[:, -1, 8].abs(),  # resistance proxy
                    torch.ones(B, device=self.device) * 500,  # cycle life proxy
                ], dim=-1)
                health_loss = F.mse_loss(health_pred, health_target)

                loss = self.recon_weight * recon_loss + self.health_weight * health_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.profiler.end_step(n_batches, B)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_health += health_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "recon_loss": total_recon / n,
            "health_loss": total_health / n,
        }

    def train(self, train_loader, val_loader=None):
        """Full pretraining loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
        """
        self.logger.info(f"Starting Mamba pretraining for {self.epochs} epochs")
        print_device_info()

        best_loss = float("inf")
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
                f"loss={metrics['loss']:.4f} recon={metrics['recon_loss']:.4f} "
                f"health={metrics['health_loss']:.4f} | {elapsed:.1f}s"
            )

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self._save_checkpoint("best.pt", epoch, metrics)

            if epoch % self.save_every == 0:
                self._save_checkpoint("latest.pt", epoch, metrics)

        self._save_checkpoint("latest.pt", self.epochs, metrics)
        self.logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict):
        """Save checkpoint."""
        path = os.path.join(self.run_dir, "checkpoints", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "recon_head_state_dict": self.recon_head.state_dict(),
            "health_head_state_dict": self.health_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        self.logger.info(f"Saved checkpoint: {path}")


def run_pretrain(config_path: str, run_name: Optional[str] = None, seed: int = 42):
    """Run Mamba pretraining from config file.

    Args:
        config_path: Path to configuration YAML.
        run_name: Optional run name.
        seed: Random seed.
    """
    set_seed(seed)
    cfg = load_config(config_path)
    device = get_device()

    if run_name is None:
        run_name = generate_run_name("pretrain_mamba", "synthetic", seed)

    run_dir = setup_run_dir("results", run_name, cfg)

    # Build data loader
    from ..datasets.build_offline_dataset import build_dataloader
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    train_loader = build_dataloader(
        data_dir=data_cfg.get("processed_dir", "data/processed"),
        split="train",
        batch_size=cfg.get("pretrain", {}).get("batch_size", 256),
        window_length=data_cfg.get("window_length", 64),
        num_workers=data_cfg.get("num_workers", 8),
    )

    trainer = MambaPretrainer(cfg, run_dir, device)
    trainer.train(train_loader)
