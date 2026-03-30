"""HM-LatentSafeRL main method trainer.

Stage 5: Train the full Health-Mamba Latent Safe Offline RL framework
with joint losses for RL, health prediction, dynamics, and risk.
"""
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional

from ..utils.seed import set_seed
from ..utils.device import get_device, get_amp_dtype, print_device_info
from ..utils.logger import setup_logger, MetricsLogger
from ..utils.config import load_config, setup_run_dir, generate_run_name
from ..utils.profiling import Profiler
from ..models.encoders.mamba_encoder import build_encoder
from ..models.rl.hm_latent_safe_rl import HMLatentSafeRL


class HMLatentSafeRLTrainer:
    """Train HM-LatentSafeRL with joint optimization."""

    def __init__(self, cfg: Dict, run_dir: str, device: torch.device):
        """Initialize trainer.

        Args:
            cfg: Full merged configuration.
            run_dir: Run output directory.
            device: Compute device.
        """
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device

        enc_cfg = cfg.get("encoder", {})
        rl_cfg = cfg.get("rl", {})
        hm_cfg = rl_cfg.get("hm_safe", {})

        # Build shared encoder
        encoder = build_encoder(cfg).to(device)
        latent_dim = enc_cfg.get("latent_dim", 64)

        # Build full HM-LatentSafeRL model
        self.model = HMLatentSafeRL(
            encoder=encoder,
            latent_dim=latent_dim,
            action_dim=1,
            hidden_dim=256,
            obs_dim=enc_cfg.get("obs_dim", 10),
            n_ensemble=cfg.get("world_model", {}).get("n_ensemble", 5),
            max_action=cfg.get("env", {}).get("max_current", 6.0),
            safety_mode=hm_cfg.get("safety_mode", "uncertainty_aware"),
            risk_threshold=cfg.get("safety_layer", {}).get("risk_threshold", 0.3),
            gamma=rl_cfg.get("gamma", 0.99),
            tau=rl_cfg.get("tau", 0.005),
            conservative_weight=hm_cfg.get("conservative_weight", 1.0),
            device=str(device),
        ).to(device)

        # Optimizer for all parameters
        self.optimizer = torch.optim.AdamW(
            self.model.get_all_parameters(),
            lr=rl_cfg.get("lr_actor", 3e-4),
            weight_decay=rl_cfg.get("weight_decay", 1e-4),
        )

        # AMP
        self.use_amp = rl_cfg.get("use_amp", True)
        self.amp_dtype = get_amp_dtype()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Logging
        self.logger = setup_logger("hm_safe_rl", run_dir)
        self.metrics = MetricsLogger(run_dir)
        self.profiler = Profiler(run_dir)

        # Training config
        self.epochs = rl_cfg.get("epochs", 200)
        self.steps_per_epoch = rl_cfg.get("steps_per_epoch", 1000)
        self.grad_clip = rl_cfg.get("grad_clip", 1.0)
        self.save_every = rl_cfg.get("save_every", 20)

        # Loss weights
        self.health_weight = hm_cfg.get("health_loss_weight", 0.5)
        self.dynamics_weight = hm_cfg.get("dynamics_loss_weight", 0.3)
        self.risk_weight = hm_cfg.get("risk_loss_weight", 0.5)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dict of epoch metrics.
        """
        self.model.train()
        agg = {}
        n = 0

        for batch in dataloader:
            obs = batch["obs"].to(self.device)
            action = batch["action"].to(self.device)
            reward = batch["reward"].to(self.device).unsqueeze(-1)
            next_obs = batch["next_obs"].to(self.device)
            done = batch["done"].to(self.device).unsqueeze(-1)

            self.profiler.start_step()

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                losses = self.model.compute_losses(
                    obs, action, reward, next_obs, done,
                    health_weight=self.health_weight,
                    dynamics_weight=self.dynamics_weight,
                    risk_weight=self.risk_weight,
                )
                total_loss = losses["total_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.get_all_parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Soft update target critic
            self.model.soft_update_target()

            self.profiler.end_step(n, obs.shape[0])

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    agg[k] = agg.get(k, 0.0) + v.item()
                else:
                    agg[k] = agg.get(k, 0.0) + float(v)
            n += 1

            if n >= self.steps_per_epoch:
                break

        return {k: v / max(n, 1) for k, v in agg.items()}

    def train(self, train_loader, val_loader=None):
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation loader.
        """
        self.logger.info(f"Starting HM-LatentSafeRL training for {self.epochs} epochs")
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
                f"total={metrics.get('total_loss', 0):.4f} "
                f"critic={metrics.get('critic_loss', 0):.4f} "
                f"actor={metrics.get('actor_loss', 0):.4f} "
                f"risk={metrics.get('risk_mean', 0):.3f} | {elapsed:.1f}s"
            )

            if metrics.get("total_loss", float("inf")) < best_loss:
                best_loss = metrics["total_loss"]
                self._save_checkpoint("best.pt", epoch, metrics)

            if epoch % self.save_every == 0:
                self._save_checkpoint("latest.pt", epoch, metrics)

        self._save_checkpoint("latest.pt", self.epochs, metrics)
        self.logger.info(f"Training complete. Best total loss: {best_loss:.4f}")

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict):
        """Save full model checkpoint."""
        path = os.path.join(self.run_dir, "checkpoints", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_pretrained_encoder(self, ckpt_path: str):
        """Load pretrained encoder weights.

        Args:
            ckpt_path: Path to pretrained encoder checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "encoder_state_dict" in ckpt:
            self.model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.logger.info(f"Loaded pretrained encoder from {ckpt_path}")
        else:
            self.logger.warning(f"No encoder_state_dict found in {ckpt_path}")


def run_hm_latent_safe_rl(config_path: str, run_name: Optional[str] = None, seed: int = 42):
    """Run HM-LatentSafeRL training from config.

    Args:
        config_path: Config YAML path.
        run_name: Optional run name.
        seed: Random seed.
    """
    set_seed(seed)
    cfg = load_config(config_path)
    device = get_device()

    if run_name is None:
        run_name = generate_run_name("hm_latent_safe_rl", "synthetic", seed)

    run_dir = setup_run_dir("results", run_name, cfg)

    from ..datasets.build_offline_dataset import build_dataloader
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    train_loader = build_dataloader(
        data_dir=data_cfg.get("processed_dir", "data/processed"),
        split="train",
        batch_size=cfg.get("rl", {}).get("batch_size", 256),
        window_length=data_cfg.get("window_length", 64),
        num_workers=data_cfg.get("num_workers", 8),
    )

    trainer = HMLatentSafeRLTrainer(cfg, run_dir, device)
    trainer.train(train_loader)
