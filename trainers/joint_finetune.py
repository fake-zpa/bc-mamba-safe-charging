"""Joint finetuning of all HM-LatentSafeRL components.

Stage 6: End-to-end finetuning of encoder + actor + critic + world model
+ risk head + degradation head with reduced learning rates.
"""
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional

from ..utils.device import get_amp_dtype
from ..utils.logger import setup_logger, MetricsLogger
from ..utils.profiling import Profiler
from .train_hm_latent_safe_rl import HMLatentSafeRLTrainer


class JointFinetuner(HMLatentSafeRLTrainer):
    """Joint finetuning with reduced learning rates.

    Inherits from HMLatentSafeRLTrainer but uses lower LR
    and optionally loads a Stage 5 checkpoint.
    """

    def __init__(self, cfg: Dict, run_dir: str, device: torch.device):
        """Initialize joint finetuner with reduced LR."""
        # Override LR before parent init
        rl_cfg = cfg.get("rl", {})
        finetune_lr = rl_cfg.get("lr_actor", 3e-4) * 0.1
        cfg.setdefault("rl", {})["lr_actor"] = finetune_lr

        super().__init__(cfg, run_dir, device)
        self.logger = setup_logger("joint_finetune", run_dir)

    def load_stage5_checkpoint(self, ckpt_path: str):
        """Load Stage 5 (HM-LatentSafeRL) checkpoint.

        Args:
            ckpt_path: Path to Stage 5 checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info(f"Loaded Stage 5 checkpoint from {ckpt_path}")
