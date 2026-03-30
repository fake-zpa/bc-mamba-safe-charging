"""Logging utilities for training and evaluation."""
import os
import sys
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    log_dir: str,
    log_file: str = "train.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger that writes to both file and stdout.

    Args:
        name: Logger name.
        log_dir: Directory for log file.
        log_file: Log file name.
        level: Logging level.

    Returns:
        Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


class MetricsLogger:
    """Logger for training/evaluation metrics that writes CSV and JSON."""

    def __init__(self, log_dir: str):
        """Initialize metrics logger.

        Args:
            log_dir: Directory for metrics files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self.json_path = self.log_dir / "metrics.json"
        self.history: list = []
        self._csv_initialized = False

    def log(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a given step.

        Args:
            step: Current step/epoch number.
            metrics: Dictionary of metric name -> value.
        """
        row = {"step": step, "timestamp": datetime.now().isoformat()}
        row.update(metrics)
        self.history.append(row)

        # Write CSV
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

        # Write JSON (overwrite with full history)
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def get_best(self, metric: str, mode: str = "max") -> Optional[Dict]:
        """Get the best step for a given metric.

        Args:
            metric: Metric name.
            mode: 'max' or 'min'.

        Returns:
            Best metrics row or None.
        """
        if not self.history:
            return None
        valid = [r for r in self.history if metric in r]
        if not valid:
            return None
        if mode == "max":
            return max(valid, key=lambda x: x[metric])
        return min(valid, key=lambda x: x[metric])
