"""Profiling utilities for GPU/CPU memory and throughput tracking."""
import os
import csv
import time
import torch
import psutil
from typing import Optional
from pathlib import Path


class Profiler:
    """Track GPU memory, CPU memory, and training throughput."""

    def __init__(self, log_dir: str):
        """Initialize profiler.

        Args:
            log_dir: Directory for profiling logs.
        """
        self.log_dir = Path(log_dir) / "profiler"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_log = self.log_dir / "gpu_mem.csv"
        self.cpu_log = self.log_dir / "cpu_mem.csv"
        self.throughput_log = self.log_dir / "throughput.csv"
        self._init_csvs()
        self._step_start: Optional[float] = None

    def _init_csvs(self):
        """Initialize CSV files with headers."""
        with open(self.gpu_log, "w", newline="") as f:
            csv.writer(f).writerow(["step", "allocated_mb", "reserved_mb", "peak_mb"])
        with open(self.cpu_log, "w", newline="") as f:
            csv.writer(f).writerow(["step", "rss_mb", "vms_mb", "percent"])
        with open(self.throughput_log, "w", newline="") as f:
            csv.writer(f).writerow(["step", "step_time_s", "samples_per_sec"])

    def log_gpu(self, step: int):
        """Log GPU memory usage.

        Args:
            step: Current step number.
        """
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6
        with open(self.gpu_log, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{allocated:.1f}", f"{reserved:.1f}", f"{peak:.1f}"])

    def log_cpu(self, step: int):
        """Log CPU memory usage.

        Args:
            step: Current step number.
        """
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        with open(self.cpu_log, "a", newline="") as f:
            csv.writer(f).writerow([
                step,
                f"{mem.rss / 1e6:.1f}",
                f"{mem.vms / 1e6:.1f}",
                f"{proc.memory_percent():.1f}",
            ])

    def start_step(self):
        """Mark the start of a training step."""
        self._step_start = time.time()

    def end_step(self, step: int, batch_size: int):
        """Mark the end of a training step and log throughput.

        Args:
            step: Current step number.
            batch_size: Batch size for throughput calculation.
        """
        if self._step_start is None:
            return
        elapsed = time.time() - self._step_start
        sps = batch_size / elapsed if elapsed > 0 else 0.0
        with open(self.throughput_log, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{elapsed:.4f}", f"{sps:.1f}"])
        self._step_start = None

    def log_all(self, step: int):
        """Log all profiling metrics.

        Args:
            step: Current step number.
        """
        self.log_gpu(step)
        self.log_cpu(step)
