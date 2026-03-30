"""Configuration management with YAML loading, merging, and saving."""
import os
import copy
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, override takes precedence.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(
    config_path: str,
    overrides: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Load config from YAML and apply optional overrides.

    Args:
        config_path: Path to main config YAML.
        overrides: Optional dictionary of overrides.

    Returns:
        Merged configuration.
    """
    cfg = load_yaml(config_path)

    # Support base config inheritance
    if "_base_" in cfg:
        base_path = cfg.pop("_base_")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_cfg = load_yaml(base_path)
        cfg = deep_merge(base_cfg, cfg)

    if overrides:
        cfg = deep_merge(cfg, overrides)

    return cfg


def save_config(cfg: Dict[str, Any], path: str):
    """Save configuration to YAML file.

    Args:
        cfg: Configuration dictionary.
        path: Output path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def generate_run_name(
    method: str = "unknown",
    dataset: str = "unknown",
    seed: int = 42,
) -> str:
    """Generate a unique run name with timestamp.

    Args:
        method: Method name.
        dataset: Dataset name.
        seed: Random seed.

    Returns:
        Run name string.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{method}_{dataset}_s{seed}"


def setup_run_dir(
    results_dir: str,
    run_name: str,
    cfg: Dict[str, Any],
    overrides: Optional[Dict] = None,
) -> str:
    """Create run directory and save configs.

    Args:
        results_dir: Base results directory.
        run_name: Run name.
        cfg: Merged configuration.
        overrides: Optional overrides dict for diff saving.

    Returns:
        Path to run directory.
    """
    run_dir = os.path.join(results_dir, run_name)
    for sub in ["checkpoints", "plots", "eval", "profiler"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    # Save merged config
    save_config(cfg, os.path.join(run_dir, "merged_config.yaml"))

    # Save config diff if overrides provided
    if overrides:
        save_config(overrides, os.path.join(run_dir, "config_diff.yaml"))

    # Save run info
    from .device import get_device_info
    run_info = {
        "run_name": run_name,
        "start_time": datetime.now().isoformat(),
        "config": cfg,
        "device_info": get_device_info(),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    return run_dir
