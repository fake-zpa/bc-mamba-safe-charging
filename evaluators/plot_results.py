"""Plotting utilities for battery charging experiment results.

Generates publication-quality plots for reward curves, voltage/temperature
profiles, SOC trajectories, current profiles, and risk scores.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def set_plot_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    })


def plot_training_curves(metrics_path: str, output_dir: str):
    """Plot training loss and reward curves from metrics.json.

    Args:
        metrics_path: Path to metrics.json file.
        output_dir: Directory for output plots.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_path, "r") as f:
        history = json.load(f)

    steps = [h["step"] for h in history]

    # Reward curve
    if "total_reward" in history[0] or "total_loss" in history[0]:
        fig, ax = plt.subplots()
        key = "total_reward" if "total_reward" in history[0] else "total_loss"
        vals = [h.get(key, 0) for h in history]
        label = "Total Reward" if "reward" in key else "Total Loss"
        ax.plot(steps, vals, label=label, color="tab:blue")
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training {label}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reward_curve.png"))
        plt.close(fig)

    # Component losses
    loss_keys = [k for k in history[0] if "loss" in k.lower() and k != "total_loss"]
    if loss_keys:
        fig, ax = plt.subplots()
        for key in loss_keys:
            vals = [h.get(key, 0) for h in history]
            ax.plot(steps, vals, label=key)
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "loss_curves.png"))
        plt.close(fig)

    # Risk curve
    if any("risk" in k for k in history[0]):
        fig, ax = plt.subplots()
        risk_keys = [k for k in history[0] if "risk" in k.lower()]
        for key in risk_keys:
            vals = [h.get(key, 0) for h in history]
            ax.plot(steps, vals, label=key)
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel("Risk Score")
        ax.set_title("Risk During Training")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "risk_curve.png"))
        plt.close(fig)


def plot_episode_trajectories(
    trajectories: List[Dict],
    output_dir: str,
    max_episodes: int = 5,
):
    """Plot episode trajectories: voltage, current, temperature, SOC.

    Args:
        trajectories: List of trajectory dicts from evaluation.
        output_dir: Directory for output plots.
        max_episodes: Maximum episodes to plot.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    n_plot = min(len(trajectories), max_episodes)

    # Voltage
    fig, ax = plt.subplots()
    for i in range(n_plot):
        ax.plot(trajectories[i]["voltage"], label=f"Ep {i}", alpha=0.8)
    ax.axhline(y=4.2, color="r", linestyle="--", alpha=0.5, label="V_max=4.2V")
    ax.set_xlabel("Step")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage Profile")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "voltage_curve.png"))
    plt.close(fig)

    # Current
    fig, ax = plt.subplots()
    for i in range(n_plot):
        ax.plot(trajectories[i]["current"], label=f"Ep {i}", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Current (A)")
    ax.set_title("Charging Current Profile")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "current_curve.png"))
    plt.close(fig)

    # Temperature
    fig, ax = plt.subplots()
    for i in range(n_plot):
        ax.plot(trajectories[i]["temperature"], label=f"Ep {i}", alpha=0.8)
    ax.axhline(y=45.0, color="r", linestyle="--", alpha=0.5, label="T_max=45°C")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Profile")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temperature_curve.png"))
    plt.close(fig)

    # SOC
    fig, ax = plt.subplots()
    for i in range(n_plot):
        ax.plot(trajectories[i]["soc"], label=f"Ep {i}", alpha=0.8)
    ax.axhline(y=0.8, color="g", linestyle="--", alpha=0.5, label="Target SOC=0.8")
    ax.set_xlabel("Step")
    ax.set_ylabel("SOC")
    ax.set_title("State of Charge")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "soc_curve.png"))
    plt.close(fig)

    # Risk
    if "risk" in trajectories[0]:
        fig, ax = plt.subplots()
        for i in range(n_plot):
            ax.plot(trajectories[i]["risk"], label=f"Ep {i}", alpha=0.8)
        ax.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="Risk threshold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Risk Score")
        ax.set_title("Safety Risk Score")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "risk_episode_curve.png"))
        plt.close(fig)

    print(f"Plots saved to {output_dir}/")


def plot_all_results(run_dir: str):
    """Generate all plots for a completed run.

    Args:
        run_dir: Path to run directory.
    """
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training curves
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        plot_training_curves(metrics_path, plots_dir)

    # Evaluation trajectories
    eval_summary = os.path.join(run_dir, "eval", "evaluation_summary.json")
    if os.path.exists(eval_summary):
        # Load trajectories from episode metrics
        ep_metrics_path = os.path.join(run_dir, "eval", "episode_metrics.json")
        if os.path.exists(ep_metrics_path):
            with open(ep_metrics_path, "r") as f:
                episodes = json.load(f)

    print(f"All plots generated in {plots_dir}/")
