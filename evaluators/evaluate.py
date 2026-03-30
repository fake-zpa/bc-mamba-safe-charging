"""Evaluation module for battery charging policies.

Evaluates trained models in the battery environment, collects
trajectories, computes metrics, and saves results.
"""
import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from ..envs.battery_env import BatteryChargingEnv, make_env
from ..utils.metrics import compute_episode_metrics


def evaluate_policy(
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    env: BatteryChargingEnv,
    n_episodes: int = 10,
    device: str = "cuda",
    deterministic: bool = True,
    safety_layer=None,
    risk_head=None,
    world_model=None,
) -> Dict:
    """Evaluate a policy in the battery charging environment.

    Args:
        encoder: Trained encoder model.
        actor: Trained actor (must have get_action or forward method).
        env: Battery charging environment.
        n_episodes: Number of evaluation episodes.
        device: Compute device.
        deterministic: Use deterministic actions.
        safety_layer: Optional safety layer for action projection.
        risk_head: Optional risk head for safety assessment.
        world_model: Optional world model for uncertainty estimation.

    Returns:
        Dict with evaluation results.
    """
    encoder.eval()
    actor.eval()

    all_metrics = []
    all_trajectories = []

    for ep in range(n_episodes):
        obs, info = env.reset(options={"initial_soc": np.random.uniform(0.0, 0.1)})

        voltages, currents, temperatures, socs = [], [], [], []
        rewards_list, risks_list = [], []
        done = False
        truncated = False

        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                z = encoder(obs_tensor)

                if hasattr(actor, "get_action"):
                    action = actor.get_action(z)
                else:
                    action, _ = actor(z, deterministic=deterministic)

                # Safety projection
                risk_score = 0.0
                if safety_layer is not None and risk_head is not None:
                    risk_info = risk_head(z, action)
                    uncertainty = None
                    if world_model is not None:
                        uncertainty = world_model.uncertainty_score(z, action)
                    action = safety_layer(action, risk_info, uncertainty)
                    risk_score = risk_info["overall_risk"].item()

            action_np = action.cpu().numpy().flatten()
            obs, reward, done, truncated, info = env.step(action_np)

            voltages.append(info["voltage"])
            currents.append(info["current"])
            temperatures.append(info["temperature"])
            socs.append(info["soc"])
            rewards_list.append(reward)
            risks_list.append(risk_score)

        # Compute episode metrics
        ep_metrics = compute_episode_metrics(
            voltage=np.array(voltages),
            current=np.array(currents),
            temperature=np.array(temperatures),
            soc=np.array(socs),
            rewards=np.array(rewards_list),
            risks=np.array(risks_list),
        )
        ep_metrics["episode"] = ep
        all_metrics.append(ep_metrics)

        all_trajectories.append({
            "voltage": voltages,
            "current": currents,
            "temperature": temperatures,
            "soc": socs,
            "reward": rewards_list,
            "risk": risks_list,
        })

    # Aggregate metrics
    agg = {}
    for key in all_metrics[0]:
        if key == "episode":
            continue
        vals = [m[key] for m in all_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    return {
        "summary": agg,
        "episodes": all_metrics,
        "trajectories": all_trajectories,
    }


def save_evaluation_results(
    results: Dict,
    output_dir: str,
):
    """Save evaluation results to files.

    Args:
        results: Evaluation results dict.
        output_dir: Output directory.
    """
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Save summary
    with open(os.path.join(eval_dir, "evaluation_summary.json"), "w") as f:
        json.dump(results["summary"], f, indent=2)

    # Save episode metrics
    with open(os.path.join(eval_dir, "episode_metrics.json"), "w") as f:
        json.dump(results["episodes"], f, indent=2)

    # Save rollout traces as CSV
    import csv
    traces_path = os.path.join(eval_dir, "rollout_traces.csv")
    with open(traces_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "voltage", "current", "temperature", "soc", "reward", "risk"])
        for ep_idx, traj in enumerate(results["trajectories"]):
            for step_idx in range(len(traj["voltage"])):
                writer.writerow([
                    ep_idx, step_idx,
                    f"{traj['voltage'][step_idx]:.4f}",
                    f"{traj['current'][step_idx]:.4f}",
                    f"{traj['temperature'][step_idx]:.4f}",
                    f"{traj['soc'][step_idx]:.4f}",
                    f"{traj['reward'][step_idx]:.4f}",
                    f"{traj['risk'][step_idx]:.4f}",
                ])

    print(f"Evaluation results saved to {eval_dir}/")


def run_evaluation(
    run_dir: str,
    checkpoint_name: str = "best.pt",
    n_episodes: int = 10,
    device: str = "cuda",
):
    """Run evaluation from a training run directory.

    Args:
        run_dir: Path to training run directory.
        checkpoint_name: Checkpoint filename.
        n_episodes: Number of evaluation episodes.
        device: Compute device.
    """
    from ..utils.config import load_yaml
    from ..models.encoders.mamba_encoder import build_encoder

    # Load config
    cfg = load_yaml(os.path.join(run_dir, "merged_config.yaml"))

    # Build models
    encoder = build_encoder(cfg).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoints", checkpoint_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state_dict"])
        elif "model_state_dict" in ckpt:
            # Extract encoder from full model state dict
            enc_state = {
                k.replace("encoder.", ""): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            if enc_state:
                encoder.load_state_dict(enc_state)

    # Build simple actor for evaluation (from checkpoint if available)
    from ..models.rl.bc import BCPolicy
    latent_dim = cfg.get("encoder", {}).get("latent_dim", 64)
    actor = BCPolicy(latent_dim=latent_dim).to(device)
    if "policy_state_dict" in ckpt:
        actor.load_state_dict(ckpt["policy_state_dict"])

    # Build environment
    env = make_env(cfg)

    # Evaluate
    results = evaluate_policy(
        encoder=encoder,
        actor=actor,
        env=env,
        n_episodes=n_episodes,
        device=device,
    )

    # Save
    save_evaluation_results(results, run_dir)

    return results
