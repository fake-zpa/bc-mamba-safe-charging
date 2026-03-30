"""
Quick demo: run one BC-Mamba+CBF episode on the PyBaMM SPMe environment.

Usage:
    conda activate mamba2
    python demo.py                        # nominal 25 °C
    python demo.py --ambient_temp 35.0    # thermal stress 35 °C
"""

import argparse
import torch
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ambient_temp", type=float, default=25.0,
                   help="Ambient temperature in °C (default: 25)")
    p.add_argument("--init_soc",    type=float, default=0.04,
                   help="Initial SOC (default: 0.04)")
    p.add_argument("--checkpoint",  type=str,   default=None,
                   help="Path to .pt checkpoint (optional; random weights if None)")
    p.add_argument("--seq_len",     type=int,   default=64,
                   help="History window length (default: 64)")
    p.add_argument("--device",      type=str,   default="cpu")
    return p.parse_args()


def build_model(seq_len: int, device: str, checkpoint: str | None):
    from models.encoders.mamba_encoder import MambaEncoder
    from models.safety.cbf_safety import CBFSafetyLayer

    encoder = MambaEncoder(
        obs_dim=10,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_latent=64,
        seq_len=seq_len,
    ).to(device)

    safety = CBFSafetyLayer(
        T_lim=38.0,
        V_lim=4.15,
        k_heat=0.09,
        k_cool=0.02,
        alpha_c=0.8,
        margin_T=1.5,
        margin_V=0.15,
    )

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        print(f"Loaded checkpoint from {checkpoint}")
    else:
        print("No checkpoint provided – using random weights for architecture demo.")

    encoder.eval()
    return encoder, safety


def run_episode(args):
    try:
        from envs.pybamm_env import PyBaMMMambaEnv
    except ImportError as e:
        print(f"[ERROR] Cannot import PyBaMM environment: {e}")
        print("Make sure PyBaMM is installed: pip install pybamm")
        return

    device = args.device
    encoder, safety = build_model(args.seq_len, device, args.checkpoint)

    env = PyBaMMMambaEnv(
        ambient_temp=args.ambient_temp,
        T_lim=38.0,
        V_lim=4.15,
        seq_len=args.seq_len,
    )

    obs_history, _ = env.reset(init_soc=args.init_soc)
    done = False
    step = 0

    total_reward = 0.0
    t_viols = 0
    v_viols = 0
    interventions = 0

    print(f"\n{'='*55}")
    print(f"  BC-Mamba+CBF Demo  |  T_amb={args.ambient_temp}°C  |  SOC0={args.init_soc:.2f}")
    print(f"{'='*55}")
    print(f"{'Step':>5} {'SOC':>6} {'Temp(°C)':>9} {'Volt(V)':>8} {'C-rate':>7} {'CBF?':>5}")
    print(f"{'-'*55}")

    while not done:
        # Encode history
        x = torch.tensor(obs_history, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            latent = encoder(x)
            # Simple linear head (demo): map latent → raw action in [0, 1]
            raw_action = torch.sigmoid(latent.mean()).item() * 6.0  # [0, 6C]

        # Safety filter
        T_t = float(obs_history[-1][2])   # temperature from last obs
        V_t = float(obs_history[-1][0])   # voltage from last obs
        safe_action, intervened = safety.filter(raw_action, T_t, V_t)
        if intervened:
            interventions += 1

        obs_history, reward, terminated, truncated, info = env.step(safe_action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        soc  = info.get("soc", float("nan"))
        temp = info.get("temperature", float("nan"))
        volt = info.get("voltage", float("nan"))
        crate = info.get("c_rate", float("nan"))

        if temp > 38.0:
            t_viols += 1
        if volt > 4.15:
            v_viols += 1

        if step % 5 == 0 or done:
            cbf_flag = "✓" if intervened else " "
            print(f"{step:>5} {soc:>6.3f} {temp:>9.2f} {volt:>8.3f} {crate:>7.2f} {cbf_flag:>5}")

    print(f"{'-'*55}")
    print(f"  Final SOC       : {soc:.3f}")
    print(f"  Total steps     : {step}")
    print(f"  T-violations    : {t_viols} ({100*t_viols/max(step,1):.1f}%)")
    print(f"  V-violations    : {v_viols} ({100*v_viols/max(step,1):.1f}%)")
    print(f"  CBF interventions: {interventions} ({100*interventions/max(step,1):.1f}%)")
    print(f"  Cumulative reward: {total_reward:.2f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    args = parse_args()
    run_episode(args)
