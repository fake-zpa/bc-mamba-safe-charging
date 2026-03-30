#!/usr/bin/env python3
"""
Generalization experiments for BC_Mamba+CBF (margin_T=1.5).

Tests our best method under conditions outside the training distribution:
  A. Temperature: 15°C, 20°C, 25°C, 30°C, 33°C, 35°C  (trained on 25°C)
     NOTE: 40°C excluded — T_amb > T_limit=38°C is physically degenerate
  B. Initial SOC:  0-8% (train), 10-20%, 20-30%, 30-40%, 40-50%
  C. Noise:        observation noise σ=0, 0.005, 0.01, 0.02, 0.05

Only evaluates BC_Mamba+CBF(m=1.5) — baselines not re-run.
5 seeds × 5 episodes per condition.

Usage:
  conda activate mamba2
  screen -S generalize
  PYTHONPATH=/root/autodl-tmp/GJ PYBAMM_DISABLE_TELEMETRY=true \
    python run_generalization.py 2>&1 | tee results/generalization/gen.log
"""
import os, sys, time, json, numpy as np, torch

sys.path.insert(0, '/root/autodl-tmp/GJ')
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'

from battery_mamba_safe_rl.envs.pybamm_env import PyBaMMChargingEnv
from battery_mamba_safe_rl.models.encoders.mamba_encoder import MambaHealthEncoder
from battery_mamba_safe_rl.models.rl.hm_latent_safe_rl import SafeActor
from battery_mamba_safe_rl.models.safety.cbf_safety import CBFSafetyFilter

CKPT_JSON  = 'results/bc_clean/checkpoints.json'
OUT_DIR    = 'results/generalization'
LOG_FILE   = 'results/generalization/gen.log'
MARGIN_T   = 1.5
CAP        = 0.681
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EP       = 5
N_SEEDS    = 5

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_episode(env, get_action_fn, init_soc_range=(0.0, 0.08), obs_noise=0.0):
    obs, info = env.reset(options={'initial_soc': np.random.uniform(*init_soc_range)})
    done = trunc = False
    cs, ts, degs = [], [], []
    v_viol = t_viol = interv = steps = 0
    while not (done or trunc):
        if obs_noise > 0:
            obs = obs + np.random.normal(0, obs_noise, obs.shape).astype(np.float32)
        raw_a, safe_a = get_action_fn(obs, info)
        if safe_a < raw_a * 0.95:
            interv += 1
        obs, _, done, trunc, info = env.step(np.array([safe_a], dtype=np.float32))
        cs.append(safe_a / CAP)
        ts.append(info['temperature'])
        degs.append(info.get('degradation_proxy', 0))
        if info['voltage'] > 4.15:     v_viol += 1
        if info['temperature'] > 38.0: t_viol += 1
        steps += 1
    return {
        'time':         steps * 0.5,
        'final_soc':    info['soc'],
        'avg_crate':    float(np.mean(cs)) if cs else 0.0,
        'max_t':        float(np.max(ts))  if ts else 0.0,
        'v_viol':       v_viol / max(steps, 1),
        't_viol':       t_viol / max(steps, 1),
        'intervention': interv / max(steps, 1),
        'degradation':  float(np.sum(degs)),
    }


def load_mamba_actor(ckpt_dir):
    ckpt = torch.load(os.path.join(ckpt_dir, 'best.pt'), map_location='cpu')
    sd = ckpt['model_state_dict']
    enc = MambaHealthEncoder(obs_dim=10, d_model=128, n_layer=4,
                              d_state=16, latent_dim=64).to(DEVICE)
    enc.load_state_dict(
        {k[8:]: v for k, v in sd.items() if k.startswith('encoder.')}, strict=False)
    enc.train()
    actor = SafeActor(latent_dim=64, action_dim=1,
                      hidden_dim=256, max_action=6.0 * CAP).to(DEVICE)
    actor.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith('actor.')}, strict=False)
    actor.eval()
    return enc, actor


def make_action_fn(enc, actor, sf):
    def get_action(obs, info):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            z = enc(obs_t)
            raw, _ = actor(z)
            T_t = torch.tensor([info['temperature']], device=DEVICE)
            V_t = torch.tensor([info['voltage']],     device=DEVICE)
            safe, _ = sf(raw.squeeze(0), T_t, V_t)
        return (float(np.clip(raw.cpu().item(),  0, 6*CAP)),
                float(np.clip(safe.cpu().item(), 0, 6*CAP)))
    return get_action


def eval_condition(ckpt_dirs, ambient_t, init_soc_range, obs_noise, n_ep):
    cbf = CBFSafetyFilter(
        T_limit=38.0, V_limit=4.15, cbf_alpha=0.8,
        sigmoid_margin_T=MARGIN_T, sigmoid_margin_V=0.15,
        k_heat=0.09, k_cool=0.02, T_amb=ambient_t
    ).to(DEVICE)
    env = PyBaMMChargingEnv(
        window_length=64, max_steps=120, dt=30.0,
        max_c_rate=6.0, ambient_temp=ambient_t
    )
    eps = []
    for ckpt in ckpt_dirs:
        enc, actor = load_mamba_actor(ckpt)
        fn = make_action_fn(enc, actor, cbf)
        for _ in range(n_ep):
            eps.append(run_episode(env, fn, init_soc_range, obs_noise))
    return eps


def agg(eps, k):
    v = [e[k] for e in eps]
    return np.mean(v), np.std(v)


def print_table(results, title):
    log(f"\n{'='*80}")
    log(f"GENERALIZATION: {title}")
    log(f"{'='*80}")
    log(f"{'Condition':30s} {'Time':>10s} {'avgC':>6s} {'maxT':>6s} "
        f"{'T_viol':>7s} {'V_viol':>7s} {'Deg':>8s} {'Interv':>7s}")
    log("-" * 80)
    for name, eps in results.items():
        tm, ts = agg(eps, 'time')
        cm, _  = agg(eps, 'avg_crate')
        mt, _  = agg(eps, 'max_t')
        tv, _  = agg(eps, 't_viol')
        vv, _  = agg(eps, 'v_viol')
        dg, _  = agg(eps, 'degradation')
        iv, _  = agg(eps, 'intervention')
        log(f"{name:30s} {tm:5.1f}±{ts:.1f}  {cm:5.2f}C  {mt:5.1f}  "
            f"{tv*100:5.1f}%  {vv*100:5.1f}%  {dg:8.1f}  {iv*100:5.1f}%")


if __name__ == '__main__':
    t0 = time.time()
    log("=" * 80)
    log(f"Generalization | BC_Mamba+CBF m={MARGIN_T} | {N_SEEDS}seeds×{N_EP}eps")
    log("=" * 80)

    with open(CKPT_JSON) as f:
        ckpts = json.load(f)
    mamba_ckpts = ckpts['BC_Mamba'][:N_SEEDS]
    log(f"Loaded {len(mamba_ckpts)} checkpoints")

    out_path = os.path.join(OUT_DIR, 'results.json')
    all_out = {}
    np.random.seed(42)

    # ── A. Temperature generalization ──────────────────────────────────────────
    log("\n[A] Temperature generalization (init_SOC=0-8%, no noise)")
    log("    [Note: T_amb=40C excluded — ambient > T_limit=38C is physically degenerate]")
    temp_results = {}
    for t_amb in [15.0, 20.0, 25.0, 30.0, 33.0, 35.0]:
        label = f"T_amb={t_amb:.0f}°C"
        log(f"  Eval {label}...")
        temp_results[label] = eval_condition(
            mamba_ckpts, ambient_t=t_amb,
            init_soc_range=(0.0, 0.08), obs_noise=0.0, n_ep=N_EP
        )
    print_table(temp_results, "Temperature Generalization")
    all_out['temperature'] = {
        k: [{kk: float(vv) for kk, vv in e.items()} for e in v]
        for k, v in temp_results.items()
    }
    # intermediate save
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    log(f"  [checkpoint] saved temperature results → {out_path}")

    # ── B. Initial SOC generalization ──────────────────────────────────────────
    log("\n[B] Initial SOC generalization (T_amb=25°C, no noise)")
    soc_results = {}
    soc_ranges = {
        'SOC_0-8%  (train)': (0.00, 0.08),
        'SOC_10-20%':        (0.10, 0.20),
        'SOC_20-30%':        (0.20, 0.30),
        'SOC_30-40%':        (0.30, 0.40),
        'SOC_40-50%':        (0.40, 0.50),
    }
    for label, soc_range in soc_ranges.items():
        log(f"  Eval {label}...")
        soc_results[label] = eval_condition(
            mamba_ckpts, ambient_t=25.0,
            init_soc_range=soc_range, obs_noise=0.0, n_ep=N_EP
        )
    print_table(soc_results, "Initial SOC Generalization")
    all_out['initial_soc'] = {
        k: [{kk: float(vv) for kk, vv in e.items()} for e in v]
        for k, v in soc_results.items()
    }
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    log(f"  [checkpoint] saved SOC results → {out_path}")

    # ── C. Observation noise robustness ───────────────────────────────────────
    log("\n[C] Observation noise robustness (T_amb=25°C, SOC=0-8%)")
    noise_results = {}
    for noise in [0.0, 0.005, 0.01, 0.02, 0.05]:
        label = f"noise=σ{noise:.3f}"
        log(f"  Eval {label}...")
        noise_results[label] = eval_condition(
            mamba_ckpts, ambient_t=25.0,
            init_soc_range=(0.0, 0.08), obs_noise=noise, n_ep=N_EP
        )
    print_table(noise_results, "Observation Noise Robustness")
    all_out['obs_noise'] = {
        k: [{kk: float(vv) for kk, vv in e.items()} for e in v]
        for k, v in noise_results.items()
    }

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, 'generalization_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    log(f"\nSaved → {out_path}")
    log(f"Total: {(time.time()-t0)/60:.1f} min")
    log("DONE")
