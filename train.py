#!/usr/bin/env python3
"""
Clean BC + inference-time safety ablation pipeline.

Strategy:
  - Train BC_Mamba and BC_GRU WITHOUT any safety filter.
    Policy learns to charge fast from expert demonstrations.
  - At evaluation, apply safety filters at inference time:
    NoSafety / Sigmoid / CBF  →  clean ablation.

Ablation axes:
  1. Encoder:  Mamba vs GRU
  2. Safety:   None  vs Sigmoid vs CBF

Run (screen):
  conda activate mamba2
  screen -S bc_clean
  PYTHONPATH=/root/autodl-tmp/GJ PYBAMM_DISABLE_TELEMETRY=true \
    python run_bc_clean.py [--quick] [--seeds 42,123,456,789,1024]
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/GJ')
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'

from battery_mamba_safe_rl.envs.pybamm_env import PyBaMMChargingEnv
from battery_mamba_safe_rl.models.encoders.mamba_encoder import MambaHealthEncoder, GRUHealthEncoder
from battery_mamba_safe_rl.models.rl.hm_latent_safe_rl import HMLatentSafeRL, SafeActor
from battery_mamba_safe_rl.models.safety.cbf_safety import (
    CBFSafetyFilter, NoCBFSafetyFilter, SigmoidOnlySafetyFilter
)
from battery_mamba_safe_rl.utils.metrics import compute_episode_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Quick mode: 1 seed, 30 epochs')
parser.add_argument('--seeds', type=str, default='42,123,456,789,1024')
args = parser.parse_args()

RESULTS_DIR = 'results/bc_clean'
os.makedirs(RESULTS_DIR, exist_ok=True)

CAP        = 0.681
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS      = [42] if args.quick else [int(s) for s in args.seeds.split(',')]
BC_EPOCHS  = 30 if args.quick else 150
N_EP_EVAL  = 3 if args.quick else 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# BUILD MODEL  (BC-only variant: encoder + actor, no safety)
# ============================================================
def build_bc_model(encoder_type, seed):
    torch.manual_seed(seed)
    if encoder_type == 'mamba':
        encoder = MambaHealthEncoder(obs_dim=10, d_model=128, n_layer=4, d_state=16, latent_dim=64)
    else:
        encoder = GRUHealthEncoder(obs_dim=10, hidden_dim=128, n_layers=2, latent_dim=64)
    model = HMLatentSafeRL(
        encoder=encoder, latent_dim=64, action_dim=1, hidden_dim=256,
        n_ensemble=3, max_action=6.0 * CAP, safety_mode='none', device=DEVICE
    ).to(DEVICE)
    return model


# ============================================================
# TRAINING  (pure BC, no safety during training)
# ============================================================
def train_bc(name, encoder_type, seed, dataset_path, bc_epochs):
    data = np.load(dataset_path)
    obs     = torch.FloatTensor(data['obs']).to(DEVICE)
    actions = torch.FloatTensor(data['actions']).to(DEVICE)
    n       = obs.shape[0]
    bs      = min(256, n)

    model   = build_bc_model(encoder_type, seed)
    params  = list(model.encoder.parameters()) + list(model.actor.parameters())
    opt     = torch.optim.Adam(params, lr=1e-3)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bc_epochs)

    best_loss, best_state = float('inf'), None
    t0 = time.time()

    for epoch in range(bc_epochs):
        idx = torch.randperm(n)
        tl, nb = 0.0, 0
        for i in range(0, n, bs):
            b   = idx[i:i+bs]
            z   = model.encoder(obs[b])
            pred, _ = model.actor(z)
            loss = F.mse_loss(pred, actions[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tl += loss.item(); nb += 1
        sched.step()
        avg = tl / nb
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % max(1, bc_epochs // 5) == 0 or epoch == bc_epochs - 1:
            log(f"    [{name} s{seed}] ep{epoch}/{bc_epochs}: loss={avg:.6f}")

    model.load_state_dict(best_state)
    log(f"    [{name} s{seed}] done  {time.time()-t0:.0f}s  best={best_loss:.6f}")

    save_dir = os.path.join(RESULTS_DIR, f'{name}_s{seed}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'encoder_type': encoder_type},
               os.path.join(save_dir, 'best.pt'))
    return save_dir


# ============================================================
# EVALUATION
# ============================================================
def run_episode(env, get_action_fn):
    obs, info = env.reset(options={'initial_soc': np.random.uniform(0, 0.08)})
    done_flag = truncated = False
    vs, cs, ts, socs, rews = [], [], [], [], []
    v_viol = t_viol = interv = steps = 0

    while not (done_flag or truncated):
        raw_a, safe_a = get_action_fn(obs, info)
        if safe_a < raw_a * 0.95:
            interv += 1
        obs, rew, done_flag, truncated, info = env.step(np.array([safe_a], dtype=np.float32))
        v, t = info['voltage'], info['temperature']
        vs.append(v); cs.append(safe_a / CAP); ts.append(t); socs.append(info['soc']); rews.append(rew)
        if v > 4.15: v_viol += 1
        if t > 38.0: t_viol += 1
        steps += 1

    m = compute_episode_metrics(
        np.array(vs), np.array([c * CAP for c in cs]), np.array(ts),
        np.array(socs), np.array(rews), np.zeros(len(rews)),
        v_max=4.15, t_max=38.0
    )
    return {
        'time':       steps * 0.5,
        'final_soc':  info['soc'],
        'avg_crate':  float(np.mean(cs)) if cs else 0.0,
        'max_v':      float(np.max(vs))  if vs else 0.0,
        'max_t':      float(np.max(ts))  if ts else 0.0,
        'v_viol':     v_viol / max(steps, 1),
        't_viol':     t_viol / max(steps, 1),
        'deg':        m.get('degradation_proxy', 0),
        'intervention': interv / max(steps, 1),
    }


def eval_rl(ckpt_dir, safety_filter, env, n_ep):
    """Load saved BC model and evaluate with given inference-time safety filter."""
    ckpt        = torch.load(os.path.join(ckpt_dir, 'best.pt'), map_location='cpu')
    sd          = ckpt['model_state_dict']
    enc_type    = ckpt.get('encoder_type', 'mamba')

    if enc_type == 'mamba':
        enc = MambaHealthEncoder(obs_dim=10, d_model=128, n_layer=4, d_state=16, latent_dim=64).to(DEVICE)
    else:
        enc = GRUHealthEncoder(obs_dim=10, hidden_dim=128, n_layers=2, latent_dim=64).to(DEVICE)

    enc.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith('encoder.')}, strict=False)
    enc.train()  # keep train mode to avoid Mamba caching issues

    actor = SafeActor(latent_dim=64, action_dim=1, hidden_dim=256, max_action=6.0 * CAP).to(DEVICE)
    actor.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('actor.')}, strict=False)
    actor.eval()

    sf = safety_filter

    def get_action(obs, info):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            z     = enc(obs_t)
            raw, _ = actor(z)
            T_t   = torch.tensor([info['temperature']], device=DEVICE)
            V_t   = torch.tensor([info['voltage']],     device=DEVICE)
            safe, _ = sf(raw.squeeze(0), T_t, V_t)
        raw_a  = float(np.clip(raw.cpu().item(),  0, 6.0 * CAP))
        safe_a = float(np.clip(safe.cpu().item(), 0, 6.0 * CAP))
        return raw_a, safe_a

    return [run_episode(env, get_action) for _ in range(n_ep)]


def eval_rule(policy_fn, env, n_ep):
    def get_action(obs, info):
        raw_a = policy_fn(info)
        return raw_a, float(np.clip(raw_a, 0, 6.0 * CAP))
    return [run_episode(env, get_action) for _ in range(n_ep)]


# ============================================================
# RULE BASELINES
# ============================================================
def make_cc_cv(c_rate):
    def policy(info):
        if info['voltage'] >= 4.1:
            return max(0.1 * CAP, (4.15 - info['voltage']) / 0.05 * c_rate * CAP)
        return c_rate * CAP
    return policy


def ms543_policy(info):
    soc = info['soc']
    c   = 5.0 if soc < 0.3 else (4.0 if soc < 0.55 else 3.0)
    if info['voltage'] >= 4.1:
        return max(0.1 * CAP, (4.15 - info['voltage']) / 0.05 * c * CAP)
    return c * CAP


# ============================================================
# PRINT RESULTS
# ============================================================
def agg(eps, key):
    vals = [e[key] for e in eps]
    return np.mean(vals), np.std(vals)


def print_results(results, scenario):
    log(f"\n{'='*85}")
    log(f"RESULTS: {scenario}")
    log(f"{'='*85}")
    log(f"{'Method':30s} {'Time(min)':>11s} {'avgC':>6s} {'maxT':>6s} "
        f"{'T_viol':>7s} {'V_viol':>7s} {'Deg':>7s} {'Interv':>7s}")
    log("-" * 85)
    order = [
        'CC-CV_3C', 'CC-CV_4C', 'CC-CV_5C', 'CC-CV_6C', 'MS-CC_5-4-3C',
        'BC_GRU+NoSafety', 'BC_GRU+Sig', 'BC_GRU+CBF',
        'BC_Mamba+NoSafety', 'BC_Mamba+Sig', 'BC_Mamba+CBF',
    ]
    for name in order:
        if name not in results:
            continue
        eps         = results[name]
        t_m,  t_s  = agg(eps, 'time')
        c_m,  _    = agg(eps, 'avg_crate')
        mt_m, _    = agg(eps, 'max_t')
        tv_m, _    = agg(eps, 't_viol')
        vv_m, _    = agg(eps, 'v_viol')
        dg_m, _    = agg(eps, 'deg')
        iv_m, _    = agg(eps, 'intervention')
        log(f"{name:30s} {t_m:5.1f}±{t_s:.1f}  {c_m:5.2f}C  {mt_m:5.1f}  "
            f"{tv_m*100:5.1f}%  {vv_m*100:5.1f}%  {dg_m:6.1f}  {iv_m*100:5.1f}%")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t0_total = time.time()
    log("=" * 85)
    log(f"BC Clean  |  quick={args.quick}  |  seeds={SEEDS}  |  device={DEVICE}")
    log(f"BC_EPOCHS={BC_EPOCHS}  N_EP_EVAL={N_EP_EVAL}")
    log("=" * 85)

    # ---- Dataset ----
    dataset_path = 'results/spme_thermal/expert_dataset_env_large.npz'
    if not os.path.exists(dataset_path):
        dataset_path = 'results/spme_thermal/expert_dataset_env.npz'
    if not os.path.exists(dataset_path):
        log("ERROR: dataset not found. Run generate_env_dataset_large.py first.")
        sys.exit(1)
    d = np.load(dataset_path)
    log(f"Dataset: {d['obs'].shape[0]} transitions, mean action={d['actions'].mean()/CAP:.2f}C")

    # ---- Train 2 base models (NO safety filter) ----
    train_specs = [
        ('BC_Mamba', 'mamba'),
        ('BC_GRU',   'gru'),
    ]
    checkpoints = {}
    for name, enc_type in train_specs:
        log(f"\n>>> Training {name}...")
        checkpoints[name] = []
        for seed in SEEDS:
            ckpt = train_bc(name, enc_type, seed, dataset_path, BC_EPOCHS)
            checkpoints[name].append(ckpt)

    with open(os.path.join(RESULTS_DIR, 'checkpoints.json'), 'w') as f:
        json.dump(checkpoints, f, indent=2)
    log("\nAll models trained. Starting evaluation...")

    # ---- Scenarios ----
    scenarios = [
        ('nominal_25C', 25.0),
        ('hot_35C',     35.0),
    ]
    all_results = {}

    for scenario_name, ambient_t in scenarios:
        log(f"\n>>> Evaluating scenario: {scenario_name} ({ambient_t}°C)...")

        env = PyBaMMChargingEnv(
            window_length=64, max_steps=200, dt=30.0,
            max_c_rate=6.0, ambient_temp=ambient_t
        )
        np.random.seed(42)

        # Safety filters — use correct T_amb for each scenario
        cbf_filter = CBFSafetyFilter(
            T_limit=38.0, V_limit=4.15, cbf_alpha=0.8,
            sigmoid_margin_T=3.0, sigmoid_margin_V=0.15,
            k_heat=0.09, k_cool=0.02, T_amb=ambient_t
        ).to(DEVICE)
        sig_filter = SigmoidOnlySafetyFilter(
            T_limit=38.0, V_limit=4.15, margin_T=3.0, margin_V=0.15, min_scale=0.05
        ).to(DEVICE)
        no_filter  = NoCBFSafetyFilter().to(DEVICE)

        # Eval configs: (display_name, base_model, safety_filter)
        eval_specs = [
            ('BC_GRU+NoSafety',   'BC_GRU',   no_filter),
            ('BC_GRU+Sig',        'BC_GRU',   sig_filter),
            ('BC_GRU+CBF',        'BC_GRU',   cbf_filter),
            ('BC_Mamba+NoSafety', 'BC_Mamba', no_filter),
            ('BC_Mamba+Sig',      'BC_Mamba', sig_filter),
            ('BC_Mamba+CBF',      'BC_Mamba', cbf_filter),
        ]

        res = {}

        # Rule baselines
        for rname, rpol in [
            ('CC-CV_3C',     make_cc_cv(3.0)),
            ('CC-CV_4C',     make_cc_cv(4.0)),
            ('CC-CV_5C',     make_cc_cv(5.0)),
            ('CC-CV_6C',     make_cc_cv(6.0)),
            ('MS-CC_5-4-3C', ms543_policy),
        ]:
            log(f"  Eval {rname}...")
            res[rname] = eval_rule(rpol, env, N_EP_EVAL)

        # RL methods
        for eval_name, base_name, sf in eval_specs:
            ckpt_list = checkpoints[base_name]
            log(f"  Eval {eval_name} ({len(ckpt_list)} seeds)...")
            eps_all = []
            for ckpt_dir in ckpt_list:
                eps_all.extend(eval_rl(ckpt_dir, sf, env, N_EP_EVAL))
            res[eval_name] = eps_all

        print_results(res, scenario_name)
        all_results[scenario_name] = {
            n: [{k: float(v) for k, v in e.items()} for e in eps]
            for n, eps in res.items()
        }

    out_path = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved → {out_path}")
    log(f"Total runtime: {(time.time()-t0_total)/60:.1f} min")
