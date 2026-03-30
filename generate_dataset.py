#!/usr/bin/env python3
"""Generate dataset using env.step() to ensure obs format matches exactly.
Run: conda activate mamba2 && PYTHONPATH=/root/autodl-tmp/GJ PYBAMM_DISABLE_TELEMETRY=true python generate_env_dataset.py
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/GJ')
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'

from battery_mamba_safe_rl.envs.pybamm_env import PyBaMMChargingEnv

CAP = 0.681
RESULTS_DIR = 'results/spme_thermal'
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

env = PyBaMMChargingEnv(window_length=64, max_steps=200, dt=30.0, max_c_rate=6.0)

# Generate protocols: CC at various C-rates + multi-stage
protocols = []
# CC protocols at different C-rates and initial SOCs
for c_rate in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    for init_soc in [0.0, 0.02, 0.05, 0.08, 0.10]:
        protocols.append({'type': 'cc', 'c_rate': c_rate, 'init_soc': init_soc})

# Multi-stage protocols
for stages in [(5,4,3), (5,3,2), (4,3,2), (5,4,2), (6,4,2), (5,3,1), (4,3,1)]:
    for init_soc in [0.0, 0.03, 0.05, 0.08, 0.10]:
        protocols.append({'type': 'ms', 'stages': stages, 'init_soc': init_soc})

log(f"Total protocols: {len(protocols)}")

all_obs, all_act, all_rew, all_next, all_done = [], [], [], [], []
n_traj = 0
t_start = time.time()

for i, proto in enumerate(protocols):
    if i % 20 == 0:
        elapsed = time.time() - t_start
        eta = (elapsed / max(i, 1)) * (len(protocols) - i) / 60
        log(f"Protocol {i}/{len(protocols)} (ETA: {eta:.0f}min)...")
    
    try:
        init_soc = proto['init_soc']
        obs, info = env.reset(options={'initial_soc': init_soc})
        done, truncated = False, False
        ep_obs, ep_act, ep_rew, ep_next = [], [], [], []
        step = 0
        
        while not (done or truncated):
            # Determine C-rate
            if proto['type'] == 'cc':
                current = proto['c_rate'] * CAP
            elif proto['type'] == 'ms':
                stages = proto['stages']
                soc = info['soc']
                # Split SOC range into equal stages
                n_stages = len(stages)
                stage_idx = min(int(soc / 0.8 * n_stages), n_stages - 1)
                current = stages[stage_idx] * CAP
            else:
                current = 3.0 * CAP
            
            # CC-CV transition
            if info['voltage'] >= 4.1:
                c = current / CAP
                current = max(0.1 * CAP, (4.15 - info['voltage']) / 0.05 * c * CAP)
            
            current = np.clip(current, 0, 6.0 * CAP)
            
            prev_obs = obs.copy()
            obs, rew, done, truncated, info = env.step(np.array([current], dtype=np.float32))
            
            ep_obs.append(prev_obs)
            ep_act.append(np.array([current], dtype=np.float32))
            ep_rew.append(float(rew))
            ep_next.append(obs.copy())
            step += 1
        
        if step >= 3 and info['soc'] >= 0.7:
            # Compute avg C-rate
            avg_c = np.mean([a[0] / CAP for a in ep_act])
            if avg_c >= 2.0:
                all_obs.extend(ep_obs)
                all_act.extend(ep_act)
                all_rew.extend(ep_rew)
                all_next.extend(ep_next)
                # Done signals
                dones = [0.0] * len(ep_obs)
                dones[-1] = 1.0
                all_done.extend(dones)
                n_traj += 1
    except Exception as e:
        if i < 5:
            log(f"  Error on protocol {i}: {e}")
        continue

total_time = time.time() - t_start
log(f"Done: {n_traj} trajectories, {len(all_obs)} transitions in {total_time/60:.1f}min")

if len(all_obs) > 0:
    obs_arr = np.array(all_obs)
    act_arr = np.array(all_act)
    rew_arr = np.array(all_rew, dtype=np.float32)
    next_arr = np.array(all_next)
    done_arr = np.array(all_done, dtype=np.float32)
    
    dataset_path = os.path.join(RESULTS_DIR, 'expert_dataset_env.npz')
    np.savez_compressed(dataset_path, obs=obs_arr, actions=act_arr, rewards=rew_arr, 
                        next_obs=next_arr, done=done_arr)
    
    log(f"Saved: {dataset_path}")
    log(f"  obs={obs_arr.shape}, actions={act_arr.shape}")
    log(f"  avg C-rate: {act_arr.mean()/CAP:.2f}C")
    log(f"  avg reward: {rew_arr.mean():.2f}")
    
    # Quick sanity check: compare first window with env format
    log(f"\n  Sample obs[0] last timestep: V={obs_arr[0,-1,0]:.3f} I={obs_arr[0,-1,1]:.3f} T={obs_arr[0,-1,2]:.1f} SOC={obs_arr[0,-1,3]:.3f} deg={obs_arr[0,-1,9]:.2f}")
    log(f"  Sample obs[100] last timestep: V={obs_arr[100,-1,0]:.3f} I={obs_arr[100,-1,1]:.3f} T={obs_arr[100,-1,2]:.1f} SOC={obs_arr[100,-1,3]:.3f} deg={obs_arr[100,-1,9]:.2f}")
else:
    log("ERROR: no valid trajectories generated!")
