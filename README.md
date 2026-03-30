# BC-Mamba+CBF: Safe Fast Charging of Lithium-Ion Batteries

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code for the paper:

> **Safe Fast Charging of Lithium-Ion Batteries via Behaviour Cloning with Mamba Encoder and Control Barrier Function Safety Filter**  
> *Submitted to Journal of Energy Storage, 2026*

---

## Overview

This repository implements a small-data offline safe fast-charging framework that combines:

- **Mamba state-space model encoder** — compresses a 64-step observation window into a compact latent representation
- **Behaviour Cloning (BC) actor** — imitates expert charging trajectories on a 3 918-transition offline dataset
- **CBF dual-layer safety filter** — enforces hard thermal (T ≤ 38 °C) and voltage (V ≤ 4.15 V) constraints at inference time *without modifying the BC training objective*

The framework is developed and evaluated entirely on [PyBaMM](https://www.pybamm.org/) SPMe electrochemical simulation with a lumped thermal model.

```
┌─────────────────────────────────────────────────────┐
│  Inference Pipeline                                  │
│                                                     │
│  obs_window [64×10]                                 │
│       │                                             │
│       ▼                                             │
│  ┌─────────────┐     latent z     ┌─────────────┐  │
│  │ Mamba Enc.  │ ──────────────►  │  BC Actor   │  │
│  └─────────────┘                  └──────┬──────┘  │
│                                          │ raw a_t  │
│                                          ▼          │
│                                  ┌──────────────┐  │
│                                  │ CBF Safety   │  │
│                                  │ Filter       │  │
│                                  └──────┬───────┘  │
│                                         │ safe â_t │
│                                         ▼          │
│                                    PyBaMM SPMe      │
└─────────────────────────────────────────────────────┘
```

---

## Key Results

| Method | 25 °C Time (min) | 35 °C Time (min) | T-viol (35 °C) |
|---|---|---|---|
| CC-CV 4C (rule) | 11.6 ± 0.4 | † early stop | — |
| BC-Mamba+NoSafety | 11.1 ± 0.7 | 22.8 ± 2.9 | 81.4% |
| BC-GRU+CBF | 11.1 ± 0.4 | 39.8 ± 1.9 | 0.0% |
| BC-Mamba+Sig | 11.4 ± 0.6 | 36.2 ± 4.0 | 0.0% |
| **BC-Mamba+CBF (Ours)** | **11.3 ± 0.5** | **36.2 ± 3.9** | **0.0%** |
| IQL-Mamba+CBF | 10.4 ± 0.5 | — | 3.5% |
| CQL-Mamba+CBF | 13.0 ± 1.2 | — | 63.7% |

†Rule-based methods terminate early at 35 °C before reaching 80% SOC target.

---

## Installation

```bash
# 1. Create conda environment
conda create -n mamba2 python=3.10 -y
conda activate mamba2

# 2. Install PyTorch (CUDA 11.8)
pip install torch==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. Install mamba-ssm
pip install mamba-ssm==2.2.2

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** `mamba-ssm` requires a CUDA-capable GPU. For CPU-only testing, replace `MambaEncoder` with `GRUEncoder` in `models/encoders/`.

---

## Project Structure

```
bc-mamba-safe-charging/
├── envs/                    # PyBaMM SPMe simulation environment
│   ├── pybamm_env.py        #   Main Gym-compatible environment
│   ├── battery_env.py       #   Low-level battery step logic
│   ├── constraints.py       #   Safety constraint definitions
│   └── reward.py            #   Reward function
├── models/
│   ├── encoders/
│   │   ├── mamba_encoder.py #   Mamba SSM encoder (main)
│   │   └── mamba_backend.py #   Low-level Mamba block
│   ├── safety/
│   │   ├── cbf_safety.py    #   CBF dual-layer safety filter
│   │   └── action_projection.py
│   ├── rl/                  #   IQL / CQL offline RL heads
│   └── heads/               #   BC actor head
├── trainers/
│   ├── train_bc.py          #   Behaviour cloning trainer
│   ├── train_iql.py         #   IQL offline RL trainer
│   └── train_cql.py         #   CQL offline RL trainer
├── datasets/
│   ├── generate_pybamm_dataset.py  # Expert dataset generation
│   └── build_offline_dataset.py    # Dataset preprocessing
├── evaluators/
│   └── evaluate.py          #   Evaluation loop + metrics
├── utils/                   #   Logging, seeding, metrics
├── configs/                 #   YAML hyperparameter configs
├── train.py                 #   Main training entry point
├── generate_dataset.py      #   Expert dataset generation
├── demo.py                  #   Quick demo (1 episode)
└── requirements.txt
```

---

## Quick Start

### Step 1 — Generate expert dataset

```bash
conda activate mamba2
python generate_dataset.py \
    --n_episodes 64 \
    --ambient_temp 25.0 \
    --output_dir data/processed_pybamm_expert
```

### Step 2 — Train BC-Mamba+CBF

```bash
python train.py \
    --data_dir data/processed_pybamm_expert \
    --encoder mamba \
    --safety cbf \
    --margin_T 1.5 \
    --epochs 150 \
    --seeds 42 123 456 789 1024 \
    --output_dir results/bc_mamba_cbf
```

### Step 3 — Evaluate

```bash
python evaluate_generalization.py \
    --checkpoint results/bc_mamba_cbf/seed42/encoder_best.pt \
    --ambient_temp 25.0 35.0 \
    --n_episodes 5
```

### Demo (single episode, no checkpoint needed)

```bash
# Nominal 25 °C
python demo.py

# Thermal stress 35 °C
python demo.py --ambient_temp 35.0

# With trained checkpoint
python demo.py --ambient_temp 35.0 --checkpoint results/bc_mamba_cbf/seed42/encoder_best.pt
```

Expected output (random weights — architecture demo only):
```
=======================================================
  BC-Mamba+CBF Demo  |  T_amb=25.0°C  |  SOC0=0.04
=======================================================
 Step    SOC   Temp(°C)  Volt(V)  C-rate  CBF?
-------------------------------------------------------
    5  0.142      33.81    3.924    3.85      
   10  0.278      35.20    3.971    3.72      
   ...
=======================================================
  Final SOC       : 0.804
  T-violations    : 0 (0.0%)
  CBF interventions: 3 (8.1%)
=======================================================
```

---

## CBF Safety Layer

The Control Barrier Function safety filter enforces $T_t \leq T_{lim} = 38$°C via:

$$I_{cbf}(T_t) = \frac{(k_{cool} + \alpha_c)(T_{lim} - T_t)}{k_{heat}}$$

with a soft Sigmoid pre-scaling when $T_t > T_{lim} - m_T$:

$$s = 1 - 0.5 \cdot \sigma\!\left(\frac{T_t - (T_{lim} - m_T)}{0.5}\right), \quad \hat{a}_t = \min(a_t \cdot s,\; I_{cbf}(T_t))$$

An analogous voltage Sigmoid with margin $m_V = 0.15$ V is applied in parallel.  
Key parameters: $k_{heat}=0.09$, $k_{cool}=0.02$, $\alpha_c=0.8$, $m_T=1.5$°C.

---

## Citation

If you use this code, please cite:

```bibtex
@article{bc_mamba_cbf_2026,
  title   = {Safe Fast Charging of Lithium-Ion Batteries via Behaviour Cloning
             with Mamba Encoder and Control Barrier Function Safety Filter},
  journal = {Journal of Energy Storage},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The PyBaMM simulation environment is subject to the [PyBaMM BSD 3-Clause License](https://github.com/pybamm-team/PyBaMM/blob/develop/LICENSE).
