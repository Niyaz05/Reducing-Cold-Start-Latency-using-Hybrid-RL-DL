# CASR+ — Adaptive Cold Start Latency Reduction in Serverless Computing

> **3rd Year B.Tech – Computer Science (AIML Specialization)**  
> Based on: *CASR: Optimizing cold start and resources utilization in serverless computing* — Yu Chen et al., Future Generation Computer Systems, 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is a Cold Start?](#2-what-is-a-cold-start)
3. [Dataset](#3-dataset)
4. [Project Architecture](#4-project-architecture)
5. [S-Cache: Container Keep-Alive Strategy](#5-s-cache-container-keep-alive-strategy)
6. [Reinforcement Learning Formulation](#6-reinforcement-learning-formulation)
7. [Reward Function — Deep Dive](#7-reward-function--deep-dive)
8. [PPO Agent — Architecture and Hyperparameters](#8-ppo-agent--architecture-and-hyperparameters)
9. [Meta-model — Dynamic Weight Prediction](#9-meta-model--dynamic-weight-prediction)
10. [File Structure](#10-file-structure)
11. [Installation](#11-installation)
12. [Usage](#12-usage)
13. [Evaluation and Baselines](#13-evaluation-and-baselines)
14. [Expected Results](#14-expected-results)
15. [Glossary](#15-glossary)
16. [References](#16-references)

---

## 1. Project Overview

**CASR+** is an extended implementation of the **CASR (Cache-Based Adaptive Scheduler for Serverless Runtime)** algorithm. The system reduces cold start latency in serverless computing using **deep reinforcement learning** — specifically **Proximal Policy Optimisation (PPO)**.

### Key Innovations Beyond the Original CASR Paper

| Feature | Original CASR | CASR+ (This Project) |
|---|---|---|
| Reward weights | Fixed θ = 0.8 for all workloads | Dynamically predicted per workload by Meta-model |
| Penalty terms | R1 (cold starts) + R2 (WMT) | + R3 (oscillation) + R4 (overhead-weighted CSO) + SLA sigmoid |
| Weight tuning | Manual | Automated via XGBoost meta-learning loop |

### High-Level Goals

| Goal | Target | Method |
|---|---|---|
| Reduce cold start rate | Below 15% | PPO scales S-Cache queues adaptively |
| Reduce memory waste | Minimise WMT | Reward penalises idle container time |
| Optimise heavy cold starts | Reduce CSO | R4 penalty weighted by overhead duration |
| Avoid erratic scaling | Low oscillation | R3 oscillation penalty |
| Enforce SLA | Rate < threshold | Smooth sigmoid SLA penalty |
| Auto-tune reward weights | Workload-adaptive | XGBoost meta-model |

---

## 2. What is a Cold Start?

In **serverless computing (Functions as a Service / FaaS)**, every function runs inside a container. When a function is called for the first time — or after being idle for too long — the platform must:

1. Download the function's code and dependencies
2. Create and initialise a new container
3. Bootstrap the runtime environment

This process adds **100 ms to over 60 seconds** of latency before the function even begins executing. This delay is called a **cold start**. Frequent cold starts directly harm user experience and violate SLA agreements.

### The Solution: Container Keep-Alive with RL

The standard mitigation is to **keep containers in memory** after execution so they can be reused on the next call (warm start). However, keeping too many containers **wastes memory**.

CASR+ solves this with a three-component system:

```
┌──────────────────────────────────────────────────────────┐
│                        CASR+ System                      │
│                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌───────────────┐  │
│  │   S-Cache   │   │  PPO Agent  │   │  Meta-model   │  │
│  │  (3 queues) │◄──│  (PyTorch)  │◄──│  (XGBoost)    │  │
│  │  Q0/Q1/Q2   │   │  27 actions │   │  θ,α,β,γ pred │  │
│  └─────────────┘   └─────────────┘   └───────────────┘  │
└──────────────────────────────────────────────────────────┘
```

- **S-Cache**: A three-queue caching strategy organised by cold start overhead
- **PPO Agent**: A deep RL model that dynamically scales each queue's capacity
- **Meta-model** *(our addition)*: Predicts optimal reward weights for the current workload

---

## 3. Dataset

### Azure Functions 2019 Public Dataset

The dataset used is the **Microsoft Azure Functions public dataset from 2019**, referenced as [41] in the CASR paper. It contains **anonymised telemetry** from Azure's production serverless platform over a **14-day period**, covering **over 50,000 functions**.

> **Dataset available at:** [https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset](https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset)

### File Format

Your dataset folder should contain two types of files:

| File Type | Column Pattern | What It Contains |
|---|---|---|
| Invocation traces | `HashOwner, HashApp, HashFunction, 1, 2, ..., 1440` | Invocations per minute for each function over a full day (1440 minutes) |
| Duration percentiles | `HashOwner, HashApp, HashFunction, Average, 1, 25, 50, ..., 100` | Cold start overhead duration statistics per function |

> **Note:** If only invocation files are present (no duration file), the loader automatically simulates `cold_start_ms` values using a log-normal mixture calibrated to the Azure dataset distribution cited in the CASR paper.

### Preprocessing

All preprocessing is handled automatically by `dataset_loader.py`. Steps performed:

1. Scans the data folder for all `.xlsx` and `.csv` files
2. Identifies invocation files vs duration files by column structure
3. Melts invocation files from wide format (1440 columns) to long format (one row per function-minute)
4. Extracts day number from the filename (e.g., `d01`, `day_01`)
5. Merges duration data as `cold_start_ms`; simulates if missing
6. Simulates `memory_mb` per function, correlated with cold start overhead
7. Samples up to **300,000 rows** for training (configurable)

### Final DataFrame Schema

| Column | Type | Meaning |
|---|---|---|
| `func_id` | string | Anonymised function hash identifier |
| `day` | integer | Day number (1–14) from filename |
| `arrival_minute` | integer | Minute of the day (1–1440) |
| `count` | integer | Number of invocations in that minute |
| `cold_start_ms` | float | Cold start overhead duration in milliseconds |
| `memory_mb` | float | Simulated memory footprint of the container |

### Simulated Cold Start Distribution

When no duration file is available, `cold_start_ms` is simulated from a log-normal mixture matching the Azure dataset's overhead bucket statistics:

| Bucket | Range | Proportion | Centred Around |
|---|---|---|---|
| Q0 | 0 – 1,000 ms | ~55% of functions | ~500 ms |
| Q1 | 1,000 – 60,000 ms | ~30% of functions | ~10,000 ms |
| Q2 | > 60,000 ms | ~15% of functions | ~120,000 ms |

---

## 4. Project Architecture

```
casr_project/
├── data/                  ← Put ALL Azure dataset files here
├── checkpoints/           ← Saved model weights (auto-created)
├── results/               ← Training logs & evaluation CSV (auto-created)
├── run.py                 ← Entry point (train / eval)
├── train.py               ← Main training pipeline
├── evaluate.py            ← Evaluation vs baselines
├── environment.py         ← CASREnv (Gymnasium-compatible RL env)
├── ppo_agent.py           ← PPO Actor-Critic agent (PyTorch)
├── meta_model.py          ← XGBoost meta-model for weight prediction
├── reward.py              ← Extended reward function (R1–R4 + SLA)
├── scache.py              ← S-Cache simulation (3 queues)
├── dataset_loader.py      ← Azure dataset loader & preprocessor
└── requirements.txt       ← Python dependencies
```

### Training Data Flow (one episode)

```
1.  Dataset Loader reads Azure Excel/CSV files from data/ folder
2.  Workload features (10 scalars) extracted from episode batch
3.  Meta-model predicts θ, α, β, γ for this workload
4.  CASREnv initialised with predicted weights
5.  For each RL step:
    agent → selects action → env processes STEP_SIZE invocations
6.  S-Cache routes each invocation to Q0/Q1/Q2 based on cold_start_ms
7.  Reward Calculator computes all 5 penalty terms and combines them
8.  PPO Agent stores transition (state, action, reward, log_prob, value)
9.  Every 200 steps: PPO update runs 10 epochs over collected transitions
10. At episode end: outcome recorded; meta-model retrained every 10 episodes
```

---

## 5. S-Cache: Container Keep-Alive Strategy

S-Cache is the core caching mechanism from the CASR paper. It manages containers in **three independent queues**, each for functions whose cold start duration falls within a specific range.

### Queue Definitions

| Queue | Cold Start Range | Typical Functions | Initial Capacity |
|---|---|---|---|
| **Q0** | 0 – 1 second | HTTP / lightweight functions | 200 containers |
| **Q1** | 1 – 60 seconds | Medium-weight, library-heavy | 100 containers |
| **Q2** | > 60 seconds | ML dependencies, large frameworks | 50 containers |

### W-TinyLFU Eviction Policy

Each queue uses a **W-TinyLFU inspired** structure:

- **Window cache** (20% of capacity): Absorbs new and bursty arrivals using LRU
- **Main cache** (80% of capacity): Stores frequently reused containers; promoted from window via Count-Min Sketch frequency comparison
- **Count-Min Sketch**: Probabilistic data structure that tracks approximate invocation frequency in O(1) time
- **Freshness mechanism**: The sketch periodically halves all counts to deprioritise stale access patterns

### Container Availability — Key CASR Insight

Unlike regular web caches, a container that is **cached but currently loading** (running) cannot serve a new request:

```
C_available(ξ) = C_mem(ξ) - C_running(ξ)
```

This is why Q2 requires special treatment: a function with a 60-second cold start remains unavailable for 60 seconds even after being triggered, meaning all instances can be busy simultaneously.

### Scaling Action

The RL agent adjusts queue capacities using a **scaling ratio ξ = 0.25** (25% per step):

| Action Component | Effect |
|---|---|
| `0` (scale down) | `new_cap = cap × 0.75` |
| `1` (hold) | No change |
| `2` (scale up) | `new_cap = cap × 1.25` |

Minimum capacity is always 1 to prevent complete queue depletion.

---

## 6. Reinforcement Learning Formulation

This project uses a **standard MDP (Markov Decision Process)** formulation with the Gymnasium interface.

### State Space — 21 Dimensions

The state vector contains **7 features × 3 queues** = 21 total dimensions. All features are normalised online using running mean/std and clipped to [-5, 5].

| Dimension | Feature | Description |
|---|---|---|
| 0, 7, 14 | `cap` | Maximum containers allowed in this queue |
| 1, 8, 15 | `current_len` | Containers currently cached |
| 2, 9, 16 | `invocations` | Invocations processed in this RL step |
| 3, 10, 17 | `cold_starts` | Cold starts that occurred in this step |
| 4, 11, 18 | `evictions` | Containers evicted in this step |
| 5, 12, 19 | `running` | Containers currently in cold-start phase (unavailable) |
| 6, 13, 20 | `wmt` | Wasted memory time accumulated in this step |

### Action Space — 27 Discrete Actions

Each action is a **3-tuple** — one scaling decision per queue. Each decision is:
- `0` = scale down
- `1` = hold
- `2` = scale up

With 3 queues × 3 choices each: **3³ = 27 unique action combinations**

### Episode and Step

| Term | Definition |
|---|---|
| **Episode** | One full pass through the sampled dataset |
| **RL Step** | `STEP_SIZE` (default: 500) invocations processed, then one agent action |

### Algorithm Overview

```
Algorithm: CASR+ Training Loop
────────────────────────────────────────────────────────────────
For each episode (1 to N_EPISODES):
  1. Sample episode data from Azure dataset
  2. Extract workload features (10 scalars)
  3. If meta-model trained:
       weights = meta_model.predict(features)
     Else:
       weights = random(θ, α, β, γ)   ← exploration/warmup phase

  4. Initialise CASREnv(data, weights)
  5. state = env.reset()

  6. While not done:
       action, log_prob, value = ppo_agent.select_action(state)
       next_state, reward, done, info = env.step(action)
       buffer.store(state, action, log_prob, reward, next_state, done, value)
       state = next_state

       If buffer.count >= UPDATE_INTERVAL or done:
           ppo_agent.update()          ← PPO update (10 epochs)

  7. meta_model.record_episode(features, weights, outcome)
  8. If episode % META_TRAIN_INTERVAL == 0:
       meta_model.train()              ← Retrain XGBoost
────────────────────────────────────────────────────────────────
```

---

## 7. Reward Function — Deep Dive

The reward function is the most important design decision. It is the **only signal the agent uses to learn** what behaviour is desirable. A poorly designed reward leads to the agent optimising the wrong objective even with a perfect architecture.

### 7.1 CASR Base Reward (Original Paper)

The original CASR paper defines its reward as (Equation 6):

```
Reward = -(θ × norm(R1) + (1 - θ) × norm(R2))
```

The **negative sign** means the agent maximises reward by minimising both terms. Both R1 and R2 are normalised to [0, 1] using running min/max statistics so they remain on the same scale regardless of workload size.

### 7.2 Extended CASR+ Reward (This Project)

```
Reward = -(θ·norm(R1)  +  (1-θ)·norm(R2))
         - α·norm(R3)
         - β·SLA_sigmoid
         - γ·norm(R4)
```

### 7.3 Every Term Explained

#### R1 — Cold Start Count
Total cold starts across all queues in the current RL step.
```
R1 = cold_starts_Q0 + cold_starts_Q1 + cold_starts_Q2
```
More cold starts = worse. This is the **primary latency signal**.

#### R2 — Wasted Memory Time (WMT)
Change in Wasted Memory Time since the previous step. WMT = total time (ms) containers spent idle in memory without serving any request.
```
R2 = ΔWMT = current_WMT - previous_WMT
```
A container kept alive for 10 minutes that never gets a request wastes 600,000 ms of memory-time.

#### R3 — Oscillation Penalty *(Our Addition)*
Measures erratic capacity changes. Penalises the agent for flip-flopping between scale-up and scale-down without improving performance.
```
R3 = Σ |new_cap_k - old_cap_k|   for k in {Q0, Q1, Q2}
```

#### R4 — Cold Start Overhead (CSO) Penalty *(Our Addition)*
The CASR paper defines Cold Start Overhead in Equation 3:
```
R4 = CSO = Σ (cold_start_overhead_ms_k × cold_starts_k)
```
A function with 60-second cold start overhead is penalised **60× more** than one with 1-second overhead. Without R4, the agent treats all cold starts equally and may neglect to keep heavy ML-dependency functions warm.

#### SLA Penalty — Smooth Sigmoid
Fires when cold start rate approaches or exceeds the threshold (default 15%):
```
SLA = scale / (1 + exp(-slope × (cold_start_rate - threshold)))
```
Using a smooth sigmoid (instead of a hard step function) provides a gradient signal **even before** the threshold is breached, helping the agent learn to stay safely below the SLA limit.

### 7.4 Reward Weight Parameters

| Symbol | What It Measures | Range | Default | Effect of Increasing |
|---|---|---|---|---|
| **θ (theta)** | Cold starts vs memory trade-off | 0 – 1 | **0.8** | Agent cares more about cold starts |
| **α (alpha)** | Oscillation penalty weight | 0 – 0.5 | **0.1** | Agent scales more gradually |
| **β (beta)** | SLA violation penalty weight | 0 – 0.8 | **0.3** | Agent enforces SLA more strictly |
| **γ (gamma)** | Overhead-weighted cold start weight | 0 – 0.5 | **0.2** | Agent prioritises heavy Q2 functions |

#### Theta (θ) — Main Trade-off Weight
- `θ = 0.8` → 80% weight on cold starts, 20% on memory (**CASR default; agent cares mostly about latency**)
- `θ = 0.2` → 20% weight on cold starts, 80% on memory (agent aggressively evicts containers)
- The Meta-model **predicts θ dynamically per workload** rather than fixing it at 0.8

#### Alpha (α) — Oscillation Weight ("patience dial")
- `α = 0.0` → Ignore oscillation; agent may flip-flop every step
- `α = 0.1` ← *Default*: Light penalty; discourages unnecessary oscillation without over-constraining the agent
- `α = 0.4` → Heavy penalty; agent prefers gradual, consistent scaling over rapid changes

#### Beta (β) — SLA Weight ("consequence dial")
- `β = 0.1` → SLA is a soft suggestion; agent may breach it for better memory savings
- `β = 0.3` ← *Default*: Balanced; treats SLA as important but not an absolute constraint
- `β = 0.8` → SLA is a hard constraint; agent sacrifices memory efficiency to stay below threshold

#### Gamma (γ) — CSO Weight ("heavy function priority dial")
- `γ = 0.0` → Ignore overhead heterogeneity; same as naive cold start count
- `γ = 0.2` ← *Default*: Agent mildly prefers keeping heavy functions warm
- `γ = 0.5` → Agent strongly prioritises Q2 (>60s) functions to avoid costly cold starts

---

## 8. PPO Agent — Architecture and Hyperparameters

### 8.1 Network Architecture

Both Actor and Critic use **orthogonal weight initialisation** and are trained with the **Adam optimiser**. Architecture matches CASR paper Table 2.

| Component | Input | Hidden | Output | Activation |
|---|---|---|---|---|
| **Actor** | 21-dim state | 128 neurons | 27-class softmax | Tanh |
| **Critic** | 21-dim state | 128 neurons | 1 scalar value | Tanh |

```
State (21-dim)
     │
     ├──► Actor  ──► Linear(21→128) ──► Tanh ──► Linear(128→27) ──► Softmax ──► π(a|s)
     │
     └──► Critic ──► Linear(21→128) ──► Tanh ──► Linear(128→1)  ──────────────► V(s)
```

- **Actor**: Outputs a probability distribution over 27 actions. The agent samples from this to pick an action. Over training, high-reward actions become more probable.
- **Critic**: Outputs a single number — the estimated total future reward from the current state. Used to compute the advantage: was the action taken better or worse than average?

### 8.2 PPO Hyperparameters (from CASR Table 2)

| Parameter | Value | Reason |
|---|---|---|
| Discount factor (γ_RL) | **0.63** | Low value: workload changes fast, future is less predictable |
| GAE lambda (λ) | **0.95** | Balances bias and variance in advantage estimation |
| Clip epsilon (ε) | **0.2** | Maximum policy change per update step |
| Entropy coefficient (ρ) | **0.01** | Encourages exploration; prevents premature convergence |
| Learning rate (actor) | **0.001** | Low rate prevents gradient explosion |
| Learning rate (critic) | **0.001** | Matches actor rate for stable training |
| Epochs per update | **10** | Number of passes over collected trajectories |
| Mini-batch size | **20** | Small batches from CASR paper Table 2 |
| Buffer size | **1000** | Rollout buffer capacity in steps |

> **Important:** The RL discount factor `γ = 0.63` (used in GAE) is **separate** from the reward weight `γ` (CSO penalty). The notation is reused from the CASR paper.

### 8.3 PPO-Clip Objective

PPO limits policy updates using a clipped surrogate objective (Equation 8 in CASR paper):

```
L_clip(θ) = E[ min(r_t(θ)·A_t,   clip(r_t(θ), 1-ε, 1+ε)·A_t) ]
```

Where:
- `r_t = π_new(a|s) / π_old(a|s)` — probability ratio
- `A_t` — advantage estimate (was the action better or worse than average?)
- `ε = 0.2` — clip range

The clip prevents too-large updates that would destabilise training.

### 8.4 Entropy Bonus

An entropy bonus is added to encourage exploration (Equations 11–12 in CASR paper):

```
L(θ) = -L_clip(θ)  +  ρ·H(π(·|s_t))
H(π(·|s_t)) = E[-log π(a_t|s_t)]
```

### 8.5 Generalised Advantage Estimation (GAE-λ)

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ_{k=0}^{T} (γ·λ)^k · δ_{t+k}
```

Advantages are normalised before each PPO update:

```
A_t ← (A_t - mean(A)) / (std(A) + ε)
```

---

## 9. Meta-model — Dynamic Weight Prediction

The Meta-model is the **key contribution beyond the original CASR paper**. It predicts the optimal reward weights (θ, α, β, γ) for each episode based on observed workload characteristics.

### 9.1 Motivation

The CASR paper uses a **fixed θ = 0.8** for all workloads. This works on average but is suboptimal for specific workload types:

| Workload Type | Optimal Strategy |
|---|---|
| Bursty with many heavy functions | High θ (punish cold starts) + high γ (prioritise Q2) |
| Stable low-traffic periods | Lower θ (memory efficiency matters more) |
| Irregular calling patterns | Higher β (enforce SLA even when agent prefers eviction) |

The Meta-model **learns these associations from the data** rather than requiring manual tuning.

### 9.2 Input Features (10 Dimensions)

| Feature | What It Captures |
|---|---|
| `mean_invocations_per_minute` | Overall traffic volume |
| `std_invocations_per_minute` | Burstiness of traffic |
| `mean_cold_start_ms` | Average overhead severity |
| `std_cold_start_ms` | Heterogeneity of overhead |
| `frac_low_overhead` | Fraction of Q0 (0–1s) functions |
| `frac_medium_overhead` | Fraction of Q1 (1–60s) functions |
| `frac_high_overhead` | Fraction of Q2 (>60s) functions |
| `mean_memory_mb` | Average memory footprint |
| `n_unique_functions` | Diversity of the function set |
| `total_invocations` | Total volume of the batch |

### 9.3 Meta-learning Training Loop

```
Phase 1 — Warmup (first 20 episodes):
  • Random weights (θ, α, β, γ) are used for exploration
  • Record: (workload_features, weights_used, cold_start_rate, total_WMT)

Phase 2 — Training (after 20 episodes):
  • XGBoost MultiOutputRegressor trained on collected (X → y) data
  • Learns: workload_features → (θ, α, β, γ)

Phase 3 — Inference (subsequent episodes):
  • Meta-model predicts weights from current workload features
  • Weights passed to CASREnv before episode begins

Phase 4 — Periodic Retraining:
  • Meta-model retrained every 10 episodes as more data accumulates
```

### 9.4 Why XGBoost?

| Reason | Explanation |
|---|---|
| **Small dataset** | Only 20–200 episodes of training data; neural networks need far more |
| **Fast training** | Trains in milliseconds; minimal overhead to the RL loop |
| **Interpretable** | Feature importances reveal which workload characteristics influence optimal weight selection |
| **Robust** | Handles correlated features and small samples without overfitting |

### 9.5 Meta-model Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | XGBoost MultiOutputRegressor |
| Estimators | 100 |
| Max depth | 4 |
| Learning rate | 0.1 |
| Subsample | 0.8 |
| Feature preprocessing | StandardScaler |
| Warmup episodes | 20 |
| Retraining interval | Every 10 episodes |
| Output clipping (θ) | [0.10, 0.95] |
| Output clipping (α) | [0.00, 0.50] |
| Output clipping (β) | [0.00, 0.80] |
| Output clipping (γ) | [0.00, 0.50] |

---

## 10. File Structure

| File | Role |
|---|---|
| `run.py` | Entry point; dispatches to train or evaluate |
| `train.py` | Main training loop (Algorithm 2 from CASR paper) |
| `evaluate.py` | Evaluation against baselines; saves results CSV |
| `environment.py` | `CASREnv` — Gymnasium-compatible RL environment |
| `ppo_agent.py` | PPO Actor-Critic agent with rollout buffer |
| `meta_model.py` | XGBoost meta-model for dynamic weight prediction |
| `reward.py` | `RewardCalculator` and `RewardWeights` dataclass |
| `scache.py` | `SCache` + `SCacheQueue` simulation |
| `dataset_loader.py` | Azure dataset loader, preprocessor, and feature extractor |
| `requirements.txt` | Python dependencies |

---

## 11. Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Navigate to the project folder
cd niyaz_ug

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
gymnasium>=0.29.0
```

### Dataset Setup

1. Download the Azure 2019 dataset from Kaggle:
   [https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset](https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset)

2. Place all downloaded `.csv` or `.xlsx` files inside a `./data/` folder:

```
niyaz_ug/
└── data/
    ├── invocations_per_function_md.anon.d01.csv
    ├── invocations_per_function_md.anon.d02.csv
    ├── function_durations_percentiles.anon.d01.csv
    └── ...
```

---

## 12. Usage

### Training

```bash
# Quick start (default: 200 episodes, 300,000 rows sampled)
python run.py --mode train --data_dir ./data

# Custom settings
python train.py \
  --data_dir ./data \
  --episodes 200 \
  --sample_rows 300000 \
  --step_size 500 \
  --update_interval 200 \
  --meta_train_interval 10 \
  --sla_threshold 0.15 \
  --seed 42
```

### Evaluation

```bash
python run.py --mode eval --data_dir ./data

# Or directly with full options:
python evaluate.py \
  --data_dir ./data \
  --model_path ./checkpoints/ppo_latest.pt \
  --meta_model_path ./checkpoints/meta_model.pkl \
  --sample_rows 100000
```

### Training Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `../data` | Folder with Azure dataset files |
| `--episodes` | `200` | Total training episodes |
| `--sample_rows` | `300000` | Max rows loaded per run |
| `--step_size` | `500` | Invocations processed per RL step (delta) |
| `--update_interval` | `200` | Steps between PPO updates |
| `--meta_train_interval` | `10` | Episodes between meta-model retraining |
| `--sla_threshold` | `0.15` | Cold start rate SLA limit (15%) |
| `--seed` | `42` | Random seed for reproducibility |
| `--save_every` | `20` | Episodes between checkpoint saves |

### Monitoring Training

The training loop prints one line per episode:

```
Ep  001 | Reward= -0.245 | CS_rate=0.182 | θ=0.76 α=0.12 β=0.28 γ=0.19 | MetaTrained=False
Ep  021 | Reward= -0.180 | CS_rate=0.143 | θ=0.82 α=0.09 β=0.31 γ=0.21 | MetaTrained=True
```

- Once `MetaTrained=True` (after episode 20+), weights are predicted by the meta-model rather than sampled randomly.
- Results are logged to `results/training_log.jsonl` and saved as `results/training_results.csv`.

---

## 13. Evaluation and Baselines

The evaluation script compares CASR+ against three baselines:

| Method | Description |
|---|---|
| **Fixed** | Containers kept alive with a fixed capacity of 350 (platform default simulation) |
| **LRU (LCS)** | Least Recently Used eviction, overhead-unaware, no scaling |
| **CASR base** | CASR with fixed θ = 0.8, no extended reward terms (α = β = γ = 0) |
| **CASR+ (ours)** | PPO + extended reward + meta-model predicted weights |

### Evaluation Metrics (CASR Paper Section 5.1.5)

| Metric | Formula |
|---|---|
| **Cold Start Rate** | `cold_starts / total_invocations` |
| **Avg CSO (ms)** | `total_cold_start_overhead_ms / total_invocations` |
| **Avg WMT (ms)** | `total_wasted_memory_time_ms / total_invocations` |

### Sample Output

```
Method             |  CS Rate |  Avg CSO (ms) |  Avg WMT (ms)
-----------------------------------------------------------
Fixed              |   0.2100 |       45321.0 |       12450.0
LRU (LCS)          |   0.1950 |       41200.0 |       11800.0
CASR base          |   0.1450 |       28900.0 |        9200.0
CASR+ (ours)       |   0.1280 |       24100.0 |        8600.0

CASR+ vs Fixed: +39.0% cold start rate improvement
```

---

## 14. Expected Results

### Performance Targets

| Metric | Fixed Baseline | Target (CASR+) |
|---|---|---|
| Cold Start Rate | ~21% | **< 15%** |
| CSO Improvement | Baseline | **> 35% reduction** |
| WMT Improvement | Baseline | **> 25% reduction** |

> **Note:** The CASR paper reports **38.75% cold start reduction vs Fixed** on common workloads. Our extended reward with dynamic weight prediction is expected to improve this marginally, particularly on heterogeneous workloads with many high-overhead functions.

### When to Consider the Model Successful

A cold start rate consistently **below 15%** on the Azure dataset, with improved CSO compared to Fixed and LRU, is sufficient to demonstrate effectiveness. Even a modest improvement over base CASR is a valid research contribution because:

1. **Dynamic weight prediction** (Meta-model) is a novel addition not present in the original paper
2. The **CSO penalty (R4)** directly addresses the cold start overhead metric that the CASR paper highlights as important
3. The **smooth SLA sigmoid** provides better gradient signal than a hard threshold

---

## 15. Glossary

| Term | Definition |
|---|---|
| **Cold Start** | Latency when a container must be initialised from scratch (no warm instance available) |
| **Warm Start** | Reuse of an already-running container, avoiding initialisation overhead |
| **S-Cache** | Three-queue container caching strategy from CASR paper |
| **WMT (Wasted Memory Time)** | Total time idle containers occupy memory without serving requests |
| **CSO (Cold Start Overhead)** | Overhead-weighted cold start penalty: sum of (overhead_ms × cold_starts) |
| **SLA** | Service Level Agreement; maximum acceptable cold start rate (default 15%) |
| **PPO** | Proximal Policy Optimisation; on-policy RL algorithm using clipped surrogate objective |
| **GAE** | Generalised Advantage Estimation; method to compute policy gradient advantages |
| **Meta-model** | XGBoost model that predicts optimal reward weights (θ, α, β, γ) per workload |
| **W-TinyLFU** | Window TinyLFU; eviction policy combining LRU window with frequency-based main cache |
| **Count-Min Sketch** | Probabilistic data structure for frequency approximation in O(1) time |
| **Q0 / Q1 / Q2** | S-Cache queues for functions with 0–1s / 1–60s / >60s cold start overhead |
| **ξ (xi)** | Scaling ratio for queue capacity adjustments (default 0.25) |
| **θ (theta)** | Primary reward weight: cold starts vs memory trade-off |
| **α (alpha)** | Oscillation penalty weight |
| **β (beta)** | SLA violation penalty weight |
| **γ (gamma)** | CSO penalty weight (overhead-weighted cold starts) |
| **MDP** | Markov Decision Process; mathematical framework for sequential decision making |
| **Actor-Critic** | RL architecture with separate policy network (actor) and value network (critic) |
| **Rollout Buffer** | Storage for experience transitions collected during environment interaction |

---

## 16. References

1. **Yu Chen, Bo Liu, Weiwei Lin, Yulin Guo, Zhiping Peng.**  
   *"CASR: Optimizing cold start and resources utilization in serverless computing."*  
   Future Generation Computer Systems, Volume 170, 2025, Article 107851.  
   DOI: [https://doi.org/10.1016/j.future.2025.107851](https://doi.org/10.1016/j.future.2025.107851)

2. **Microsoft Azure Functions Public Dataset (2019):**  
   [https://github.com/Azure/AzurePublicDataset/](https://github.com/Azure/AzurePublicDataset/)  
   Kaggle mirror: [https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset](https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset)

3. **Schulman et al. (2017).**  
   *Proximal Policy Optimization Algorithms.* arXiv:1707.06347

4. **Schulman et al. (2015).**  
   *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438

5. **Chen et al. (2016).**  
   *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD.

---

*CASR+ · 3rd Year B.Tech AIML · PPO + XGBoost + Gymnasium + PyTorch*
