# Reinforcement Learning for On-Demand Taxi Dynamics

> **Master's Dissertation** — Nanyang Technological University  
> Multi-Agent Reinforcement Learning for Real-Time Fleet Rebalancing over Singapore's Road Network

---

## Architecture — Centralized Training, Decentralized Execution (CTDE)

```
                     ┌─────────────────────────────────────────┐
                     │          Shared PPO Policy Network       │
                     │  ┌──────────┐  ┌──────────┐  ┌───────┐ │
                     │  │ FC (256) │→ │ FC (256) │→ │Softmax│ │
                     │  └──────────┘  └──────────┘  └───────┘ │
                     │         ↑ Action Mask (−∞ on invalid)   │
                     └──────────┬───────────────────────────────┘
                                │  Parameter Sharing
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
     ┌─────────┐          ┌─────────┐          ┌─────────┐
     │ Taxi  0 │          │ Taxi  1 │   ...    │ Taxi N-1│
     │ obs → a │          │ obs → a │          │ obs → a │
     └────┬────┘          └────┬────┘          └────┬────┘
          │                    │                    │
          ▼                    ▼                    ▼
    ┌──────────────────────────────────────────────────────┐
    │         H3-Discretised Singapore Road Network         │
    │         (Uber H3 Resolution 8 Hexagonal Grid)         │
    └──────────────────────────────────────────────────────┘
```

Every taxi is an independent agent but shares the **same policy network** (parameter sharing).
During training, all agents contribute gradients to one set of weights.
During execution, each taxi runs the policy locally using only its own observation — no central coordinator needed.

### Core Innovation — Density Penalty

The reward function includes a **density penalty** that creates anti-bunching pressure:

```
R = E[Revenue] − Travel Cost − α · max(0, supply/demand − 1)
```

When α = 0.5 (PPO-Ours), the policy learns to spatially disperse.
When α = 0.0 (PPO-Ablation), taxis cluster at hotspots — proving the penalty's necessity.

---

## Project Structure

```
├── data_pipeline.py          # Road network, H3 discretisation, tidal Poisson demand
├── cityflow_env.py           # Gymnasium env with CityFlow + action masking
├── train_rllib.py            # Ray RLlib PPO training (CTDE, --ablation flag)
├── evaluate_and_plot.py      # Multi-seed evaluation, ablation study, publication plots
├── kafka_producer.py         # Simulated real-time passenger request stream
├── decision_gateway.py       # RL inference gateway + WebSocket broadcast
├── dashboard/                # Next.js real-time dispatch visualisation
├── requirements.txt          # Python dependencies
├── Makefile                  # One-command pipeline orchestration
├── figures/                  # Generated publication-quality plots (300 DPI)
├── results/                  # Evaluation CSVs (multi-seed, spatial snapshots)
├── checkpoints_ours/         # Trained PPO model (α = 0.5)
└── checkpoints_ablation/     # Ablation baseline (α = 0.0)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for dashboard)
- CUDA GPU (optional, for faster training)

### Installation

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# One command — data → train both models → evaluate
make all

# Or step by step:
make data                      # Generate road network + tidal demand
make train-ours                # Train PPO with density penalty
make train-ablation            # Train PPO without density penalty (ablation)
make evaluate                  # Multi-seed evaluation + plots
```

### Launch the Dashboard

```bash
# Terminal 1: Start the RL dispatch WebSocket server
python decision_gateway.py --standalone

# Terminal 2: Start the dashboard
make dashboard
# → Open http://localhost:3000
```

---

## Observation Space (per agent)

| Slice | Feature |
|---|---|
| `[0 … H)` | One-hot H3 hex ID |
| `[H]` | Demand − Supply gap at current hex |
| `[H+1 … H+7)` | Demand − Supply gap at 6 neighbours |
| `[H+7]` | Idle vehicle count at current hex |
| `[H+8 … H+14)` | Idle vehicle count at 6 neighbours |
| `[H+14]` | Normalised time-of-day |

**Action Space**: `Discrete(7)` — 0 = stay, 1–6 = move to sorted H3 neighbour.

---

## Ablation Study Results

Evaluation over 3 random seeds (42, 1024, 2026) with 20 taxis on Downtown Core, Singapore:

| Algorithm | Mean ORR | Cumulative Reward | Matched Orders | Cruising Hops |
|---|---|---|---|---|
| **PPO-Ours (α=0.5)** | **0.405 ± 0.231** | **+4,465** | **4,535** | **1,230** |
| PPO-Ablation (α=0.0) | 0.172 ± 0.181 | +1,259 | 1,576 | 2,164 |
| Greedy | 0.214 ± 0.099 | +3,323 | 3,372 | 2,749 |
| Random | 0.337 ± 0.194 | +3,578 | 3,948 | 4,579 |

**Key finding**: Removing the density penalty (α = 0 → 0.5) causes the learned policy to cluster aggressively, performing worse than even a random baseline. The anti-bunching reward is essential.

---

## Tidal Demand Profile

Demand follows a bi-modal Gaussian tidal curve with peaks at 08:00 (AM rush) and 18:00 (PM rush), calibrated for Singapore's Downtown Core:

```
λ(t, h) = λ_base(h) × [ 0.15 + 2.8·𝒩(96, 22²) + 2.3·𝒩(216, 22²) + 0.7·𝒩(156, 36²) ]
```


---

## Tech Stack

| Layer | Technology |
|---|---|
| Spatial & Graph | `osmnx`, `h3-py`, `networkx`, `geopandas` |
| Simulation | `CityFlow` (macro traffic), Gymnasium |
| Reinforcement Learning | `Ray RLlib` (PPO), `torch` |
| Streaming | `kafka-python-ng`, `websockets` |
| Visualisation | `matplotlib`, `seaborn`, `deck.gl`, `Next.js` |

---

