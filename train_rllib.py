"""
train_rllib.py — Distributed RL Training with Ray RLlib  (CTDE)

Centralized Training / Decentralized Execution with Parameter Sharing
for multi-taxi dispatch over an H3-discretised Singapore road network.

Architecture
------------
* Every taxi is an independent *agent* inside a ``MultiAgentEnv``.
* All agents are mapped to a **single shared PPO policy** via
  ``policy_mapping_fn`` → full parameter sharing.
* Observation per agent:
      [H3 one-hot | demand-gap (self+6 nbrs) | idle-count (self+6 nbrs) | time]
* Action per agent:
      Discrete(7)  — stay + 6 hex directions  (invalid moves are masked)
* Reward per agent:
      R = E[Revenue] − TravelCost − DensityPenalty
  The **density penalty** is the dissertation's core innovation, creating
  a dispersive pressure that prevents spatial bunching of idle taxis.

Usage
-----
    python train_rllib.py                        # default: 50 iters, 20 taxis, 2 workers
    python train_rllib.py --iterations 200 --num-taxis 50 --num-workers 4

Author : <your-name>
Project: Reinforcement Learning for On-Demand Taxi Dynamics (NTU Dissertation)
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Suppress noisy warnings before heavy imports ──────────────────────────
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

import torch
import torch.nn as nn

from data_pipeline import DataPipeline, DemandConfig, H3Mapping
from cityflow_env import ActionMasker, NUM_DIRECTIONS, ACTION_DIM, ACTION_STAY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Quieten chatty Ray / RLlib loggers during training
for _log_name in ("ray", "ray.rllib", "ray.tune"):
    logging.getLogger(_log_name).setLevel(logging.WARNING)


# ========================================================================== #
#  1.  Reward Calculator                                                      #
# ========================================================================== #
@dataclass
class RewardConfig:
    """Three-component reward hyper-parameters.

    R_i = revenue_i − travel_cost_i − density_penalty_i − idle_cost_i

    * **revenue**         = base_fare × min(demand, supply) / supply
    * **travel_cost**     = cost_per_second × travel_time_per_hop  (if moved)
    * **density_penalty** = α × max(0, supply/demand − 1)          (core innovation)
    * **idle_cost**       = flat penalty when agent earns zero revenue
    """

    base_fare: float = 10.0                 # SGD per matched request
    cost_per_second: float = 0.005          # SGD per second of travel
    travel_time_per_hop: float = 90.0       # ~res-8 adj. hex at 30 km/h
    density_alpha: float = 0.5              # weight of density penalty
    idle_penalty: float = -0.10             # applied when revenue == 0


class RewardCalculator:
    """Compute per-agent rewards after all taxis have moved.

    The **density penalty** discourages agents from clustering in
    already-oversupplied hexagons.  It activates when supply exceeds
    demand at the *target* hex and scales with the surplus ratio::

        penalty = α × max(0,  supply_h / max(demand_h, 1) − 1)

    Because supply is recomputed *after* all moves, an agent that
    joins a crowd of others heading for the same hex incurs a larger
    penalty — a natural anti-bunching pressure.
    """

    def __init__(
        self,
        cfg: RewardConfig,
        hex_to_idx: Dict[str, int],
        num_hexes: int,
    ) -> None:
        self.cfg = cfg
        self._hex_to_idx = hex_to_idx
        self._num_hexes = num_hexes

    def compute(
        self,
        agent_ids: List[str],
        taxi_hexes: Dict[str, str],
        moved: Dict[str, bool],
        supply: np.ndarray,
        demand: np.ndarray,
    ) -> Dict[str, float]:
        """Return ``{agent_id: scalar_reward}`` for every agent.

        Parameters
        ----------
        agent_ids : list[str]
            Agents that acted this step.
        taxi_hexes : dict
            Current hex of each taxi (*after* moves).
        moved : dict
            Whether each taxi relocated this step.
        supply : np.ndarray, shape ``(H,)``
            Taxi count per hex (*after* moves).
        demand : np.ndarray, shape ``(H,)``
            Passenger request count per hex this step.
        """
        rewards: Dict[str, float] = {}

        for aid in agent_ids:
            idx = self._hex_to_idx.get(taxi_hexes[aid])
            if idx is None:
                rewards[aid] = 0.0
                continue

            s = float(supply[idx])
            d = float(demand[idx])

            # ── 1. Expected Revenue ───────────────────────────────────
            if d > 0 and s > 0:
                revenue = self.cfg.base_fare * min(d, s) / s
            else:
                revenue = 0.0

            # ── 2. Travel Cost ────────────────────────────────────────
            travel_cost = 0.0
            if moved.get(aid, False):
                travel_cost = (
                    self.cfg.cost_per_second * self.cfg.travel_time_per_hop
                )

            # ── 3. Density Penalty (CORE INNOVATION) ──────────────────
            density_ratio = s / max(d, 1.0)
            density_penalty = self.cfg.density_alpha * max(
                0.0, density_ratio - 1.0
            )

            # ── 4. Idle cost (nudge towards demand) ───────────────────
            idle_cost = self.cfg.idle_penalty if revenue == 0.0 else 0.0

            rewards[aid] = revenue - travel_cost - density_penalty + idle_cost

        return rewards


# ========================================================================== #
#  2.  Multi-Agent Environment (RLlib)                                        #
# ========================================================================== #
class TaxiDispatchMultiAgentEnv(MultiAgentEnv):
    """RLlib-compatible multi-agent taxi dispatch environment.

    Each of the *N* taxis is an independent agent sharing a single
    policy (CTDE with parameter sharing).

    Observation (per agent) — ``Dict``
    ----------------------------------
    ``"obs"``  : ``Box(num_hexes + 15,)``

        ================  ==========================================
        Slice             Feature
        ================  ==========================================
        [0 … H)           One-hot H3 hex ID
        [H]                Demand-gap at current hex  (demand−supply)
        [H+1 … H+7)       Demand-gap at 6 sorted neighbours
        [H+7]             Idle-vehicle count at current hex
        [H+8 … H+14)      Idle-vehicle count at 6 neighbours
        [H+14]            Normalised time-of-day
        ================  ==========================================

    ``"action_mask"``  : ``Box(7,)``  — 1.0 = valid action

    Action (per agent)
    ------------------
    ``Discrete(7)`` — 0 = stay, 1–6 = move to sorted neighbour.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        config = config or {}

        # ── Fleet / episode settings ──────────────────────────────────
        self.num_taxis: int = config.get("num_taxis", 20)
        self.max_steps: int = config.get("max_steps", 288)

        # ── Agent IDs ─────────────────────────────────────────────────
        self._agent_list: List[str] = [
            f"taxi_{i}" for i in range(self.num_taxis)
        ]
        self._agent_ids: Set[str] = set(self._agent_list)

        # ── Data pipeline (loads cached graph automatically) ──────────
        pipeline = DataPipeline(
            place=config.get("place", "Downtown Core, Singapore"),
            h3_resolution=config.get("h3_resolution", 8),
            demand_config=DemandConfig(
                seed=config.get("demand_seed", 42),
            ),
        ).run(time_steps=self.max_steps)

        self._active_hexes: List[str] = pipeline.h3_mapping.active_hexes
        self._hex_to_idx: Dict[str, int] = {
            h: i for i, h in enumerate(self._active_hexes)
        }
        self._num_hexes: int = len(self._active_hexes)
        self._masker = ActionMasker(self._active_hexes)
        self._demand_matrix: np.ndarray = (
            pipeline.demand.values.astype(np.float32)
        )
        self._graph = pipeline.graph

        # ── Reward calculator ─────────────────────────────────────────
        self._reward_calc = RewardCalculator(
            cfg=RewardConfig(
                base_fare=config.get("base_fare", 10.0),
                cost_per_second=config.get("cost_per_second", 0.005),
                travel_time_per_hop=config.get("travel_time_per_hop", 90.0),
                density_alpha=config.get("density_alpha", 0.5),
                idle_penalty=config.get("idle_penalty", -0.10),
            ),
            hex_to_idx=self._hex_to_idx,
            num_hexes=self._num_hexes,
        )

        # ── Per-agent observation & action spaces ─────────────────────
        self._obs_dim: int = self._num_hexes + 15
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._obs_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(ACTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # ── Mutable state (populated in reset) ────────────────────────
        self._taxi_hexes: Dict[str, str] = {}
        self._step_count: int = 0
        self._rng = np.random.default_rng(config.get("seed", 42))

    # ================================================================== #
    #  reset                                                               #
    # ================================================================== #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0

        for aid in self._agent_list:
            idx = int(self._rng.integers(0, self._num_hexes))
            self._taxi_hexes[aid] = self._active_hexes[idx]

        supply = self._compute_supply()
        demand = self._current_demand()

        obs_dict = {
            aid: self._build_agent_obs(aid, supply, demand)
            for aid in self._agent_list
        }
        info_dict = {aid: {} for aid in self._agent_list}
        return obs_dict, info_dict

    # ================================================================== #
    #  step                                                                #
    # ================================================================== #
    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[
        Dict[str, dict],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """One dispatch round (≈ 5 real-time minutes).

        1. Apply action masks  →  safety filter.
        2. Relocate taxis.
        3. Recompute supply.
        4. Compute three-component rewards (revenue, travel, density).
        5. Advance clock, build next observations.
        """
        # 1 ── Mask & resolve targets ─────────────────────────────────
        moved: Dict[str, bool] = {}
        for aid, action in action_dict.items():
            cur = self._taxi_hexes[aid]
            safe = self._masker.apply(cur, int(action))
            target = self._masker.resolve(cur, int(action))
            moved[aid] = safe != ACTION_STAY
            self._taxi_hexes[aid] = target

        # 2 ── Recompute spatial supply ────────────────────────────────
        supply = self._compute_supply()
        demand = self._current_demand()

        # 3 ── Three-component reward ─────────────────────────────────
        rewards = self._reward_calc.compute(
            agent_ids=list(action_dict.keys()),
            taxi_hexes=self._taxi_hexes,
            moved=moved,
            supply=supply,
            demand=demand,
        )

        # 4 ── Advance time ───────────────────────────────────────────
        self._step_count += 1
        is_truncated = self._step_count >= self.max_steps

        # 5 ── Next observations ───────────────────────────────────────
        next_supply = supply
        next_demand = self._current_demand()

        obs_dict: Dict[str, dict] = {}
        terminateds: Dict[str, bool] = {"__all__": False}
        truncateds: Dict[str, bool] = {"__all__": is_truncated}
        infos: Dict[str, dict] = {}

        for aid in action_dict:
            obs_dict[aid] = self._build_agent_obs(
                aid, next_supply, next_demand
            )
            terminateds[aid] = False
            truncateds[aid] = is_truncated
            infos[aid] = {"reward": rewards[aid]}

        return obs_dict, rewards, terminateds, truncateds, infos

    # ================================================================== #
    #  Internal helpers                                                    #
    # ================================================================== #
    def _compute_supply(self) -> np.ndarray:
        supply = np.zeros(self._num_hexes, dtype=np.float32)
        for h in self._taxi_hexes.values():
            idx = self._hex_to_idx.get(h)
            if idx is not None:
                supply[idx] += 1
        return supply

    def _current_demand(self) -> np.ndarray:
        t = self._step_count % self._demand_matrix.shape[0]
        return self._demand_matrix[t]

    def _build_agent_obs(
        self,
        agent_id: str,
        supply: np.ndarray,
        demand: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build the observation dict for a single agent.

        Feature layout
        --------------
        [0 … H)         one-hot hex ID
        [H]              demand_gap  at current hex  (demand − supply)
        [H+1 … H+7)     demand_gap  at 6 neighbours
        [H+7]           idle_count  at current hex   max(supply − demand, 0)
        [H+8 … H+14)    idle_count  at 6 neighbours
        [H+14]          normalised time-of-day
        """
        hex_id = self._taxi_hexes[agent_id]
        h_idx = self._hex_to_idx[hex_id]
        neighbors = self._masker.get_sorted_neighbors(hex_id)

        H = self._num_hexes
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # ── One-hot hex encoding ──────────────────────────────────────
        obs[h_idx] = 1.0

        # ── Demand gap (demand − supply): positive = unmet demand ─────
        obs[H] = demand[h_idx] - supply[h_idx]
        for j, nb in enumerate(neighbors[:NUM_DIRECTIONS]):
            nb_idx = self._hex_to_idx.get(nb)
            if nb_idx is not None:
                obs[H + 1 + j] = demand[nb_idx] - supply[nb_idx]

        # ── Idle vehicle count: max(0, supply − demand) ──────────────
        obs[H + 7] = max(0.0, supply[h_idx] - demand[h_idx])
        for j, nb in enumerate(neighbors[:NUM_DIRECTIONS]):
            nb_idx = self._hex_to_idx.get(nb)
            if nb_idx is not None:
                obs[H + 8 + j] = max(0.0, supply[nb_idx] - demand[nb_idx])

        # ── Time of day ──────────────────────────────────────────────
        obs[H + 14] = self._step_count / max(self.max_steps, 1)

        mask = self._masker.get_action_mask(hex_id)

        return {"obs": obs, "action_mask": mask}


# ========================================================================== #
#  3.  Action-Mask Torch Model                                                #
# ========================================================================== #
LARGE_NEGATIVE = -1e8


class ActionMaskModel(TorchModelV2, nn.Module):
    """Torch model that applies the ``action_mask`` from the observation
    dict to the logits produced by a standard FC backbone.

    Invalid actions receive logit = −∞ so the policy distribution
    assigns them zero probability.  This is the recommended RLlib
    pattern for environments with state-dependent action constraints.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        self._obs_key_space = orig_space["obs"]

        self.internal_model = FullyConnectedNetwork(
            self._obs_key_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(
        self,
        input_dict: Dict[str, Any],
        state: List[Any],
        seq_lens: Any,
    ) -> Tuple[torch.Tensor, List[Any]]:
        obs_flat = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]

        logits, state = self.internal_model(
            {"obs": obs_flat}, state, seq_lens
        )

        # Replace invalid-action logits with −∞
        inf_mask = torch.clamp(
            torch.log(action_mask), min=LARGE_NEGATIVE
        )
        return logits + inf_mask, state

    def value_function(self) -> torch.Tensor:
        return self.internal_model.value_function()


# ========================================================================== #
#  4.  PPO Configuration  (CTDE + Parameter Sharing)                          #
# ========================================================================== #
ENV_NAME = "TaxiDispatch-v0"


def build_ppo_config(
    num_taxis: int = 20,
    num_workers: int = 2,
    train_batch_size: int = 4000,
    minibatch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    num_sgd_iter: int = 10,
    clip_param: float = 0.2,
    entropy_coeff: float = 0.01,
    vf_clip_param: float = 50.0,
    env_config: dict | None = None,
) -> PPOConfig:
    """Build a fully-configured PPO ``AlgorithmConfig``.

    All N taxi agents are mapped to the **same** ``"shared_policy"``
    via ``policy_mapping_fn``, achieving full parameter sharing (CTDE).
    """
    env_config = env_config or {}
    env_config.setdefault("num_taxis", num_taxis)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=ENV_NAME,
            env_config=env_config,
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=None,
                    action_space=None,
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .framework("torch")
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=gae_lambda,
            num_sgd_iter=num_sgd_iter,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_clip_param=vf_clip_param,
            model={
                "custom_model": "action_mask_model",
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
        .env_runners(
            num_env_runners=num_workers,
        )
        .resources(
            num_gpus=int(torch.cuda.is_available()),
        )
    )
    return config


# ========================================================================== #
#  5.  Training Loop                                                          #
# ========================================================================== #
def train(
    iterations: int = 50,
    num_taxis: int = 20,
    num_workers: int = 2,
    checkpoint_freq: int = 10,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Initialise Ray, build the PPO algorithm, and run the training loop.

    Prints a per-iteration summary and saves periodic checkpoints.
    """
    # ── Register custom env + model ───────────────────────────────────
    register_env(ENV_NAME, lambda cfg: TaxiDispatchMultiAgentEnv(cfg))
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    # ── Build config ──────────────────────────────────────────────────
    ppo_config = build_ppo_config(
        num_taxis=num_taxis,
        num_workers=num_workers,
        env_config={
            "num_taxis": num_taxis,
            "max_steps": 288,
            "place": "Downtown Core, Singapore",
            "h3_resolution": 8,
            "demand_seed": 42,
            "density_alpha": 0.5,
        },
    )

    algo = ppo_config.build()

    logger.info(
        "PPO algorithm built — policy network:\n%s",
        algo.get_policy("shared_policy").model,
    )

    # ── Training iterations ───────────────────────────────────────────
    best_reward = float("-inf")
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  TRAINING  —  PPO + CTDE Parameter Sharing")
    print("=" * 72)
    print(f"  Taxis         : {num_taxis}")
    print(f"  Workers       : {num_workers}")
    print(f"  Iterations    : {iterations}")
    print(f"  Checkpoint dir: {ckpt_path.resolve()}")
    print("=" * 72 + "\n")

    for i in range(1, iterations + 1):
        result = algo.train()

        ep_reward_mean = result["env_runners"]["episode_reward_mean"]
        ep_len_mean = result["env_runners"]["episode_len_mean"]
        timesteps = result["num_env_steps_sampled_lifetime"]

        improved = ""
        if ep_reward_mean > best_reward:
            best_reward = ep_reward_mean
            improved = " ★"

        print(
            f"  [{i:>4}/{iterations}]  "
            f"reward={ep_reward_mean:+10.2f}  "
            f"ep_len={ep_len_mean:6.1f}  "
            f"timesteps={timesteps:>9,}{improved}"
        )

        if i % checkpoint_freq == 0 or i == iterations:
            save_result = algo.save(str(ckpt_path))
            logger.info("Checkpoint → %s", save_result)

    # ── Cleanup ───────────────────────────────────────────────────────
    algo.stop()
    print(f"\n  Training complete.  Best mean reward: {best_reward:+.2f}")
    print(f"  Checkpoints saved to: {ckpt_path.resolve()}\n")


# ========================================================================== #
#  __main__                                                                   #
# ========================================================================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO taxi dispatch policy (CTDE, parameter sharing)"
    )
    p.add_argument(
        "--iterations", type=int, default=50,
        help="Number of PPO training iterations (default: 50)",
    )
    p.add_argument(
        "--num-taxis", type=int, default=20,
        help="Number of taxi agents (default: 20)",
    )
    p.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of parallel env-runner workers (default: 2)",
    )
    p.add_argument(
        "--checkpoint-freq", type=int, default=10,
        help="Save checkpoint every N iterations (default: 10)",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 3-iteration test to verify the full pipeline",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke_test:
        args.iterations = 3
        args.num_workers = 0
        logger.info("Running smoke test (3 iters, 0 workers) …")

    ray.init(
        ignore_reinit_error=True,
        num_cpus=max(args.num_workers + 1, 2),
        logging_level=logging.WARNING,
    )

    try:
        train(
            iterations=args.iterations,
            num_taxis=args.num_taxis,
            num_workers=args.num_workers,
            checkpoint_freq=args.checkpoint_freq,
        )
    finally:
        ray.shutdown()
