"""
cityflow_env.py — CityFlow Integration & Gymnasium Environment

This module provides:
  1. RoadNetConverter  – converts an OSMnx graph into CityFlow's roadnet.json.
  2. ActionMasker      – H3-aware action-validity masks that prevent agents
                         from stepping off the drivable grid.
  3. H3CityFlowEnv    – a Gymnasium-compliant multi-taxi dispatch environment
                         wired for Centralized Training / Decentralized
                         Execution (CTDE) with parameter sharing.

Author : <your-name>
Project: Reinforcement Learning for On-Demand Taxi Dynamics (NTU Dissertation)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import h3
import networkx as nx
import numpy as np
import pandas as pd
from gymnasium import spaces

from data_pipeline import DataPipeline, DemandConfig, H3Mapping

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants shared across the module
# ---------------------------------------------------------------------------
NUM_DIRECTIONS: int = 6
ACTION_STAY: int = 0
ACTION_DIM: int = 1 + NUM_DIRECTIONS          # 7: stay + 6 hex neighbours
OBS_DIM: int = 16                              # see _build_observations()


# ========================================================================== #
#  1.  OSMnx  →  CityFlow roadnet.json Converter                             #
# ========================================================================== #
class RoadNetConverter:
    """Convert an OSMnx ``MultiDiGraph`` to CityFlow's ``roadnet.json``.

    CityFlow expects UTM-projected coordinates (metres), so the graph is
    reprojected before export.  Every OSMnx node becomes a CityFlow
    *intersection* and every directed edge becomes a *road* with one or more
    lanes.

    Typical usage::

        path = RoadNetConverter.convert(graph, output_dir=Path("data"))
    """

    DEFAULT_LANE_WIDTH: float = 3.5         # metres
    DEFAULT_MAX_SPEED: float = 13.89        # m/s  ≈ 50 km/h
    DEFAULT_N_LANES: int = 1

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def convert(
        cls,
        graph: nx.MultiDiGraph,
        output_dir: Path | str = Path("data"),
        filename: str = "roadnet.json",
    ) -> Path:
        """Project the graph to UTM and write ``roadnet.json``.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            Unprojected OSMnx drivable graph (WGS-84 lon/lat).
        output_dir : Path | str
            Directory for the JSON artefacts.
        filename : str
            Name of the roadnet file.

        Returns
        -------
        Path
            Absolute path to the written ``roadnet.json``.
        """
        import osmnx as ox

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        graph_proj = ox.project_graph(graph)

        intersections = cls._build_intersections(graph_proj)
        roads = cls._build_roads(graph_proj)

        roadnet = {"intersections": intersections, "roads": roads}
        with open(output_path, "w") as fh:
            json.dump(roadnet, fh, indent=2)

        logger.info(
            "roadnet.json → %d intersections, %d roads → %s",
            len(intersections),
            len(roads),
            output_path,
        )
        return output_path

    @classmethod
    def write_empty_flow(
        cls, output_dir: Path | str = Path("data")
    ) -> Path:
        """Write an empty ``flow.json`` (vehicles are injected by RL)."""
        path = Path(output_dir) / "flow.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([], indent=2))
        logger.info("flow.json → %s", path)
        return path

    @classmethod
    def write_engine_config(
        cls,
        output_dir: Path | str = Path("data"),
        interval: float = 1.0,
        seed: int = 42,
    ) -> Path:
        """Write the CityFlow ``config.json`` that points to the roadnet and
        flow files in the same directory."""
        output_dir = Path(output_dir)
        config = {
            "interval": interval,
            "seed": seed,
            "dir": str(output_dir.resolve()).replace("\\", "/") + "/",
            "roadnetFile": "roadnet.json",
            "flowFile": "flow.json",
            "rlTrafficLight": False,
            "saveReplay": False,
            "roadnetLogFile": "",
            "replayLogFile": "",
        }
        path = output_dir / "cityflow_config.json"
        path.write_text(json.dumps(config, indent=2))
        logger.info("cityflow_config.json → %s", path)
        return path

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def _build_intersections(cls, G: nx.MultiDiGraph) -> List[dict]:
        """One CityFlow intersection per OSMnx node."""
        intersections: List[dict] = []
        for nid, data in G.nodes(data=True):
            in_roads = [
                f"road_{u}_{nid}_{k}"
                for u, _, k in G.in_edges(nid, keys=True)
            ]
            out_roads = [
                f"road_{nid}_{v}_{k}"
                for _, v, k in G.out_edges(nid, keys=True)
            ]
            all_roads = in_roads + out_roads

            road_links: List[dict] = []
            for sr in in_roads:
                for er in out_roads:
                    road_links.append(
                        {
                            "type": "go_straight",
                            "startRoad": sr,
                            "endRoad": er,
                            "direction": 0,
                            "laneLinks": [
                                {
                                    "startLaneIndex": 0,
                                    "endLaneIndex": 0,
                                    "points": [],
                                }
                            ],
                        }
                    )

            has_signal = G.degree(nid) > 2
            intersections.append(
                {
                    "id": f"node_{nid}",
                    "point": {"x": float(data["x"]), "y": float(data["y"])},
                    "width": 10.0,
                    "roads": all_roads,
                    "roadLinks": road_links,
                    "trafficLight": {
                        "roadLinkIndices": list(range(len(road_links))),
                        "lightphases": [
                            {
                                "time": 30,
                                "availableRoadLinks": list(
                                    range(len(road_links))
                                ),
                            }
                        ],
                    },
                    "virtual": not has_signal,
                }
            )
        return intersections

    @classmethod
    def _build_roads(cls, G: nx.MultiDiGraph) -> List[dict]:
        """One CityFlow road per OSMnx directed edge."""
        roads: List[dict] = []
        for u, v, k, data in G.edges(data=True, keys=True):
            u_pt, v_pt = G.nodes[u], G.nodes[v]
            n_lanes = cls._parse_lanes(data)
            max_speed = cls._parse_max_speed(data)

            roads.append(
                {
                    "id": f"road_{u}_{v}_{k}",
                    "startIntersection": f"node_{u}",
                    "endIntersection": f"node_{v}",
                    "points": [
                        {"x": float(u_pt["x"]), "y": float(u_pt["y"])},
                        {"x": float(v_pt["x"]), "y": float(v_pt["y"])},
                    ],
                    "lanes": [
                        {"width": cls.DEFAULT_LANE_WIDTH, "maxSpeed": max_speed}
                    ]
                    * n_lanes,
                    "nLane": n_lanes,
                }
            )
        return roads

    # ---- Edge-attribute parsers (OSMnx stores strings / lists) -------- #
    @classmethod
    def _parse_lanes(cls, edge_data: dict) -> int:
        raw = edge_data.get("lanes", cls.DEFAULT_N_LANES)
        if isinstance(raw, list):
            raw = raw[0]
        try:
            return max(1, int(str(raw).split("|")[0]))
        except (ValueError, IndexError):
            return cls.DEFAULT_N_LANES

    @classmethod
    def _parse_max_speed(cls, edge_data: dict) -> float:
        """Return speed in m/s.  OSMnx stores ``maxspeed`` as km/h strings."""
        raw = edge_data.get("maxspeed")
        if raw is None:
            return cls.DEFAULT_MAX_SPEED
        if isinstance(raw, list):
            raw = raw[0]
        try:
            return float(str(raw).split("|")[0]) / 3.6
        except (ValueError, IndexError):
            return cls.DEFAULT_MAX_SPEED


# ========================================================================== #
#  2.  H3 Action Masker                                                       #
# ========================================================================== #
class ActionMasker:
    """Computes and caches binary action-validity masks on the H3 grid.

    Action encoding (deterministic, cached per hex)::

        0       → STAY  (always valid)
        1 … 6  → move to the k-th *sorted* neighbour hex

    If a neighbour hex is **not** in the ``active_hexes`` set (i.e. it has
    no drivable road-network nodes), the corresponding action is masked
    out.  The ``apply`` method silently forces any invalid action back to
    STAY, which is the *safety filter* that prevents CityFlow crashes.
    """

    def __init__(self, active_hexes: List[str]) -> None:
        self._active_set: Set[str] = set(active_hexes)
        self._neighbor_cache: Dict[str, List[str]] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def get_sorted_neighbors(self, hex_id: str) -> List[str]:
        """Return the ≤ 6 immediate ring-1 neighbours, sorted for
        deterministic action indexing."""
        if hex_id not in self._neighbor_cache:
            ring = h3.grid_ring(hex_id, 1)
            self._neighbor_cache[hex_id] = sorted(ring)
        return self._neighbor_cache[hex_id]

    # ------------------------------------------------------------------ #
    def get_action_mask(self, hex_id: str) -> np.ndarray:
        """Return a float32 array of shape ``(7,)`` where ``1.0`` = valid.

        Masks are cached after the first call for each hex.
        """
        if hex_id not in self._mask_cache:
            neighbors = self.get_sorted_neighbors(hex_id)
            mask = np.zeros(ACTION_DIM, dtype=np.float32)
            mask[ACTION_STAY] = 1.0
            for i, nb in enumerate(neighbors):
                if nb in self._active_set:
                    mask[i + 1] = 1.0
            self._mask_cache[hex_id] = mask
        return self._mask_cache[hex_id].copy()

    # ------------------------------------------------------------------ #
    def apply(self, hex_id: str, action: int) -> int:
        """Return *action* unchanged if valid; otherwise fall back to STAY.

        This is the **safety filter** invoked inside ``env.step()`` to
        guarantee that no agent attempts an illegal grid transition.
        """
        mask = self.get_action_mask(hex_id)
        if action < 0 or action >= ACTION_DIM or mask[action] == 0.0:
            return ACTION_STAY
        return action

    # ------------------------------------------------------------------ #
    def resolve(self, hex_id: str, action: int) -> str:
        """Apply the safety filter, then return the *target* hex id."""
        safe = self.apply(hex_id, action)
        if safe == ACTION_STAY:
            return hex_id
        neighbors = self.get_sorted_neighbors(hex_id)
        idx = safe - 1
        if idx < len(neighbors):
            return neighbors[idx]
        return hex_id


# ========================================================================== #
#  3.  Gymnasium Environment — H3CityFlowEnv                                  #
# ========================================================================== #
@dataclass
class EnvConfig:
    """Tuneable hyper-parameters for :class:`H3CityFlowEnv`."""

    num_taxis: int = 10
    max_steps: int = 288                    # 24 h × 12 steps/h
    reward_pickup: float = 1.0              # reward per fulfilled request
    penalty_idle: float = -0.05             # cost per idle step
    penalty_rebalance: float = -0.02        # cost for moving empty
    seed: int = 42


class H3CityFlowEnv(gym.Env):
    """Multi-taxi dispatch over an H3-discretised Singapore road network.

    Observation space (``Dict``)
    ----------------------------
    ``"observations"``
        ``Box(num_taxis, 16)`` — per-agent feature vector
        (normalised to [0, 1]):

        ======  =========================================
        Index   Feature
        ======  =========================================
        0       hex index (normalised)
        1       demand at current hex
        2–7     demand at 6 neighbours (0 if inactive)
        8       supply at current hex
        9–14    supply at 6 neighbours (0 if inactive)
        15      time of day
        ======  =========================================

    ``"action_masks"``
        ``Box(num_taxis, 7)`` — binary mask; ``1.0`` = valid action.

    Action space
    ------------
    ``MultiDiscrete([7] × num_taxis)`` — per-taxi choice:

    * **0** = STAY
    * **1–6** = move to sorted H3 neighbour *k*

    Any action pointing to a hex **outside** the active drivable set is
    intercepted by the :class:`ActionMasker` safety filter and silently
    replaced with STAY.

    Reward
    ------
    ``R = Σ_i  (pickup_reward_i  +  idle_penalty  +  move_penalty_i)``

    The env is compatible with RLlib's ``MultiAgentEnv`` wrapper for CTDE
    (planned for Step 3).

    Parameters
    ----------
    h3_mapping : H3Mapping
        Active hex list and node↔hex lookups from Step 1.
    demand_df : pd.DataFrame
        ``(T, H)`` Poisson demand matrix from Step 1.
    graph : nx.MultiDiGraph
        The OSMnx road network (used for graph-based travel fallback).
    config : EnvConfig, optional
        Tuneable env hyper-parameters.
    use_cityflow : bool
        If ``True``, attempt to initialise the CityFlow engine.
    cityflow_config_path : Path | str, optional
        Path to ``cityflow_config.json``  (required when *use_cityflow*).
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    #  __init__                                                            #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        h3_mapping: H3Mapping,
        demand_df: pd.DataFrame,
        graph: nx.MultiDiGraph,
        config: EnvConfig | None = None,
        use_cityflow: bool = False,
        cityflow_config_path: str | Path | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.cfg = config or EnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # --- H3 spatial grid ------------------------------------------------
        self._active_hexes: List[str] = h3_mapping.active_hexes
        self._hex_to_idx: Dict[str, int] = {
            h: i for i, h in enumerate(self._active_hexes)
        }
        self._num_hexes: int = len(self._active_hexes)
        self._masker = ActionMasker(self._active_hexes)
        self._hex_to_nodes: Dict[str, List[int]] = h3_mapping.hex_to_nodes

        # --- Demand matrix (T × H) -----------------------------------------
        # Scale demand proportionally so supply/demand ratio stays consistent
        # across fleet sizes.  Baseline: 20 taxis.
        _baseline_taxis = 20
        demand_scale = max(1.0, self.cfg.num_taxis / _baseline_taxis)
        self._demand_matrix: np.ndarray = (
            demand_df.values.astype(np.float32) * demand_scale
        )

        # --- Road graph (fallback travel-time estimation) -------------------
        self._graph: nx.MultiDiGraph = graph

        # --- CityFlow engine (optional) -------------------------------------
        self._engine: Any = None
        if use_cityflow:
            self._init_cityflow(cityflow_config_path)

        # --- Gymnasium spaces -----------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            [ACTION_DIM] * self.cfg.num_taxis
        )
        self.observation_space = spaces.Dict(
            {
                "observations": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.cfg.num_taxis, OBS_DIM),
                    dtype=np.float32,
                ),
                "action_masks": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.cfg.num_taxis, ACTION_DIM),
                    dtype=np.float32,
                ),
            }
        )

        # --- Mutable state (set properly in reset()) -----------------------
        self._taxi_hexes: np.ndarray = np.empty(
            self.cfg.num_taxis, dtype=object
        )
        self._supply: np.ndarray = np.zeros(self._num_hexes, dtype=np.int32)
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------ #
    #  CityFlow bootstrap                                                  #
    # ------------------------------------------------------------------ #
    def _init_cityflow(self, config_path: str | Path | None) -> None:
        """Best-effort CityFlow initialisation with a graceful fallback."""
        try:
            import cityflow  # type: ignore[import-untyped]

            if config_path is None:
                logger.warning(
                    "use_cityflow=True but no config path given — skipping."
                )
                return
            self._engine = cityflow.Engine(str(config_path), thread_num=1)
            logger.info("CityFlow engine ready (%s)", config_path)
        except ImportError:
            logger.warning(
                "cityflow package not found — using graph-based fallback. "
                "Install CityFlow for full traffic micro-simulation."
            )

    # ================================================================== #
    #  Gymnasium API                                                       #
    # ================================================================== #

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """Scatter taxis uniformly across the active hex grid and return
        the initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._cumulative_reward = 0.0

        idxs = self._rng.integers(0, self._num_hexes, size=self.cfg.num_taxis)
        for i, idx in enumerate(idxs):
            self._taxi_hexes[i] = self._active_hexes[idx]
        self._recompute_supply()

        if self._engine is not None:
            self._engine.reset()

        return self._build_obs(), self._info()

    # ------------------------------------------------------------------ #
    #  step                                                                #
    # ------------------------------------------------------------------ #
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute one dispatch round (≈ 5 real-time minutes).

        Workflow
        --------
        1. **Action masking** — invalid moves are forced back to STAY.
        2. **Taxi relocation** — each taxi moves to its target hex.
        3. **Traffic simulation** — advance CityFlow by 300 s (if loaded).
        4. **Demand matching** — fulfil requests at each hex
           proportionally to available supply.
        5. **Reward computation** — pickups, idle cost, rebalance cost.
        6. **Observation assembly** — build next ``obs`` and ``mask``.

        Parameters
        ----------
        actions : np.ndarray, shape ``(num_taxis,)``
            Per-taxi action indices in ``{0, …, 6}``.

        Returns
        -------
        obs, reward, terminated, truncated, info
            Standard Gymnasium 5-tuple.
        """
        actions = np.asarray(actions, dtype=np.int32)
        assert actions.shape == (self.cfg.num_taxis,), (
            f"Expected ({self.cfg.num_taxis},), got {actions.shape}"
        )

        # 1 ── Apply safety masks ──────────────────────────────────────────
        masked_actions = np.empty_like(actions)
        target_hexes = np.empty(self.cfg.num_taxis, dtype=object)

        num_masked = 0
        for i in range(self.cfg.num_taxis):
            cur = self._taxi_hexes[i]
            safe_act = self._masker.apply(cur, int(actions[i]))
            masked_actions[i] = safe_act
            target_hexes[i] = self._masker.resolve(cur, int(actions[i]))
            if safe_act != actions[i]:
                num_masked += 1

        # 2 ── Relocate taxis ──────────────────────────────────────────────
        moved = masked_actions != ACTION_STAY
        self._taxi_hexes[:] = target_hexes
        self._recompute_supply()

        # 3 ── Advance CityFlow (if available) ─────────────────────────────
        if self._engine is not None:
            for _ in range(300):
                self._engine.next_step()

        # 4+5 ── Demand matching & rewards ─────────────────────────────────
        rewards = self._compute_rewards(moved)

        # 6 ── Advance clock ───────────────────────────────────────────────
        self._step_count += 1
        scalar_reward = float(rewards.sum())
        self._cumulative_reward += scalar_reward

        terminated = False
        truncated = self._step_count >= self.cfg.max_steps

        obs = self._build_obs()
        info = self._info()
        info.update(
            {
                "per_taxi_reward": rewards,
                "num_masked": num_masked,
                "masked_actions": masked_actions,
            }
        )

        return obs, scalar_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  render                                                              #
    # ------------------------------------------------------------------ #
    def render(self) -> None:
        if self.render_mode != "human":
            return
        demand = self._current_demand()
        print(f"\n[Step {self._step_count:>3}/{self.cfg.max_steps}]")
        for hex_id in self._active_hexes:
            idx = self._hex_to_idx[hex_id]
            s, d = int(self._supply[idx]), int(demand[idx])
            if s > 0 or d > 0:
                print(f"  {hex_id}  supply={s:>2}  demand={d:>2}")

    # ================================================================== #
    #  Internal helpers                                                    #
    # ================================================================== #
    def _recompute_supply(self) -> None:
        """Rebuild the per-hex taxi count vector from scratch."""
        self._supply[:] = 0
        for h in self._taxi_hexes:
            idx = self._hex_to_idx.get(h)
            if idx is not None:
                self._supply[idx] += 1

    def _current_demand(self) -> np.ndarray:
        """Return the demand vector for the current time step, wrapping
        cyclically so the env can run beyond 24 h."""
        t = self._step_count % self._demand_matrix.shape[0]
        return self._demand_matrix[t]

    # ------------------------------------------------------------------ #
    def _compute_rewards(self, moved: np.ndarray) -> np.ndarray:
        """Fair-share demand matching + movement penalties.

        At each hex the available pickups are ``min(demand, supply)``
        and are split equally among co-located taxis.
        """
        demand = self._current_demand()
        rewards = np.full(
            self.cfg.num_taxis, self.cfg.penalty_idle, dtype=np.float64
        )

        # Apply rebalance cost for taxis that moved without passengers
        for i in range(self.cfg.num_taxis):
            if moved[i]:
                rewards[i] += self.cfg.penalty_rebalance

        # Demand fulfilment (fair-share among co-located taxis)
        hex_taxi_map: Dict[str, List[int]] = {}
        for i, h in enumerate(self._taxi_hexes):
            hex_taxi_map.setdefault(h, []).append(i)

        for hex_id, taxi_ids in hex_taxi_map.items():
            idx = self._hex_to_idx.get(hex_id)
            if idx is None:
                continue
            local_demand = demand[idx]
            if local_demand <= 0:
                continue
            fulfilled = min(local_demand, len(taxi_ids))
            per_taxi = fulfilled / len(taxi_ids)
            for tid in taxi_ids:
                rewards[tid] += self.cfg.reward_pickup * per_taxi

        return rewards

    # ------------------------------------------------------------------ #
    def _build_obs(self) -> Dict[str, np.ndarray]:
        """Assemble the ``(num_taxis, OBS_DIM)`` observation matrix and
        the ``(num_taxis, ACTION_DIM)`` action-mask matrix.

        Feature layout per taxi
        -----------------------
        ======  =========================================
        Index   Feature
        ======  =========================================
        0       hex index  (normalised)
        1       demand at current hex
        2–7     demand at sorted neighbours 1–6
        8       supply at current hex
        9–14    supply at sorted neighbours 1–6
        15      time of day  (normalised)
        ======  =========================================
        """
        demand = self._current_demand()
        max_demand = max(float(demand.max()), 1.0)
        max_supply = max(int(self._supply.max()), 1)
        time_feat = self._step_count / max(self.cfg.max_steps, 1)

        obs = np.zeros(
            (self.cfg.num_taxis, OBS_DIM), dtype=np.float32
        )
        masks = np.zeros(
            (self.cfg.num_taxis, ACTION_DIM), dtype=np.float32
        )

        norm_hexes = max(self._num_hexes - 1, 1)

        for i in range(self.cfg.num_taxis):
            hex_id = self._taxi_hexes[i]
            h_idx = self._hex_to_idx.get(hex_id, 0)
            neighbors = self._masker.get_sorted_neighbors(hex_id)

            obs[i, 0] = h_idx / norm_hexes
            obs[i, 1] = demand[h_idx] / max_demand
            obs[i, 8] = self._supply[h_idx] / max_supply
            obs[i, 15] = time_feat

            for j, nb in enumerate(neighbors[:NUM_DIRECTIONS]):
                nb_idx = self._hex_to_idx.get(nb)
                if nb_idx is not None:
                    obs[i, 2 + j] = demand[nb_idx] / max_demand
                    obs[i, 9 + j] = self._supply[nb_idx] / max_supply

            masks[i] = self._masker.get_action_mask(hex_id)

        return {"observations": obs, "action_masks": masks}

    # ------------------------------------------------------------------ #
    def _info(self) -> dict:
        return {
            "step": self._step_count,
            "cumulative_reward": self._cumulative_reward,
        }


# ========================================================================== #
#  __main__ — end-to-end smoke test                                           #
# ========================================================================== #
if __name__ == "__main__":
    import textwrap

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ── Step 1 artefacts ──────────────────────────────────────────────────
    pipeline = DataPipeline(
        place="Downtown Core, Singapore",
        h3_resolution=8,
        demand_config=DemandConfig(seed=42),
    ).run(time_steps=288)

    assert pipeline.graph is not None
    assert pipeline.h3_mapping is not None
    assert pipeline.demand is not None

    # ── Convert OSMnx → CityFlow roadnet.json ────────────────────────────
    roadnet_path = RoadNetConverter.convert(pipeline.graph)
    RoadNetConverter.write_empty_flow()
    config_path = RoadNetConverter.write_engine_config()
    logger.info("CityFlow artefacts written to data/")

    # ── Inspect action masks ─────────────────────────────────────────────
    masker = ActionMasker(pipeline.h3_mapping.active_hexes)
    print("\n" + "=" * 64)
    print("  ACTION MASKS  (active hexes only)")
    print("=" * 64)
    for hex_id in pipeline.h3_mapping.active_hexes[:5]:
        mask = masker.get_action_mask(hex_id)
        neighbors = masker.get_sorted_neighbors(hex_id)
        valid_dirs = int(mask[1:].sum())
        print(f"  {hex_id}  mask={mask.astype(int).tolist()}"
              f"  ({valid_dirs}/6 neighbours active)")

    # ── Gymnasium env smoke test ─────────────────────────────────────────
    env = H3CityFlowEnv(
        h3_mapping=pipeline.h3_mapping,
        demand_df=pipeline.demand,
        graph=pipeline.graph,
        config=EnvConfig(num_taxis=10, max_steps=288, seed=42),
        use_cityflow=False,
        render_mode="human",
    )

    obs, info = env.reset(seed=42)
    print("\n" + "=" * 64)
    print("  GYMNASIUM ENV — Smoke Test (10 taxis, 5 steps)")
    print("=" * 64)
    print(f"  Observation shapes : obs={obs['observations'].shape}, "
          f"mask={obs['action_masks'].shape}")
    print(f"  Action space       : {env.action_space}")

    total_reward = 0.0
    for t in range(5):
        raw_actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(raw_actions)
        total_reward += reward
        masked_count = info["num_masked"]
        env.render()
        print(
            f"  → reward={reward:+.3f}  masked={masked_count}  "
            f"cumulative={info['cumulative_reward']:+.3f}"
        )

    print(f"\n  Total reward over 5 steps: {total_reward:+.3f}")

    # ── Demonstrate the safety filter ────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SAFETY FILTER DEMO")
    print("=" * 64)
    demo_hex = pipeline.h3_mapping.active_hexes[0]
    mask = masker.get_action_mask(demo_hex)
    print(f"  Hex       : {demo_hex}")
    print(f"  Mask      : {mask.astype(int).tolist()}")
    for act in range(ACTION_DIM):
        safe = masker.apply(demo_hex, act)
        target = masker.resolve(demo_hex, act)
        tag = "ALLOWED" if mask[act] else "BLOCKED → STAY"
        print(f"    action {act} → safe={safe}  target={target}  [{tag}]")

    print()
