"""
data_pipeline.py — Data Engineering & Spatial Foundation for Singapore Taxi RL

This module handles three foundational concerns:
  1. Downloading and caching the Singapore drivable road network via OSMnx.
  2. Discretising geographic space into Uber H3 hexagons.
  3. Generating synthetic Poisson-distributed ride-request demand.

Author : <your-name>
Project: Reinforcement Learning for On-Demand Taxi Dynamics (NTU Dissertation)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h3
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ========================================================================== #
#  1. Singapore Road Network                                                  #
# ========================================================================== #
class RoadNetworkLoader:
    """Downloads (or loads from cache) the drivable road network for a
    specified place via OSMnx and persists it as a GraphML file."""

    DEFAULT_PLACE = "Downtown Core, Singapore"
    DEFAULT_CACHE_DIR = Path("data")
    DEFAULT_FILENAME = "singapore_road_network.graphml"

    def __init__(
        self,
        place: str = DEFAULT_PLACE,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        filename: str = DEFAULT_FILENAME,
    ) -> None:
        self.place = place
        self.cache_dir = Path(cache_dir)
        self.filepath = self.cache_dir / filename

    # ------------------------------------------------------------------ #
    def fetch_road_network(self) -> nx.MultiDiGraph:
        """Return the OSMnx drivable graph, downloading only if the local
        cache does not already exist.

        Returns
        -------
        nx.MultiDiGraph
            The road network graph with node attributes ``x`` (lon) and
            ``y`` (lat).
        """
        if self.filepath.exists():
            logger.info("Loading cached road network from %s", self.filepath)
            graph: nx.MultiDiGraph = ox.load_graphml(self.filepath)
        else:
            logger.info(
                "Downloading drivable road network for '%s' …", self.place
            )
            graph = ox.graph_from_place(self.place, network_type="drive")

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(graph, filepath=self.filepath)
            logger.info("Saved road network to %s", self.filepath)

        logger.info(
            "Road network: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph


# ========================================================================== #
#  2. Spatial Discretisation — Uber H3                                        #
# ========================================================================== #
@dataclass
class H3Mapping:
    """Container returned by :meth:`SpatialDiscretiser.map_to_h3`."""

    active_hexes: List[str]
    node_to_h3: Dict[int, str]
    hex_to_nodes: Dict[str, List[int]]


class SpatialDiscretiser:
    """Maps OSMnx road-network nodes onto the Uber H3 hexagonal grid."""

    def __init__(self, resolution: int = 8) -> None:
        if not 0 <= resolution <= 15:
            raise ValueError("H3 resolution must be in [0, 15].")
        self.resolution = resolution

    # ------------------------------------------------------------------ #
    def map_to_h3(self, graph: nx.MultiDiGraph) -> H3Mapping:
        """Extract every node's (lat, lng) and assign it to an H3 cell.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            An OSMnx road-network graph.

        Returns
        -------
        H3Mapping
            Dataclass with ``active_hexes``, ``node_to_h3``, and
            ``hex_to_nodes``.
        """
        node_to_h3: Dict[int, str] = {}
        hex_to_nodes: Dict[str, List[int]] = {}

        for node_id, data in graph.nodes(data=True):
            lat, lng = data["y"], data["x"]
            h3_index = h3.latlng_to_cell(lat, lng, self.resolution)
            node_to_h3[node_id] = h3_index
            hex_to_nodes.setdefault(h3_index, []).append(node_id)

        active_hexes = sorted(hex_to_nodes.keys())

        logger.info(
            "H3 resolution %d → %d active hexagons from %d nodes",
            self.resolution,
            len(active_hexes),
            len(node_to_h3),
        )
        return H3Mapping(
            active_hexes=active_hexes,
            node_to_h3=node_to_h3,
            hex_to_nodes=hex_to_nodes,
        )


# ========================================================================== #
#  3. Synthetic Poisson Demand Generator                                      #
# ========================================================================== #
@dataclass
class DemandConfig:
    """Hyper-parameters for the Poisson demand generator.

    Tidal parameters
    ----------------
    The time-varying multiplier is a sum of Gaussian peaks on top of a
    low overnight baseline, producing a realistic bi-modal rush-hour
    profile over 288 five-minute steps (= 24 h)::

        m(t) = overnight_base
             + am_peak · exp(−(t − am_centre)² / 2σ²)
             + pm_peak · exp(−(t − pm_centre)² / 2σ²)
             + midday  · exp(−(t − midday_ctr)² / 2σ_mid²)
    """

    hotspot_fraction: float = 0.20
    lambda_hotspot: float = 10.0
    lambda_normal: float = 2.0
    seed: Optional[int] = 42

    overnight_base: float = 0.15
    am_peak: float = 2.8
    am_centre: float = 96.0         # step 96 ≈ 08:00
    pm_peak: float = 2.3
    pm_centre: float = 216.0        # step 216 ≈ 18:00
    midday_boost: float = 0.7
    midday_centre: float = 156.0    # step 156 ≈ 13:00
    peak_sigma: float = 22.0        # ~1.8 h half-width for rush hours
    midday_sigma: float = 36.0      # broader midday plateau


class DemandGenerator:
    """Generates a synthetic ride-request demand matrix using Poisson
    arrivals with **tidal (time-varying)** rates.

    20 % of hexagons are randomly designated as *hotspots* (e.g. CBD)
    with a higher base arrival rate.  The base λ for every hex is then
    scaled by a bi-modal tidal multiplier that peaks around 08:00 AM
    and 18:00 PM, dips overnight, and has a moderate midday plateau —
    faithfully modelling non-stationary urban taxi demand.
    """

    def __init__(self, config: DemandConfig | None = None) -> None:
        self.cfg = config or DemandConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------ #
    def _designate_hotspots(
        self, h3_hexes: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Randomly split hexagons into hotspot and normal sets."""
        n_hot = max(1, int(len(h3_hexes) * self.cfg.hotspot_fraction))
        shuffled = self._rng.permutation(h3_hexes).tolist()
        hotspots = shuffled[:n_hot]
        normals = shuffled[n_hot:]
        logger.info(
            "Hotspots: %d hexagons (λ_base=%.1f) | Normal: %d hexagons (λ_base=%.1f)",
            len(hotspots),
            self.cfg.lambda_hotspot,
            len(normals),
            self.cfg.lambda_normal,
        )
        return hotspots, normals

    # ------------------------------------------------------------------ #
    def tidal_multiplier(self, time_steps: int = 288) -> np.ndarray:
        """Compute a ``(T,)`` tidal scaling curve over one day.

        The curve combines:
        * A low overnight baseline (``overnight_base``).
        * An AM Gaussian peak centred at ``am_centre``.
        * A PM Gaussian peak centred at ``pm_centre``.
        * A gentle midday plateau centred at ``midday_centre``.

        Returns
        -------
        np.ndarray, shape ``(time_steps,)``
            Multiplier ∈ [overnight_base, ~am_peak + midday_boost + base].
        """
        c = self.cfg
        t = np.arange(time_steps, dtype=np.float64)

        def _gauss(centre: float, sigma: float) -> np.ndarray:
            return np.exp(-0.5 * ((t - centre) / sigma) ** 2)

        curve = (
            c.overnight_base
            + c.am_peak * _gauss(c.am_centre, c.peak_sigma)
            + c.pm_peak * _gauss(c.pm_centre, c.peak_sigma)
            + c.midday_boost * _gauss(c.midday_centre, c.midday_sigma)
        )
        return curve

    # ------------------------------------------------------------------ #
    def generate_poisson_demand(
        self,
        h3_hexes: List[str],
        time_steps: int = 288,
    ) -> pd.DataFrame:
        """Build a ``(time_steps × num_hexes)`` demand matrix with
        **tidal (time-varying)** Poisson rates.

        For each time step *t* and hex *h*::

            λ(t, h) = λ_base(h) × tidal_multiplier(t)
            demand(t, h) ~ Poisson(λ(t, h))

        Parameters
        ----------
        h3_hexes : list[str]
            Active H3 cell identifiers.
        time_steps : int
            Number of 5-minute intervals (default 288 = 24 hours).

        Returns
        -------
        pd.DataFrame
            Rows = time steps, columns = H3 hex IDs, values = request
            counts.
        """
        hotspots, normals = self._designate_hotspots(h3_hexes)
        hotspot_set = set(hotspots)

        base_lambdas = np.array(
            [
                self.cfg.lambda_hotspot if h in hotspot_set
                else self.cfg.lambda_normal
                for h in h3_hexes
            ],
            dtype=np.float64,
        )

        tide = self.tidal_multiplier(time_steps)  # (T,)

        # λ(t, h) = base_lambda(h) × tide(t) → shape (T, H)
        lambda_matrix = tide[:, np.newaxis] * base_lambdas[np.newaxis, :]

        demand_matrix = self._rng.poisson(lam=lambda_matrix)

        df = pd.DataFrame(demand_matrix, columns=h3_hexes)
        df.index.name = "time_step"

        total = int(demand_matrix.sum())
        peak_lambda = float(lambda_matrix.max())
        logger.info(
            "Demand matrix: %d time-steps × %d hexes → %d total requests "
            "(peak λ=%.1f at morning rush)",
            time_steps,
            len(h3_hexes),
            total,
            peak_lambda,
        )
        return df


# ========================================================================== #
#  4. Orchestrating Pipeline                                                  #
# ========================================================================== #
class DataPipeline:
    """Top-level facade that wires together the road-network loader,
    spatial discretiser, and demand generator into a single callable
    pipeline."""

    def __init__(
        self,
        place: str = RoadNetworkLoader.DEFAULT_PLACE,
        h3_resolution: int = 8,
        demand_config: DemandConfig | None = None,
        cache_dir: str | Path = RoadNetworkLoader.DEFAULT_CACHE_DIR,
    ) -> None:
        self.loader = RoadNetworkLoader(place=place, cache_dir=cache_dir)
        self.discretiser = SpatialDiscretiser(resolution=h3_resolution)
        self.demand_gen = DemandGenerator(config=demand_config)

        self.graph: Optional[nx.MultiDiGraph] = None
        self.h3_mapping: Optional[H3Mapping] = None
        self.demand: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    def run(self, time_steps: int = 288) -> "DataPipeline":
        """Execute the full pipeline end-to-end.

        Parameters
        ----------
        time_steps : int
            Number of 5-min intervals for demand generation.

        Returns
        -------
        DataPipeline
            ``self`` for method chaining / downstream inspection.
        """
        logger.info("=" * 60)
        logger.info("STEP 1 — Road Network")
        logger.info("=" * 60)
        self.graph = self.loader.fetch_road_network()

        logger.info("=" * 60)
        logger.info("STEP 2 — H3 Spatial Discretisation")
        logger.info("=" * 60)
        self.h3_mapping = self.discretiser.map_to_h3(self.graph)

        logger.info("=" * 60)
        logger.info("STEP 3 — Synthetic Poisson Demand")
        logger.info("=" * 60)
        self.demand = self.demand_gen.generate_poisson_demand(
            self.h3_mapping.active_hexes, time_steps=time_steps
        )

        return self

    # ------------------------------------------------------------------ #
    def summary(self) -> None:
        """Print a concise diagnostic summary to stdout."""
        if self.h3_mapping is None or self.demand is None:
            logger.warning("Pipeline has not been run yet — call .run() first.")
            return

        print("\n" + "=" * 60)
        print("  DATA PIPELINE — Summary")
        print("=" * 60)
        print(f"  Place              : {self.loader.place}")
        print(f"  Graph nodes        : {self.graph.number_of_nodes():,}")
        print(f"  Graph edges        : {self.graph.number_of_edges():,}")
        print(f"  H3 resolution      : {self.discretiser.resolution}")
        print(f"  Active H3 hexagons : {len(self.h3_mapping.active_hexes):,}")
        print(f"  Demand matrix shape: {self.demand.shape}")
        print(f"  Total requests     : {int(self.demand.values.sum()):,}")
        print("=" * 60)

        print("\n— First 5 active H3 hex IDs:")
        for h in self.h3_mapping.active_hexes[:5]:
            lat, lng = h3.cell_to_latlng(h)
            print(f"    {h}  →  ({lat:.5f}, {lng:.5f})")

        print("\n— Demand matrix sample (first 5 time-steps × first 5 hexes):")
        sample = self.demand.iloc[:5, :5]
        print(sample.to_string(index=True))
        print()


# ========================================================================== #
#  __main__                                                                   #
# ========================================================================== #
def plot_demand_curve(pipeline: "DataPipeline") -> Path:
    """Generate a publication-quality line chart of total system demand
    over 24 h, highlighting the two rush-hour peaks.

    Returns the path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demand_matrix = pipeline.demand.values  # (T, H)
    total_demand = demand_matrix.sum(axis=1)  # (T,)
    time_steps = len(total_demand)

    tide = pipeline.demand_gen.tidal_multiplier(time_steps)
    base_total = sum(
        pipeline.demand_gen.cfg.lambda_hotspot
        if h in set(pipeline.demand_gen._rng.permutation(
            pipeline.h3_mapping.active_hexes
        ).tolist()[:max(1, int(len(pipeline.h3_mapping.active_hexes)
                                * pipeline.demand_gen.cfg.hotspot_fraction))])
        else pipeline.demand_gen.cfg.lambda_normal
        for h in pipeline.h3_mapping.active_hexes
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    ax1.plot(
        np.arange(time_steps), total_demand,
        color="#2176AE", linewidth=0.8, alpha=0.35, label="Raw demand",
    )

    window = 12
    smoothed = pd.Series(total_demand).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    ax1.plot(
        np.arange(time_steps), smoothed,
        color="#2176AE", linewidth=2.0, label="Smoothed (1-hour MA)",
    )

    ax2 = ax1.twinx()
    ax2.plot(
        np.arange(time_steps), tide,
        color="#E8503A", linewidth=1.5, linestyle="--", alpha=0.7,
        label="Tidal multiplier",
    )
    ax2.set_ylabel("Tidal Multiplier", color="#E8503A")
    ax2.tick_params(axis="y", labelcolor="#E8503A")

    cfg = pipeline.demand_gen.cfg
    for centre, lbl in [(cfg.am_centre, "08:00\nAM Rush"),
                        (cfg.pm_centre, "18:00\nPM Rush")]:
        ax1.axvline(centre, color="#555555", linestyle=":", linewidth=0.9)
        ax1.annotate(
            lbl, xy=(centre, smoothed.iloc[int(centre)]),
            xytext=(centre + 8, smoothed.iloc[int(centre)] * 1.08),
            fontsize=9, fontweight="bold", color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        )

    hours = np.arange(0, 289, 24)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{int(h * 5 / 60):02d}:00" for h in hours], rotation=45)
    ax1.set_xlabel("Time of Day")
    ax1.set_ylabel("Total System Demand (requests / 5 min)")
    ax1.set_title("Tidal Demand Profile — Singapore Downtown Core (24 h)")
    ax1.set_xlim(0, time_steps - 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", frameon=True, fancybox=False, edgecolor="black",
    )

    out = Path("figures") / "demand_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved demand curve to %s", out)
    return out


if __name__ == "__main__":
    pipeline = DataPipeline(
        place="Downtown Core, Singapore",
        h3_resolution=8,
        demand_config=DemandConfig(
            hotspot_fraction=0.20,
            lambda_hotspot=10.0,
            lambda_normal=2.0,
            seed=42,
        ),
    )

    pipeline.run(time_steps=288)
    pipeline.summary()
    plot_demand_curve(pipeline)
