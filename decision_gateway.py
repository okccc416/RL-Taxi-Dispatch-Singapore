"""
decision_gateway.py — RL Dispatch Gateway (Kafka Consumer + WebSocket)

Consumes passenger-request events from Kafka (or a local JSONL file in
standalone mode), runs RL policy inference to produce dispatch commands,
and broadcasts those commands to all connected WebSocket clients in
real time.

Data-flow
---------
::

    kafka_producer.py
          │
          ▼
    ┌─────────────┐      ┌──────────────┐      ┌───────────────┐
    │  Kafka Topic │─────▶│ RL Inference  │─────▶│  WebSocket    │
    │  (or JSONL)  │      │  (PPO Policy) │      │  Broadcast    │
    └─────────────┘      └──────────────┘      └───────────────┘
                                                       │
                                                       ▼
                                                  Dashboard /
                                                  Mobile App

Modes
-----
* **Kafka mode** (default) — consumes from the ``taxi_gps_stream`` topic.
* **Standalone mode** (``--standalone``) — reads ``data/request_stream.jsonl``
  produced by ``kafka_producer.py --standalone``, or generates demand
  internally if the file is missing.

A lightweight **built-in WebSocket test client** is included and
activated with ``--with-test-client``.

Usage
-----
    python decision_gateway.py --standalone --with-test-client

Author : <your-name>
Project: Reinforcement Learning for On-Demand Taxi Dynamics (NTU Dissertation)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import h3
import numpy as np
import websockets

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAFKA_TOPIC = "taxi_gps_stream"
STREAM_FILE = Path("data/request_stream.jsonl")
WS_HOST = "localhost"
WS_PORT = 8765


# ========================================================================== #
#  1.  RL Dispatch Oracle                                                     #
# ========================================================================== #
class DispatchOracle:
    """Wraps the trained RL policy (or a demand-chasing heuristic) to
    produce a dispatching command for each incoming request.

    In production this would call ``Policy.compute_single_action()``;
    here we provide two back-ends:

    * **checkpoint** — loads a real PPO checkpoint via RLlib
      (if ``checkpoint_path`` is given).
    * **heuristic** — a lightweight demand-aware rule that mimics what
      a trained policy learns: move towards the highest unmet demand
      among valid neighbours.
    """

    def __init__(
        self,
        h3_mapping: H3Mapping,
        demand_matrix: np.ndarray,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._active_hexes = h3_mapping.active_hexes
        self._hex_to_idx: Dict[str, int] = {
            h: i for i, h in enumerate(self._active_hexes)
        }
        self._num_hexes = len(self._active_hexes)
        self._masker = ActionMasker(self._active_hexes)
        self._demand_matrix = demand_matrix
        self._rng = np.random.default_rng(42)

        self._supply = np.zeros(self._num_hexes, dtype=np.float32)
        self._vehicle_id_counter = 100

        self._policy = None
        if checkpoint_path and Path(checkpoint_path).exists():
            self._try_load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------ #
    def _try_load_checkpoint(self, path: str | Path) -> None:
        """Best-effort load of a trained PPO checkpoint."""
        try:
            import warnings
            warnings.filterwarnings("ignore")
            import ray
            from ray.rllib.algorithms.ppo import PPO

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
            self._policy = PPO.from_checkpoint(str(path))
            logger.info("Loaded RL checkpoint from %s", path)
        except Exception as exc:
            logger.warning(
                "Could not load checkpoint (%s) — using heuristic", exc
            )

    # ------------------------------------------------------------------ #
    def dispatch(self, request: dict) -> dict:
        """Return a dispatch command for a single passenger request.

        Parameters
        ----------
        request : dict
            Must contain ``request_id``, ``h3_hex``, ``timestamp``.

        Returns
        -------
        dict
            Dispatch command with ``vehicle_id``, ``origin_h3``,
            ``target_h3``, ``action``, ``action_label``, etc.
        """
        origin_hex = request["h3_hex"]
        time_step = request.get("time_step", 0)

        if self._policy is not None:
            action = self._rl_inference(origin_hex, time_step)
        else:
            action = self._heuristic_inference(origin_hex, time_step)

        target_hex = self._masker.resolve(origin_hex, action)

        self._vehicle_id_counter += 1
        vehicle_id = self._vehicle_id_counter

        # Update internal supply tracker
        idx = self._hex_to_idx.get(target_hex)
        if idx is not None:
            self._supply[idx] += 1

        action_labels = ["STAY"] + [f"MOVE_DIR_{i}" for i in range(1, 7)]
        return {
            "vehicle_id": vehicle_id,
            "request_id": request["request_id"],
            "origin_h3": origin_hex,
            "target_h3": target_hex,
            "action": action,
            "action_label": action_labels[action],
            "timestamp": request.get("timestamp", datetime.now(tz=timezone.utc).isoformat()),
        }

    # ------------------------------------------------------------------ #
    def _rl_inference(self, hex_id: str, time_step: int) -> int:
        """Run the loaded RLlib policy on a single observation."""
        obs = self._build_obs(hex_id, time_step)
        mask = self._masker.get_action_mask(hex_id)
        result = self._policy.compute_single_action(
            {"obs": obs, "action_mask": mask},
            policy_id="shared_policy",
        )
        return int(result[0])

    # ------------------------------------------------------------------ #
    def _heuristic_inference(self, hex_id: str, time_step: int) -> int:
        """Demand-chasing heuristic: move to the valid neighbour with
        the highest unmet demand (demand − supply).  Ties broken randomly."""
        t = time_step % self._demand_matrix.shape[0]
        demand = self._demand_matrix[t]
        mask = self._masker.get_action_mask(hex_id)
        neighbors = self._masker.get_sorted_neighbors(hex_id)

        best_action = ACTION_STAY
        best_gap = -np.inf

        # Evaluate STAY
        h_idx = self._hex_to_idx.get(hex_id, 0)
        gap_stay = float(demand[h_idx]) - float(self._supply[h_idx])
        if mask[ACTION_STAY]:
            best_gap = gap_stay
            best_action = ACTION_STAY

        # Evaluate each direction
        for j, nb in enumerate(neighbors[:NUM_DIRECTIONS]):
            act = j + 1
            if mask[act] == 0.0:
                continue
            nb_idx = self._hex_to_idx.get(nb)
            if nb_idx is None:
                continue
            gap = float(demand[nb_idx]) - float(self._supply[nb_idx])
            if gap > best_gap or (gap == best_gap and self._rng.random() > 0.5):
                best_gap = gap
                best_action = act

        return best_action

    # ------------------------------------------------------------------ #
    def _build_obs(self, hex_id: str, time_step: int) -> np.ndarray:
        """Build the same observation vector as TaxiDispatchMultiAgentEnv."""
        t = time_step % self._demand_matrix.shape[0]
        demand = self._demand_matrix[t]
        h_idx = self._hex_to_idx[hex_id]
        neighbors = self._masker.get_sorted_neighbors(hex_id)

        H = self._num_hexes
        obs = np.zeros(H + 15, dtype=np.float32)
        obs[h_idx] = 1.0
        obs[H] = demand[h_idx] - self._supply[h_idx]
        for j, nb in enumerate(neighbors[:NUM_DIRECTIONS]):
            nb_idx = self._hex_to_idx.get(nb)
            if nb_idx is not None:
                obs[H + 1 + j] = demand[nb_idx] - self._supply[nb_idx]
        obs[H + 7] = max(0.0, self._supply[h_idx] - demand[h_idx])
        for j, nb in enumerate(neighbors[:NUM_DIRECTIONS]):
            nb_idx = self._hex_to_idx.get(nb)
            if nb_idx is not None:
                obs[H + 8 + j] = max(0.0, self._supply[nb_idx] - demand[nb_idx])
        obs[H + 14] = time_step / 288.0
        return obs


# ========================================================================== #
#  2.  Request Source (Kafka or JSONL Fallback)                               #
# ========================================================================== #
class RequestSource:
    """Abstract-ish async iterable over incoming passenger requests.

    Tries Kafka first; falls back to reading the JSONL file produced by
    ``kafka_producer.py --standalone``; finally generates demand internally
    if neither is available.
    """

    def __init__(
        self,
        standalone: bool = False,
        kafka_bootstrap: str = "localhost:9092",
        pipeline: DataPipeline | None = None,
    ) -> None:
        self._standalone = standalone
        self._kafka_bootstrap = kafka_bootstrap
        self._pipeline = pipeline

    async def __aiter__(self):
        if not self._standalone:
            async for req in self._from_kafka():
                yield req
        elif STREAM_FILE.exists():
            async for req in self._from_jsonl():
                yield req
        else:
            async for req in self._generate_internal():
                yield req

    # ------------------------------------------------------------------ #
    async def _from_kafka(self):
        """Consume from a live Kafka broker."""
        from kafka import KafkaConsumer

        logger.info("Connecting Kafka consumer → %s", self._kafka_bootstrap)
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=self._kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            consumer_timeout_ms=60_000,
        )
        logger.info("Kafka consumer ready on topic '%s'", KAFKA_TOPIC)

        loop = asyncio.get_running_loop()
        try:
            for msg in consumer:
                yield msg.value
                await asyncio.sleep(0)
        finally:
            consumer.close()

    # ------------------------------------------------------------------ #
    async def _from_jsonl(self):
        """Replay the local JSONL file in a continuous loop at a
        human-visible pace (~5 dispatches / second)."""
        logger.info("Reading request stream from %s (looping)", STREAM_FILE)
        while True:
            with open(STREAM_FILE, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
                        await asyncio.sleep(0.2)
            logger.info("JSONL exhausted — rewinding for continuous demo")
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------ #
    async def _generate_internal(self):
        """Generate demand on-the-fly in a continuous loop."""
        logger.info("No external source — generating demand internally (looping)")
        assert self._pipeline is not None
        dm = self._pipeline.demand.values
        hexes = self._pipeline.h3_mapping.active_hexes
        counter = 0

        while True:
            sim_start = datetime.now(tz=timezone.utc)
            for t in range(dm.shape[0]):
                for hex_idx, hex_id in enumerate(hexes):
                    for _ in range(int(dm[t, hex_idx])):
                        counter += 1
                        lat, lng = h3.cell_to_latlng(hex_id)
                        yield {
                            "request_id": f"REQ-{counter:06d}",
                            "timestamp": (
                                sim_start + timedelta(minutes=t * 5)
                            ).isoformat(),
                            "h3_hex": hex_id,
                            "lat": round(lat, 6),
                            "lng": round(lng, 6),
                            "time_step": t,
                        }
                        await asyncio.sleep(0.2)
            logger.info("24-h cycle complete — restarting")
            await asyncio.sleep(1.0)


# ========================================================================== #
#  3.  Decision Gateway  (Kafka Consumer + WebSocket Server)                  #
# ========================================================================== #
class DecisionGateway:
    """Orchestrates request consumption, RL inference, and WebSocket
    broadcast of dispatch commands.

    Parameters
    ----------
    oracle : DispatchOracle
        Produces dispatch decisions.
    source : RequestSource
        Async iterable of passenger requests.
    ws_host, ws_port : str, int
        WebSocket server bind address.
    """

    def __init__(
        self,
        oracle: DispatchOracle,
        source: RequestSource,
        ws_host: str = WS_HOST,
        ws_port: int = WS_PORT,
    ) -> None:
        self._oracle = oracle
        self._source = source
        self._ws_host = ws_host
        self._ws_port = ws_port

        self._ws_clients: Set[Any] = set()
        self._dispatch_count = 0
        self._start_time: float = 0.0
        self._command_log: List[dict] = []

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Launch the WebSocket server and the consumer loop
        concurrently."""
        self._start_time = time.monotonic()

        consumer_task = asyncio.create_task(self._consume_and_dispatch())

        async with websockets.serve(
            self._ws_handler, self._ws_host, self._ws_port
        ):
            logger.info(
                "WebSocket server listening on ws://%s:%d",
                self._ws_host,
                self._ws_port,
            )
            await consumer_task

        self._print_summary()

    # ------------------------------------------------------------------ #
    async def _ws_handler(self, websocket) -> None:
        """Handle a single WebSocket client connection."""
        self._ws_clients.add(websocket)
        remote = websocket.remote_address
        logger.info("WebSocket client connected: %s", remote)
        try:
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._ws_clients.discard(websocket)
            logger.info("WebSocket client disconnected: %s", remote)

    # ------------------------------------------------------------------ #
    async def _broadcast(self, command: dict) -> None:
        """Push a JSON dispatch command to every connected WS client."""
        if not self._ws_clients:
            return
        payload = json.dumps(command)
        stale: Set[Any] = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send(payload)
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedOK,
            ):
                stale.add(ws)
        self._ws_clients -= stale

    # ------------------------------------------------------------------ #
    async def _consume_and_dispatch(self) -> None:
        """Main loop: consume requests → RL inference → broadcast."""
        # Brief pause to let the WS server bind before processing
        await asyncio.sleep(0.5)

        async for request in self._source:
            command = self._oracle.dispatch(request)
            self._dispatch_count += 1
            self._command_log.append(command)

            await self._broadcast(command)

            if self._dispatch_count % 100 == 0:
                n_clients = len(self._ws_clients)
                logger.info(
                    "  dispatched %d commands  (ws_clients=%d)  "
                    "latest: vehicle_%d → %s [%s]",
                    self._dispatch_count,
                    n_clients,
                    command["vehicle_id"],
                    command["target_h3"],
                    command["action_label"],
                )

        logger.info(
            "Request source exhausted — %d commands dispatched",
            self._dispatch_count,
        )

    # ------------------------------------------------------------------ #
    def _print_summary(self) -> None:
        elapsed = time.monotonic() - self._start_time
        print("\n" + "=" * 64)
        print("  DECISION GATEWAY — Session Summary")
        print("=" * 64)
        print(f"  Total dispatches : {self._dispatch_count:,}")
        print(f"  Elapsed time     : {elapsed:.1f}s")
        if self._dispatch_count > 0:
            rate = self._dispatch_count / max(elapsed, 0.01)
            print(f"  Throughput       : {rate:,.0f} dispatches/sec")
        print("=" * 64)

        if self._command_log:
            print("\n  Last 5 dispatch commands:")
            for cmd in self._command_log[-5:]:
                print(
                    f"    vehicle_{cmd['vehicle_id']:>4}  "
                    f"{cmd['origin_h3']} → {cmd['target_h3']}  "
                    f"[{cmd['action_label']}]"
                )
        print()


# ========================================================================== #
#  4.  Built-in WebSocket Test Client                                         #
# ========================================================================== #
async def run_test_client(
    uri: str = f"ws://{WS_HOST}:{WS_PORT}",
    max_messages: int = 20,
) -> None:
    """Connect to the gateway's WS server and print received commands."""
    await asyncio.sleep(1.5)
    logger.info("Test client connecting to %s …", uri)

    try:
        async with websockets.connect(uri) as ws:
            logger.info("Test client connected")
            count = 0
            async for raw in ws:
                cmd = json.loads(raw)
                count += 1
                print(
                    f"    [WS #{count:>3}]  "
                    f"vehicle_{cmd['vehicle_id']:>4}  "
                    f"{cmd['origin_h3']} → {cmd['target_h3']}  "
                    f"[{cmd['action_label']}]"
                )
                if count >= max_messages:
                    break
            print(f"\n    Test client received {count} messages — closing.\n")
    except Exception as exc:
        logger.warning("Test client error: %s", exc)


# ========================================================================== #
#  __main__                                                                   #
# ========================================================================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL Dispatch Gateway — Kafka consumer + WebSocket server"
    )
    p.add_argument(
        "--standalone",
        action="store_true",
        help="Read from JSONL file or generate internally (no Kafka)",
    )
    p.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka broker address (default: localhost:9092)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to an RLlib PPO checkpoint directory",
    )
    p.add_argument(
        "--ws-port",
        type=int,
        default=WS_PORT,
        help=f"WebSocket server port (default: {WS_PORT})",
    )
    p.add_argument(
        "--with-test-client",
        action="store_true",
        help="Also run a built-in WS test client that prints received commands",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    print("\n" + "=" * 64)
    print("  DECISION GATEWAY — RL Dispatch Server")
    print("=" * 64)

    # ── Load data pipeline ────────────────────────────────────────────
    pipeline = DataPipeline(
        place="Downtown Core, Singapore",
        h3_resolution=8,
        demand_config=DemandConfig(seed=42),
    ).run(time_steps=288)

    mode = "STANDALONE" if args.standalone else f"KAFKA ← {args.kafka_bootstrap}"
    inference = "RL checkpoint" if args.checkpoint else "demand-chasing heuristic"
    print(f"  Source     : {mode}")
    print(f"  Inference  : {inference}")
    print(f"  WebSocket  : ws://{WS_HOST}:{args.ws_port}")
    print(f"  Hexagons   : {len(pipeline.h3_mapping.active_hexes)}")
    print("=" * 64 + "\n")

    # ── Build components ──────────────────────────────────────────────
    oracle = DispatchOracle(
        h3_mapping=pipeline.h3_mapping,
        demand_matrix=pipeline.demand.values.astype(np.float32),
        checkpoint_path=args.checkpoint,
    )

    source = RequestSource(
        standalone=args.standalone,
        kafka_bootstrap=args.kafka_bootstrap,
        pipeline=pipeline,
    )

    gateway = DecisionGateway(
        oracle=oracle,
        source=source,
        ws_host=WS_HOST,
        ws_port=args.ws_port,
    )

    # ── Launch ────────────────────────────────────────────────────────
    tasks = [gateway.start()]

    if args.with_test_client:
        tasks.append(
            run_test_client(
                uri=f"ws://{WS_HOST}:{args.ws_port}",
                max_messages=25,
            )
        )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
