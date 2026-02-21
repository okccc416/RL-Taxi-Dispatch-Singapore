"""
kafka_producer.py — Real-Time Passenger Request Stream

Reads the synthetic Poisson demand matrix from Step 1 and streams
individual ride-request events as JSON payloads to the Kafka topic
``taxi_gps_stream``.  Each 5-minute simulation interval is compressed
into a configurable wall-clock delay so the demo runs in seconds.

Modes
-----
* **Kafka mode** (default)  — sends to a live Kafka broker.
* **Standalone mode** (``--standalone``) — prints payloads to stdout
  and writes them to ``data/request_stream.jsonl`` for the gateway
  to consume without a Kafka broker.

Usage
-----
    python kafka_producer.py                           # Kafka mode
    python kafka_producer.py --standalone              # no broker needed
    python kafka_producer.py --time-steps 50 --delay 0.3

Author : <your-name>
Project: Reinforcement Learning for On-Demand Taxi Dynamics (NTU Dissertation)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import h3
import numpy as np

from data_pipeline import DataPipeline, DemandConfig, H3Mapping

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


# ========================================================================== #
#  1.  Request Payload Builder                                                #
# ========================================================================== #
class RequestFactory:
    """Turns a (hex_id, time_step) pair into a JSON-ready request dict."""

    def __init__(self, sim_start: datetime | None = None) -> None:
        self._sim_start = sim_start or datetime.now(tz=timezone.utc)
        self._counter = 0

    def build(self, hex_id: str, time_step: int) -> dict:
        """Create a single passenger-request payload.

        Returns
        -------
        dict
            ``request_id``, ``timestamp``, ``h3_hex``, ``lat``, ``lng``,
            ``time_step``.
        """
        self._counter += 1
        lat, lng = h3.cell_to_latlng(hex_id)
        ts = self._sim_start + timedelta(minutes=time_step * 5)
        return {
            "request_id": f"REQ-{self._counter:06d}",
            "timestamp": ts.isoformat(),
            "h3_hex": hex_id,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "time_step": time_step,
        }


# ========================================================================== #
#  2.  Kafka Transport (best-effort)                                          #
# ========================================================================== #
class KafkaTransport:
    """Thin wrapper around ``kafka.KafkaProducer``.

    Falls back gracefully if the broker is unreachable.
    """

    def __init__(
        self, bootstrap_servers: str = "localhost:9092", topic: str = KAFKA_TOPIC
    ) -> None:
        self.topic = topic
        self._producer = None
        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=3000,
                max_block_ms=3000,
            )
            logger.info("Kafka producer connected → %s", bootstrap_servers)
        except Exception as exc:
            logger.warning("Kafka unavailable (%s) — messages will not be sent", exc)

    @property
    def available(self) -> bool:
        return self._producer is not None

    def send(self, payload: dict) -> None:
        if self._producer is not None:
            self._producer.send(self.topic, payload)

    def flush(self) -> None:
        if self._producer is not None:
            self._producer.flush()

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close()


# ========================================================================== #
#  3.  Stream Producer                                                        #
# ========================================================================== #
class RequestStreamProducer:
    """Reads the demand matrix row-by-row and emits individual requests.

    Parameters
    ----------
    pipeline : DataPipeline
        A *run* pipeline (graph, h3_mapping, demand populated).
    kafka_bootstrap : str
        Kafka broker address.
    standalone : bool
        If ``True``, skip Kafka and write to a local JSONL file.
    """

    def __init__(
        self,
        pipeline: DataPipeline,
        kafka_bootstrap: str = "localhost:9092",
        standalone: bool = False,
    ) -> None:
        assert pipeline.h3_mapping is not None and pipeline.demand is not None

        self._active_hexes: List[str] = pipeline.h3_mapping.active_hexes
        self._demand_matrix: np.ndarray = pipeline.demand.values
        self._factory = RequestFactory()

        self._standalone = standalone
        self._kafka: Optional[KafkaTransport] = None
        if not standalone:
            self._kafka = KafkaTransport(kafka_bootstrap)

        self._file_handle = None

    # ------------------------------------------------------------------ #
    async def stream(
        self, max_steps: int | None = None, delay: float = 0.5
    ) -> AsyncIterator[dict]:
        """Async generator yielding one request payload at a time.

        Parameters
        ----------
        max_steps : int, optional
            Cap the number of 5-minute steps to emit (default: all 288).
        delay : float
            Wall-clock seconds between simulated 5-min intervals.
        """
        T = self._demand_matrix.shape[0]
        steps = min(max_steps or T, T)

        for t in range(steps):
            demand_row = self._demand_matrix[t]
            step_count = 0

            for hex_idx, hex_id in enumerate(self._active_hexes):
                n_requests = int(demand_row[hex_idx])
                for _ in range(n_requests):
                    payload = self._factory.build(hex_id, t)
                    yield payload
                    step_count += 1

            logger.info(
                "t=%03d/%03d  emitted %d requests", t, steps, step_count
            )
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------ #
    async def run(
        self, max_steps: int | None = None, delay: float = 0.5
    ) -> None:
        """Consume the stream and route payloads to Kafka / file / stdout."""
        STREAM_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = open(STREAM_FILE, "w", encoding="utf-8") if self._standalone else None

        total = 0
        try:
            async for payload in self.stream(max_steps=max_steps, delay=delay):
                # Route to Kafka
                if self._kafka is not None and self._kafka.available:
                    self._kafka.send(payload)

                # Route to file (standalone mode)
                if fh is not None:
                    fh.write(json.dumps(payload) + "\n")

                total += 1

                if total % 200 == 0:
                    logger.info(
                        "  → %d requests streamed so far "
                        "(latest: %s @ %s)",
                        total,
                        payload["request_id"],
                        payload["h3_hex"],
                    )
        finally:
            if self._kafka is not None:
                self._kafka.flush()
                self._kafka.close()
            if fh is not None:
                fh.close()
                logger.info("JSONL stream written → %s", STREAM_FILE)

        print(f"\n  Streaming complete: {total:,} requests emitted.\n")


# ========================================================================== #
#  __main__                                                                   #
# ========================================================================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream synthetic taxi requests")
    p.add_argument(
        "--standalone",
        action="store_true",
        help="Skip Kafka; write to local JSONL file instead",
    )
    p.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka broker address (default: localhost:9092)",
    )
    p.add_argument(
        "--time-steps",
        type=int,
        default=None,
        help="Number of 5-min steps to stream (default: all 288)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between simulated 5-min intervals (default: 0.5)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    print("\n" + "=" * 64)
    print("  KAFKA PRODUCER — Passenger Request Stream")
    print("=" * 64)

    pipeline = DataPipeline(
        place="Downtown Core, Singapore",
        h3_resolution=8,
        demand_config=DemandConfig(seed=42),
    ).run(time_steps=288)

    mode = "STANDALONE (JSONL)" if args.standalone else f"KAFKA → {args.kafka_bootstrap}"
    print(f"  Mode       : {mode}")
    print(f"  Topic      : {KAFKA_TOPIC}")
    print(f"  Hexagons   : {len(pipeline.h3_mapping.active_hexes)}")
    print(f"  Time steps : {args.time_steps or 288}")
    print(f"  Delay      : {args.delay}s per interval")
    print("=" * 64 + "\n")

    producer = RequestStreamProducer(
        pipeline=pipeline,
        kafka_bootstrap=args.kafka_bootstrap,
        standalone=args.standalone,
    )
    await producer.run(max_steps=args.time_steps, delay=args.delay)


if __name__ == "__main__":
    asyncio.run(main())
