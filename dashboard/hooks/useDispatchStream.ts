"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cellToLatLng } from "h3-js";
import type {
  ArcData,
  DispatchCommand,
  HexDemandState,
  Metrics,
  TaxiState,
} from "@/lib/types";

const MAX_LOG = 200;
const MAX_ARCS = 40;
const ARC_TTL_MS = 8_000;
const RECONNECT_MS = 3_000;

export function useDispatchStream(url: string = "ws://localhost:8765") {
  const [taxis, setTaxis] = useState<Map<number, TaxiState>>(new Map());
  const [hexActivity, setHexActivity] = useState<Map<string, number>>(
    new Map()
  );
  const [arcs, setArcs] = useState<ArcData[]>([]);
  const [log, setLog] = useState<DispatchCommand[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    activeVehicles: 0,
    liveORR: 0,
    totalDispatches: 0,
  });
  const [connected, setConnected] = useState(false);

  const dispatchCount = useRef(0);
  const fulfilledCount = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);

  const processCommand = useCallback((cmd: DispatchCommand) => {
    const [oLat, oLng] = cellToLatLng(cmd.origin_h3);
    const [tLat, tLng] = cellToLatLng(cmd.target_h3);
    const now = Date.now();

    dispatchCount.current += 1;
    if (cmd.action !== 0) fulfilledCount.current += 1;

    // Update taxi position
    setTaxis((prev) => {
      const next = new Map(prev);
      next.set(cmd.vehicle_id, {
        vehicleId: cmd.vehicle_id,
        h3Hex: cmd.target_h3,
        lat: oLat,
        lng: oLng,
        targetLat: tLat,
        targetLng: tLng,
        actionLabel: cmd.action_label,
        lastUpdate: now,
      });
      return next;
    });

    // Accumulate hex activity
    setHexActivity((prev) => {
      const next = new Map(prev);
      next.set(cmd.target_h3, (next.get(cmd.target_h3) ?? 0) + 1);
      return next;
    });

    // Add dispatch arc (only for MOVE actions)
    if (cmd.action !== 0) {
      setArcs((prev) => {
        const fresh = prev.filter((a) => now - a.timestamp < ARC_TTL_MS);
        return [
          ...fresh,
          {
            sourcePosition: [oLng, oLat] as [number, number],
            targetPosition: [tLng, tLat] as [number, number],
            vehicleId: cmd.vehicle_id,
            actionLabel: cmd.action_label,
            timestamp: now,
          },
        ].slice(-MAX_ARCS);
      });
    }

    // Log
    setLog((prev) => [cmd, ...prev].slice(0, MAX_LOG));

    // Metrics
    const total = dispatchCount.current;
    setMetrics({
      activeVehicles: 0,
      liveORR: total > 0 ? fulfilledCount.current / total : 0,
      totalDispatches: total,
    });
  }, []);

  // Periodically prune stale arcs
  useEffect(() => {
    const timer = setInterval(() => {
      setArcs((prev) => prev.filter((a) => Date.now() - a.timestamp < ARC_TTL_MS));
    }, 2_000);
    return () => clearInterval(timer);
  }, []);

  // WebSocket connection with auto-reconnect
  useEffect(() => {
    let alive = true;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      if (!alive) return;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (event) => {
        try {
          const cmd: DispatchCommand = JSON.parse(event.data);
          processCommand(cmd);
        } catch {
          /* skip malformed */
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (alive) reconnectTimer = setTimeout(connect, RECONNECT_MS);
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      alive = false;
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [url, processCommand]);

  const enrichedMetrics: Metrics = {
    ...metrics,
    activeVehicles: taxis.size,
  };

  const hexData: HexDemandState[] = Array.from(hexActivity.entries()).map(
    ([h3Index, dispatchCount]) => ({ h3Index, dispatchCount })
  );

  return {
    taxis: Array.from(taxis.values()),
    hexData,
    arcs,
    log,
    metrics: enrichedMetrics,
    connected,
  };
}
