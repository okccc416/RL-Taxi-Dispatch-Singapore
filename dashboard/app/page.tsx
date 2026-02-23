"use client";

import { useEffect } from "react";
import dynamic from "next/dynamic";
import { useDispatchStream } from "@/hooks/useDispatchStream";
import MetricsPanel from "@/components/MetricsPanel";
import DispatchTerminal from "@/components/DispatchTerminal";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

export default function DashboardPage() {
  const { taxis, hexData, arcs, log, metrics, connected, fleetSize, simTime } =
    useDispatchStream("ws://localhost:8765");

  useEffect(() => {
    const handler = (event: ErrorEvent) => {
      if (event.message?.includes("maxTextureDimension2D")) {
        event.preventDefault();
      }
    };
    window.addEventListener("error", handler);
    return () => window.removeEventListener("error", handler);
  }, []);

  return (
    <main className="relative h-screen w-screen overflow-hidden">
      {/* ── Full-screen map ──────────────────────────────────── */}
      <MapView taxis={taxis} hexData={hexData} arcs={arcs} />

      {/* ── Top-left: title + sim info ─────────────────────── */}
      <div className="pointer-events-none absolute left-5 top-5 z-10 flex flex-col gap-1">
        <h1 className="text-lg font-semibold tracking-tight text-white/90">
          RL Taxi Dispatch
        </h1>
        <p className="text-[11px] text-gray-500">
          Singapore Downtown Core — CTDE Multi-Agent PPO
        </p>
        <div className="mt-1 flex items-center gap-4">
          <span className="text-sm font-semibold tabular-nums text-violet-400">
            {simTime || "--:--"}
          </span>
          <span className="text-xs text-sky-300">
            {fleetSize > 0 ? `Fleet: ${fleetSize} Taxis` : ""}
          </span>
        </div>
      </div>

      {/* ── Metrics bar (top-center) ─────────────────────────── */}
      <div className="pointer-events-none absolute right-5 top-5 z-10">
        <MetricsPanel metrics={metrics} connected={connected} />
      </div>

      {/* ── Live terminal (bottom-right) ─────────────────────── */}
      <div className="pointer-events-none absolute bottom-5 right-5 z-10 h-72 w-[420px]">
        <DispatchTerminal log={log} />
      </div>

      {/* ── Legend (bottom-left) ──────────────────────────────── */}
      <div className="pointer-events-none absolute bottom-5 left-5 z-10 rounded-lg border border-white/10 bg-gray-950/70 px-4 py-3 text-xs text-gray-400 backdrop-blur-md">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="inline-block h-3 w-3 rounded-full bg-emerald-400" />
          Moving taxi
        </div>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="inline-block h-3 w-3 rounded-full bg-gray-500" />
          Idle taxi (STAY)
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-6 rounded bg-sky-600/60" />
          Dispatch activity (hex)
        </div>
      </div>

      {/* ── Bottom watermark ──────────────────────────────────── */}
      <div className="pointer-events-none absolute bottom-1.5 left-1/2 z-10 -translate-x-1/2">
        <span className="text-[10px] text-gray-700">
          MSc Dissertation — Reinforcement Learning for On-Demand Taxi Dynamics
        </span>
      </div>
    </main>
  );
}
