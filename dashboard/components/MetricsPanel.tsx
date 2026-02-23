"use client";

import type { Metrics } from "@/lib/types";

interface MetricsPanelProps {
  metrics: Metrics;
  connected: boolean;
}

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-widest text-gray-400">
        {label}
      </span>
      <span className={`text-xl font-bold tabular-nums ${accent}`}>
        {value}
      </span>
    </div>
  );
}

export default function MetricsPanel({
  metrics,
  connected,
}: MetricsPanelProps) {
  return (
    <div className="pointer-events-auto flex items-center gap-5 rounded-xl border border-white/10 bg-gray-950/80 px-5 py-3 shadow-2xl backdrop-blur-lg">
      {/* connection indicator */}
      <div className="flex items-center gap-2">
        <span
          className={`inline-block h-2.5 w-2.5 rounded-full ${
            connected
              ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]"
              : "bg-red-500 animate-pulse"
          }`}
        />
        <span className="text-xs text-gray-400">
          {connected ? "LIVE" : "OFFLINE"}
        </span>
      </div>

      <div className="h-8 w-px bg-white/10" />

      <StatCard
        label="Active Vehicles"
        value={metrics.activeVehicles.toLocaleString()}
        accent="text-sky-400"
      />

      <div className="h-8 w-px bg-white/10" />

      <StatCard
        label="Live ORR"
        value={`${(metrics.liveORR * 100).toFixed(1)}%`}
        accent="text-emerald-400"
      />

      <div className="h-8 w-px bg-white/10" />

      <StatCard
        label="Dispatches"
        value={metrics.totalDispatches.toLocaleString()}
        accent="text-amber-400"
      />

      <div className="h-8 w-px bg-white/10" />

      <StatCard
        label="Utilisation"
        value={`${(metrics.fleetUtilisation * 100).toFixed(0)}%`}
        accent="text-rose-400"
      />
    </div>
  );
}
