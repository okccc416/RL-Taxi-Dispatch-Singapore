"use client";

import { useEffect, useRef } from "react";
import type { DispatchCommand } from "@/lib/types";

interface DispatchTerminalProps {
  log: DispatchCommand[];
}

export default function DispatchTerminal({ log }: DispatchTerminalProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [log.length]);

  return (
    <div className="pointer-events-auto flex h-full flex-col overflow-hidden rounded-xl border border-white/10 bg-gray-950/80 shadow-2xl backdrop-blur-lg">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
        <div className="flex gap-1.5">
          <span className="h-3 w-3 rounded-full bg-red-500/80" />
          <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <span className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <span className="ml-2 text-xs font-medium tracking-wide text-gray-400">
          RL DISPATCH LOG
        </span>
      </div>

      {/* Scrolling log */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-2 font-mono text-[11px] leading-5"
      >
        {log.length === 0 && (
          <p className="text-gray-600 italic">
            Waiting for dispatch commands…
          </p>
        )}
        {log.map((cmd, i) => (
          <div key={`${cmd.vehicle_id}-${cmd.timestamp}-${i}`} className="flex gap-1">
            <span className="text-gray-600 select-none">
              {new Date(cmd.timestamp).toLocaleTimeString()}
            </span>
            <span className="text-sky-400">
              Vehicle_{cmd.vehicle_id}
            </span>
            <span className="text-gray-500">→</span>
            <span className="text-amber-300 truncate">
              {cmd.target_h3}
            </span>
            <span
              className={
                cmd.action === 0 ? "text-gray-500" : "text-emerald-400"
              }
            >
              [{cmd.action_label}]
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
