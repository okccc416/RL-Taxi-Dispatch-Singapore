"use client";

interface SimClockProps {
  simTime: string;
  fleetSize: number;
}

export default function SimClock({ simTime, fleetSize }: SimClockProps) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold tabular-nums tracking-tight text-white/90">
          {simTime || "--:--"}
        </span>
        <span className="text-[10px] uppercase tracking-widest text-gray-500">
          SIM
        </span>
      </div>
      {fleetSize > 0 && (
        <span className="text-xs font-medium text-sky-400/80">
          Fleet: {fleetSize.toLocaleString()} Taxis
        </span>
      )}
    </div>
  );
}
