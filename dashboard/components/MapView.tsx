"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Map from "react-map-gl/maplibre";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ArcLayer, ScatterplotLayer } from "@deck.gl/layers";
import "maplibre-gl/dist/maplibre-gl.css";

import type { ArcData, HexDemandState, TaxiState } from "@/lib/types";

const INITIAL_VIEW = {
  longitude: 103.8536,
  latitude: 1.2905,
  zoom: 14.5,
  pitch: 40,
  bearing: -20,
};

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// Colour ramp: dark blue → cyan → yellow → orange (dispatch intensity)
function heatColor(t: number): [number, number, number, number] {
  if (t < 0.33) {
    const s = t / 0.33;
    return [
      Math.floor(20 + 10 * s),
      Math.floor(60 + 140 * s),
      Math.floor(140 + 80 * s),
      140,
    ];
  }
  if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return [
      Math.floor(30 + 220 * s),
      Math.floor(200 + 40 * s),
      Math.floor(220 - 180 * s),
      160,
    ];
  }
  const s = (t - 0.66) / 0.34;
  return [
    Math.floor(250),
    Math.floor(240 - 140 * s),
    Math.floor(40 - 30 * s),
    180,
  ];
}

interface MapViewProps {
  taxis: TaxiState[];
  hexData: HexDemandState[];
  arcs: ArcData[];
}

export default function MapView({ taxis, hexData, arcs }: MapViewProps) {
  const [DeckGL, setDeckGL] = useState<any>(null);

  useEffect(() => {
    (async () => {
      const { luma } = await import("@luma.gl/core");
      const { webgl2Adapter } = await import("@luma.gl/webgl");
      luma.registerAdapters([webgl2Adapter]);
      const deckMod = await import("@deck.gl/react");
      setDeckGL(() => deckMod.default);
    })();
  }, []);

  const maxCount = useMemo(
    () => Math.max(1, ...hexData.map((h) => h.dispatchCount)),
    [hexData]
  );

  const now = Date.now();

  // ── Layer 1: H3 hex zones (flat, colour-coded by intensity) ──────
  const hexLayer = new H3HexagonLayer<HexDemandState>({
    id: "h3-zones",
    data: hexData,
    pickable: true,
    filled: true,
    extruded: false,
    wireframe: true,
    lineWidthMinPixels: 1,
    getHexagon: (d: HexDemandState) => d.h3Index,
    getFillColor: (d: HexDemandState) => heatColor(d.dispatchCount / maxCount),
    getLineColor: [255, 255, 255, 50],
    updateTriggers: { getFillColor: [maxCount] },
  });

  // ── Layer 2: Dispatch arcs (cyan trails fading with age) ─────────
  const arcLayer = new ArcLayer<ArcData>({
    id: "dispatch-arcs",
    data: arcs,
    pickable: false,
    getSourcePosition: (d: ArcData) => d.sourcePosition,
    getTargetPosition: (d: ArcData) => d.targetPosition,
    getSourceColor: [0, 255, 200, 220],
    getTargetColor: [0, 180, 255, 80],
    getWidth: 2.5,
    greatCircle: false,
    updateTriggers: { getSourcePosition: [arcs.length] },
  });

  // ── Layer 3: Taxi positions (large glowing dots) ─────────────────
  // Outer glow ring
  const taxiGlowLayer = new ScatterplotLayer<TaxiState>({
    id: "taxi-glow",
    data: taxis,
    pickable: false,
    filled: true,
    stroked: false,
    radiusMinPixels: 12,
    radiusMaxPixels: 24,
    getPosition: (d: TaxiState) => [d.targetLng, d.targetLat],
    getFillColor: (d: TaxiState) =>
      d.actionLabel === "STAY"
        ? [120, 120, 120, 50]
        : [0, 255, 180, 70],
    getRadius: 60,
  });

  // Inner solid dot
  const taxiLayer = new ScatterplotLayer<TaxiState>({
    id: "taxis",
    data: taxis,
    pickable: true,
    filled: true,
    stroked: true,
    radiusMinPixels: 6,
    radiusMaxPixels: 14,
    lineWidthMinPixels: 1.5,
    getPosition: (d: TaxiState) => [d.targetLng, d.targetLat],
    getFillColor: (d: TaxiState) =>
      d.actionLabel === "STAY"
        ? [160, 160, 160, 240]
        : [0, 255, 180, 255],
    getLineColor: [255, 255, 255, 180],
    getRadius: 35,
  });

  const getTooltip = useCallback(
    ({ object }: { object?: TaxiState | HexDemandState }) => {
      if (!object) return null;
      if ("vehicleId" in object) {
        return {
          text: `Vehicle ${object.vehicleId}\nHex: ${object.h3Hex.slice(-8)}\nAction: ${object.actionLabel}`,
        };
      }
      if ("h3Index" in object) {
        return {
          text: `Hex ${object.h3Index.slice(-8)}\nDispatches: ${object.dispatchCount}`,
        };
      }
      return null;
    },
    []
  );

  if (!DeckGL) {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-[#0a0a0f]">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
          <span className="text-sm text-gray-500">Initialising WebGL…</span>
        </div>
      </div>
    );
  }

  return (
    <DeckGL
      initialViewState={INITIAL_VIEW}
      controller={true}
      layers={[hexLayer, arcLayer, taxiGlowLayer, taxiLayer]}
      getTooltip={getTooltip}
      style={{ position: "absolute", inset: "0" }}
    >
      <Map reuseMaps mapStyle={MAP_STYLE} />
    </DeckGL>
  );
}
