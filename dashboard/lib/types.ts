/** JSON payload from decision_gateway.py WebSocket */
export interface DispatchCommand {
  vehicle_id: number;
  request_id: string;
  origin_h3: string;
  target_h3: string;
  action: number;
  action_label: string;
  timestamp: string;
  fleet_size?: number;
  sim_time?: string;
}

/** Internal taxi state for map rendering */
export interface TaxiState {
  vehicleId: number;
  h3Hex: string;
  lat: number;
  lng: number;
  targetLat: number;
  targetLng: number;
  actionLabel: string;
  lastUpdate: number;
}

/** Aggregated metrics */
export interface Metrics {
  activeVehicles: number;
  liveORR: number;
  totalDispatches: number;
  fleetUtilisation: number;
}

/** Hex demand state for the H3 layer */
export interface HexDemandState {
  h3Index: string;
  dispatchCount: number;
}

/** Recent dispatch arc for visualisation */
export interface ArcData {
  sourcePosition: [number, number];
  targetPosition: [number, number];
  vehicleId: number;
  actionLabel: string;
  timestamp: number;
}
