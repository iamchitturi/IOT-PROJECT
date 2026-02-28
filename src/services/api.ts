/**
 * AquaQuality — Unified API Service
 *
 * Data Flow:
 *   ESP32 (TDS Sensor) → ThingSpeak Cloud (HTTP/JSON) → Python ML API (Isolation Forest)
 *
 * Handles:
 *   1. Live sensor readings from ThingSpeak (TDS + voltage)
 *   2. Simulated temperature (26–30 °C) added by the Python backend
 *   3. Isolation Forest anomaly detection via the local Flask API
 */

const THINGSPEAK_CHANNEL  = '3286342';
const THINGSPEAK_READ_KEY = 'SP9YNCTZL6X09L94';
const PYTHON_API          = 'http://localhost:5000';

/* ═══════════════════════════════════════
   TYPE DEFINITIONS
   ═══════════════════════════════════════ */

/** A single live sensor reading fetched directly from ThingSpeak */
export interface SensorReading {
  entryId:     number;
  timestamp:   string;
  tdsValue:    number;
  voltage:     number;
  temperature: number;   // °C — simulated 26–30 °C (sensor not yet connected)
}

/** A single ML prediction record returned by the Python API */
export interface MLPrediction {
  entry_id:      number;
  timestamp:     string;
  tds_value:     number;
  voltage:       number;
  temperature:   number;
  is_anomaly:    boolean;
  anomaly_score: number;
}

/** Response from /api/model-info */
export interface ModelInfo {
  success: boolean;
  model: {
    name:                  string;
    algorithm:             string;
    n_estimators:          number;
    contamination:         number;
    training_time_seconds: number;
    features_used:         string[];
    n_features:            number;
    trained_at:            string;
  };
  dataset: {
    total_samples: number;
    tds_mean:      number;
    tds_std:       number;
    tds_min:       number;
    tds_max:       number;
    temp_mean:     number;
    temp_min:      number;
    temp_max:      number;
  };
  anomaly_detection: {
    total_anomalies: number;
    anomaly_rate:    number;
    normal_count:    number;
  };
}

/** Response from /api/predict (live 100 readings) */
export interface PredictResponse {
  success:         boolean;
  model:           string;
  total_readings:  number;
  anomalies_found: number;
  anomaly_rate:    number;
  predictions:     MLPrediction[];
}

/** Response from /api/full-dataset (~890 historical + live combined) */
export interface FullDatasetResponse {
  success:         boolean;
  model:           string;
  total_readings:  number;
  anomalies_found: number;
  anomaly_rate:    number;
  dataset_stats: {
    tds_mean:  number;
    tds_std:   number;
    tds_min:   number;
    tds_max:   number;
    temp_mean: number;
    temp_min:  number;
    temp_max:  number;
  };
  predictions: MLPrediction[];
}

export interface HealthResponse {
  status:       string;
  method:       string;
  model_loaded: boolean;
}

/* ═══════════════════════════════════════
   THINGSPEAK — LIVE SENSOR DATA
   Protocol: HTTP GET → JSON
   ESP32 uploads every 20 s via WiFi
   ═══════════════════════════════════════ */

export async function fetchLiveReadings(count = 8000): Promise<SensorReading[]> {
  const url = `https://api.thingspeak.com/channels/${THINGSPEAK_CHANNEL}/feeds.json`
            + `?api_key=${THINGSPEAK_READ_KEY}&results=${count}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ThingSpeak error: ${res.status}`);
  const data = await res.json();

  return (data.feeds || []).map((feed: any) => {
    let tempVal = parseFloat(feed.field4);
    if (isNaN(tempVal)) {
      try {
        const dt = new Date(feed.created_at);
        const hour = dt.getUTCHours();
        const min = dt.getUTCMinutes();
        const timeVal = hour + (min / 60.0);
        const diurnal = Math.cos((timeVal - 14.0) * 2.0 * Math.PI / 24.0);
        tempVal = 29.5 + diurnal * 3.5;
        // Float clamping and rounding
        tempVal = Math.min(33.0, Math.max(26.0, tempVal));
        tempVal = Math.round(tempVal * 100) / 100;
      } catch (e) {
        tempVal = 28.0;
      }
    }

    return {
      entryId:     feed.entry_id,
      timestamp:   feed.created_at,
      tdsValue:    parseFloat(feed.field1) || 0,
      voltage:     parseFloat(feed.field2) || 0,
      temperature: tempVal,
    };
  });
}

/* ═══════════════════════════════════════
   PYTHON ML API — ISOLATION FOREST
   Runs on localhost:5000
   ═══════════════════════════════════════ */

export async function checkMLHealth(): Promise<HealthResponse> {
  const res = await fetch(`${PYTHON_API}/api/health`);
  if (!res.ok) throw new Error('ML API unreachable');
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${PYTHON_API}/api/model-info`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

/** Latest 100 live readings with Isolation Forest predictions */
export async function fetchPredictions(): Promise<PredictResponse> {
  const res = await fetch(`${PYTHON_API}/api/predict`);
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

/** Full ~890-entry dataset (historical + live) with predictions */
export async function fetchFullDataset(): Promise<FullDatasetResponse> {
  const res = await fetch(`${PYTHON_API}/api/full-dataset`);
  if (!res.ok) throw new Error('Failed to fetch full dataset');
  return res.json();
}

/** Trigger a full model retrain on the latest ThingSpeak data */
export async function retrainModel(): Promise<any> {
  const res = await fetch(`${PYTHON_API}/api/train`);
  if (!res.ok) throw new Error('Failed to retrain');
  return res.json();
}

/* ═══════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════ */

export function classifyWaterQuality(tds: number): {
  label: string;
  color: string;
  description: string;
} {
  if (tds <=  50) return { label: 'Excellent', color: '#06b6d4', description: 'Pure drinking water' };
  if (tds <= 300) return { label: 'Good',      color: '#10b981', description: 'Acceptable for drinking' };
  if (tds <= 600) return { label: 'Fair',      color: '#f59e0b', description: 'Less acceptable, consider filtration' };
  if (tds <= 900) return { label: 'Poor',      color: '#f97316', description: 'Not recommended for drinking' };
  if (tds <=1200) return { label: 'Bad',       color: '#ef4444', description: 'Unacceptable, requires treatment' };
  return               { label: 'Hazardous',   color: '#dc2626', description: 'Harmful, do not consume' };
}

export function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch {
    return ts;
  }
}
