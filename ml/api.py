"""
Flask API Server — AquaQuality Water Quality Anomaly Detection
==============================================================
Serves Isolation Forest predictions to the React frontend.

Key design:
  - /api/full-dataset  returns ALL ~890 historical+live readings (no augmentation labels)
  - /api/predict       runs live inference on the latest ThingSpeak readings
  - /api/train         retrains the model on fresh live + historical data
  - /api/health        health check
  - /api/model-info    current model metadata + dataset statistics

Temperature readings (26–30 °C) are simulated per-reading since the
physical sensor is not yet connected to ThingSpeak.
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

from pipeline import (
    fetch_thingspeak_data,
    build_historical_dataset,
    extract_features,
    train_isolation_forest,
    detect_anomalies,
    save_results,
    generate_charts,
    HISTORICAL_TARGET,
    MODEL_DIR,
    RESULTS_DIR,
)

app = Flask(__name__)
CORS(app)

# ── In-memory model cache ──────────────────────────────────────
_model         = None
_scaler        = None
_feature_names = None
_last_results  = None   # dict returned by save_results()


# ═══════════════════════════════════════════════════════════════
# MODEL MANAGEMENT HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_model() -> bool:
    """Load model artefacts from disk into the in-memory cache."""
    global _model, _scaler, _feature_names

    mpath = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    spath = os.path.join(MODEL_DIR, "scaler.pkl")
    fpath = os.path.join(MODEL_DIR, "feature_names.pkl")

    if os.path.exists(mpath) and os.path.exists(spath) and os.path.exists(fpath):
        _model         = joblib.load(mpath)
        _scaler        = joblib.load(spath)
        _feature_names = joblib.load(fpath)
        print("OK: Model loaded from disk")
        return True
    return False


def _ensure_model():
    """Ensure a trained model is available; train one from scratch if not."""
    global _model, _scaler, _feature_names, _last_results

    if _model is not None:
        return

    if _load_model():
        return

    print("INFO: No saved model — training from scratch …")
    _train_and_cache()


def _train_and_cache():
    """Full retrain pipeline and cache artefacts."""
    global _model, _scaler, _feature_names, _last_results

    live_df  = fetch_thingspeak_data(results=8000)
    full_df  = build_historical_dataset(live_df, history_count=HISTORICAL_TARGET)
    features, feature_names = extract_features(full_df)
    model, scaler, t_time   = train_isolation_forest(features, feature_names)
    results                 = detect_anomalies(model, scaler, features, feature_names, full_df)
    chart_paths             = generate_charts(results, features, feature_names, t_time)
    output                  = save_results(results, feature_names, t_time, chart_paths)

    _model         = model
    _scaler        = scaler
    _feature_names = feature_names
    _last_results  = output

    return output


def _row_to_dict(row: pd.Series) -> dict:
    """Serialise a DataFrame row safely."""
    return {
        "entry_id":      int(row["entry_id"]),
        "timestamp":     str(row["timestamp"]) if not isinstance(row["timestamp"], str)
                         else row["timestamp"],
        "tds_value":     round(float(row["tds_value"]),    2),
        "voltage":       round(float(row["voltage"]),      4),
        "temperature":   round(float(row["temperature"]),  2),
        "is_anomaly":    bool(row["is_anomaly"]),
        "anomaly_score": round(float(row["anomaly_score"]), 4),
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": _model is not None,
        "service":      "AquaQuality ML API",
        "method":       "Isolation Forest",
    })


# ── /api/full-dataset ──────────────────────────────────────────
@app.route("/api/full-dataset", methods=["GET"])
def full_dataset():
    """
    Return all historical + live readings with anomaly predictions.
    This is the primary source for the Dashboard and Analytics charts.
    Total rows ≈ HISTORICAL_TARGET (default 890).
    """
    try:
        _ensure_model()

        live_df  = fetch_thingspeak_data(results=8000)
        full_df  = build_historical_dataset(live_df, history_count=HISTORICAL_TARGET)
        features, _ = extract_features(full_df)
        results     = detect_anomalies(_model, _scaler, features, _feature_names, full_df)

        all_predictions = [_row_to_dict(row) for _, row in results.iterrows()]

        n_anom = sum(1 for p in all_predictions if p["is_anomaly"])
        total  = len(all_predictions)

        # Dataset stats
        tds_vals  = [p["tds_value"]  for p in all_predictions]
        temp_vals = [p["temperature"] for p in all_predictions]

        return jsonify({
            "success":        True,
            "model":          "Isolation Forest",
            "total_readings": total,
            "anomalies_found": n_anom,
            "anomaly_rate":   round(n_anom / total * 100, 2) if total else 0,
            "dataset_stats": {
                "tds_mean":  round(float(np.mean(tds_vals)),  2),
                "tds_std":   round(float(np.std(tds_vals)),   2),
                "tds_min":   round(float(np.min(tds_vals)),   2),
                "tds_max":   round(float(np.max(tds_vals)),   2),
                "temp_mean": round(float(np.mean(temp_vals)), 2),
                "temp_min":  round(float(np.min(temp_vals)),  2),
                "temp_max":  round(float(np.max(temp_vals)),  2),
            },
            "predictions": all_predictions,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/predict ───────────────────────────────────────────────
@app.route("/api/predict", methods=["GET"])
def predict():
    """
    Run anomaly detection on the latest 100 live readings from ThingSpeak.
    Used by the Recent Data page and the live KPI cards.
    """
    try:
        _ensure_model()

        live_df     = fetch_thingspeak_data(results=100)
        features, _ = extract_features(live_df)
        results     = detect_anomalies(_model, _scaler, features, _feature_names, live_df)

        preds   = [_row_to_dict(row) for _, row in results.iterrows()]
        n_anom  = sum(1 for p in preds if p["is_anomaly"])
        total   = len(preds)

        return jsonify({
            "success":         True,
            "model":           "Isolation Forest",
            "total_readings":  total,
            "anomalies_found": n_anom,
            "anomaly_rate":    round(n_anom / total * 100, 2) if total else 0,
            "predictions":     preds,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/model-info ────────────────────────────────────────────
@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return saved model metadata + dataset statistics."""
    try:
        results_path = os.path.join(RESULTS_DIR, "predictions.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            return jsonify({"success": True, **data})

        _ensure_model()
        if _last_results:
            return jsonify({"success": True, **_last_results})

        return jsonify({
            "success": True,
            "model":   {"name": "Isolation Forest",
                        "status": "loaded" if _model else "not_trained"},
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/train ─────────────────────────────────────────────────
@app.route("/api/train", methods=["GET"])
def train():
    """Retrain the model on the latest data and return updated stats."""
    try:
        output = _train_and_cache()
        return jsonify({
            "success": True,
            "message": "Model retrained successfully on latest ThingSpeak data",
            **output,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  AquaQuality ML API Server")
    print("   Isolation Forest | TDS + Temperature | Live ThingSpeak")
    print("   Endpoints:")
    print("     GET /api/health       — Liveness check")
    print("     GET /api/predict      — Live anomaly predictions (100 readings)")
    print("     GET /api/full-dataset — Full ~890-entry historical+live dataset")
    print("     GET /api/model-info   — Model metadata & statistics")
    print("     GET /api/train        — Retrain model on fresh data")
    print()

    if not _load_model():
        print("  No saved model found — will train on first request.\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
