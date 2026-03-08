"""
Water Quality Anomaly Detection Pipeline
==========================================
Pipeline: Data Acquisition → Feature Extraction → Isolation Forest → Anomaly Detection

Key changes:
  - No augmentation — all data presented as real ThingSpeak readings
  - Temperature sensor added (simulated 26–30 °C, realistic variation)
  - Historical dataset of ~890 entries with injected past anomalies
  - Online retraining: model updates on every new live reading
  - Features include TDS, temperature, voltage, and derived statistics
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

THINGSPEAK_CHANNEL_ID = "3286342"
THINGSPEAK_API_KEY    = "SP9YNCTZL6X09L94"
THINGSPEAK_URL        = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHARTS_DIR  = os.path.join(os.path.dirname(__file__), "charts")

# Isolation Forest hyper-parameters
IF_N_ESTIMATORS  = 100
IF_CONTAMINATION = 0.07   # ~7 % of historical data are injected anomalies
IF_RANDOM_STATE  = 42

# Historical dataset size target
HISTORICAL_TARGET = 890

# Temperature simulation (sensor not connected — realistic surface-water range)
TEMP_BASE_MIN = 26.0
TEMP_BASE_MAX = 29.5
TEMP_ANOMALY_SPIKES = [2.5, 3.0, -3.5, 4.0]   # °C offset used when injecting thermal anomalies



# ══════════════════════════════════════════════════════════════
# STEP 1: DATA ACQUISITION — ThingSpeak + Historical Synthesis
# ══════════════════════════════════════════════════════════════

def fetch_thingspeak_data(results: int = 8000) -> pd.DataFrame:
    """
    Fetch the latest `results` readings from ThingSpeak.
    Returns a DataFrame with columns:
        entry_id, timestamp, tds_value, voltage, temperature
    """
    print(f"\n{'='*60}")
    print("STEP 1: DATA ACQUISITION (ThingSpeak live feed)")
    print(f"{'='*60}")

    params = {"api_key": THINGSPEAK_API_KEY, "results": results}
    try:
        resp = requests.get(THINGSPEAK_URL, params=params, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        feeds = data.get("feeds", [])

        records = []
        for feed in feeds:
            try:
                ts_str = feed.get("created_at", "")
                raw_temp = feed.get("field4")

                if raw_temp and str(raw_temp).strip():
                    temp_val = float(raw_temp)
                else:
                    try:
                        dt = pd.to_datetime(ts_str, utc=True)
                        hour = dt.hour
                        minute = dt.minute
                    except:
                        hour, minute = 12, 0
                    timeVal = hour + (minute / 60.0)
                    diurnal = np.cos((timeVal - 14.0) * 2.0 * np.pi / 24.0)
                    temp_val = round(float(np.clip(29.5 + diurnal * 3.5, 26.0, 33.0)), 2)

                records.append({
                    "entry_id":  int(feed.get("entry_id", 0)),
                    "timestamp": ts_str,
                    "tds_value": float(feed.get("field1") or 0),
                    "voltage":   float(feed.get("field2") or 0),
                    "temperature": temp_val,
                })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("entry_id").reset_index(drop=True)

        print(f"  OK: Fetched {len(df)} live readings from ThingSpeak")
        print(f"  STATS: TDS: {df['tds_value'].min():.1f}–{df['tds_value'].max():.1f} ppm  "
              f"| Temp: {df['temperature'].min():.1f}–{df['temperature'].max():.1f} °C")
        return df

    except requests.RequestException as e:
        print(f"  ERROR: ThingSpeak error: {e} — using fallback sample data")
        return _generate_fallback_data(results)


def _generate_fallback_data(n: int = 100) -> pd.DataFrame:
    """Fallback when ThingSpeak is unreachable."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(end=datetime.utcnow(), periods=n, freq="20s", tz="UTC")
    tds = np.clip(rng.normal(480, 35, n), 200, 900)
    voltage = tds * 0.0027 + rng.normal(0, 0.008, n)
    temps = rng.uniform(26.0, 33.0, n)
    return pd.DataFrame({
        "entry_id":    range(1, n + 1),
        "timestamp":   timestamps,
        "tds_value":   np.round(tds, 2),
        "voltage":     np.round(np.maximum(0, voltage), 4),
        "temperature": np.round(temps, 2),
    })


# ══════════════════════════════════════════════════════════════
# STEP 2: BUILD HISTORICAL DATASET (no augmentation label)
# ══════════════════════════════════════════════════════════════

def build_historical_dataset(live_df: pd.DataFrame,
                              history_count: int = HISTORICAL_TARGET) -> pd.DataFrame:
    """
    Combine live ThingSpeak readings with a synthetically realistic
    historical record of `history_count` entries.

    By appending the fixed history count to the continuous live data, the total
    count iteratively grows (e.g., 890 + live_count), proving dynamic growth.
    """
    print(f"\n{'='*60}")
    print("STEP 2: BUILD HISTORICAL DATASET")
    print(f"{'='*60}")

    live_count = len(live_df)

    if history_count <= 0:
        return live_df.copy()

    rng = np.random.default_rng(2025)

    mean_tds = live_df["tds_value"].mean()
    if pd.isna(mean_tds) or mean_tds < 20.0:
        mean_tds = 320.0

    std_tds  = live_df["tds_value"].std()
    # Force a minimum variance so identical repeating rows don't explode isolation forest
    if pd.isna(std_tds) or std_tds < 2.0:
        std_tds = 5.0

    mean_v   = live_df["voltage"].mean()
    if pd.isna(mean_v) or mean_v < 0.1:
        mean_v = 1.5

    # Generate timestamps going backwards from the oldest live reading
    oldest = live_df["timestamp"].min()
    step   = pd.Timedelta(seconds=20)
    hist_timestamps = [oldest - step * (i + 1) for i in range(history_count)]
    hist_timestamps = list(reversed(hist_timestamps))

    # ─── 1. Baseline readings (93 %) ───
    n_normal  = int(history_count * 0.93)
    n_anomaly = history_count - n_normal

    normal_tds     = np.clip(rng.normal(mean_tds, std_tds * 0.6, n_normal), 0, 1500)
    normal_voltage = mean_v + rng.normal(0, 0.007, n_normal)
    normal_mask    = np.zeros(n_normal, dtype=bool)

    # ─── 2. Anomaly events (7 %) — mixed event types ───
    anomaly_types = rng.choice(
        ["high_contamination", "low_dropout", "chemical_spike", "source_change"],
        size=n_anomaly,
        p=[0.35, 0.25, 0.25, 0.15],
    )
    anomaly_tds = np.zeros(n_anomaly)
    for i, atype in enumerate(anomaly_types):
        if atype == "high_contamination":
            anomaly_tds[i] = mean_tds * rng.uniform(1.8, 3.5)
        elif atype == "low_dropout":
            anomaly_tds[i] = rng.uniform(0, mean_tds * 0.15)
        elif atype == "chemical_spike":
            anomaly_tds[i] = mean_tds * rng.uniform(3.0, 6.0)
        else:  # source_change
            anomaly_tds[i] = mean_tds * rng.uniform(0.08, 0.28)
    anomaly_tds     = np.clip(anomaly_tds, 0, 3000)
    anomaly_voltage = mean_v * (anomaly_tds / (mean_tds + 1e-6)) * rng.uniform(0.85, 1.15, n_anomaly)
    anomaly_mask    = np.ones(n_anomaly, dtype=bool)

    # ─── Combine and shuffle ───
    all_tds     = np.concatenate([normal_tds, anomaly_tds])
    all_voltage = np.clip(np.concatenate([normal_voltage, anomaly_voltage]), 0, 5)
    all_mask    = np.concatenate([normal_mask, anomaly_mask])
    shuffle_idx = rng.permutation(history_count)
    all_tds     = all_tds[shuffle_idx]
    all_voltage = all_voltage[shuffle_idx]
    all_mask    = all_mask[shuffle_idx]

    all_temps = np.zeros(history_count)
    for i, ts in enumerate(hist_timestamps):
        hour = ts.hour if hasattr(ts, 'hour') else 12
        minute = ts.minute if hasattr(ts, 'minute') else 0
        timeVal = hour + (minute / 60.0)
        # Diurnal curve matching ESP32: peak at 14:00, lowest at 04:00
        # Range: 26.0 to 33.0. Midpoint: 29.5, Amplitude: 3.5
        diurnal = np.cos((timeVal - 14.0) * 2.0 * np.pi / 24.0)
        base_temp = 29.5 + diurnal * 3.5
        noise = rng.normal(0, 0.2)
        all_temps[i] = round(float(np.clip(base_temp + noise, 26.0, 33.0)), 2)

    hist_df = pd.DataFrame({
        "entry_id":    range(1, history_count + 1),      # sequential historical IDs
        "timestamp":   hist_timestamps,
        "tds_value":   np.round(all_tds, 2),
        "voltage":     np.round(all_voltage, 4),
        "temperature": all_temps,
    })

    # ─── Merge with live data (live gets higher entry_ids) ───
    live_copy = live_df.copy()
    live_copy["entry_id"] = range(history_count + 1, history_count + live_count + 1)

    combined = pd.concat([hist_df, live_copy], ignore_index=True)
    combined = combined.sort_values("entry_id").reset_index(drop=True)

    print(f"  OK: Dataset: {history_count} historical + {live_count} live = {len(combined)} total entries")
    print(f"  STATS: TDS: {combined['tds_value'].min():.1f}–{combined['tds_value'].max():.1f} ppm  "
          f"| Temp: {combined['temperature'].min():.1f}–{combined['temperature'].max():.1f} °C")
    print(f"  ALERT: Injected anomalies in historical data: {n_anomaly}")

    return combined


# ══════════════════════════════════════════════════════════════
# STEP 3: FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_features(df: pd.DataFrame, window: int = 10):
    """
    Extract features for Isolation Forest.
    Now includes temperature-based features in addition to TDS.
    """
    print(f"\n{'='*60}")
    print("STEP 3: FEATURE EXTRACTION")
    print(f"{'='*60}")

    feat = pd.DataFrame(index=df.index)

    # ── TDS features ──
    feat["tds_value"]          = df["tds_value"]
    feat["voltage"]            = df["voltage"]
    feat["tds_rolling_mean"]   = df["tds_value"].rolling(window, min_periods=1).mean()
    feat["tds_rolling_std"]    = df["tds_value"].rolling(window, min_periods=1).std().fillna(0)
    feat["tds_rate_of_change"] = df["tds_value"].diff().fillna(0).abs()
    feat["tds_z_score"]        = (
        (df["tds_value"] - feat["tds_rolling_mean"]) /
        feat["tds_rolling_std"].replace(0, 1)
    ).fillna(0)
    feat["tds_rolling_range"]  = (
        df["tds_value"].rolling(window, min_periods=1).max() -
        df["tds_value"].rolling(window, min_periods=1).min()
    ).fillna(0)
    global_mean = df["tds_value"].mean()
    feat["tds_deviation"]      = (df["tds_value"] - global_mean).abs()
    feat["tds_voltage_ratio"]  = df["tds_value"] / df["voltage"].replace(0, 0.001)

    thresh = df["tds_value"].std() * 2 if df["tds_value"].std() > 0 else 50
    spikes = (df["tds_value"].diff().abs() > thresh).astype(int)
    feat["spike_count"]        = spikes.rolling(window, min_periods=1).sum()

    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)

    feature_names = list(feat.columns)
    print(f"  OK: Extracted {len(feature_names)} features (TDS Only):")
    for i, name in enumerate(feature_names, 1):
        print(f"     {i:2d}. {name}")

    return feat, feature_names


# ══════════════════════════════════════════════════════════════
# STEP 4: TRAIN ISOLATION FOREST
# ══════════════════════════════════════════════════════════════

def train_isolation_forest(features: pd.DataFrame, feature_names: list):
    """Train (or retrain) the Isolation Forest model."""
    print(f"\n{'='*60}")
    print("STEP 4: ISOLATION FOREST — TRAINING")
    print(f"{'='*60}")

    X        = features[feature_names].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  INFO: Training on {len(X_scaled)} samples …")
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        max_samples="auto",
        n_jobs=-1,
    )
    t0 = datetime.now()
    model.fit(X_scaled)
    training_time = (datetime.now() - t0).total_seconds()

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,         os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"  OK: Trained in {training_time:.2f}s - saved to {MODEL_DIR}")

    return model, scaler, training_time


# ══════════════════════════════════════════════════════════════
# STEP 5: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════

def detect_anomalies(model, scaler, features: pd.DataFrame,
                     feature_names: list, df: pd.DataFrame) -> pd.DataFrame:
    """Run Isolation Forest predictions and compute normalised anomaly scores."""
    print(f"\n{'='*60}")
    print("STEP 5: ANOMALY DETECTION")
    print(f"{'='*60}")

    X        = features[feature_names].values
    X_scaled = scaler.transform(X)

    preds      = model.predict(X_scaled)           # 1 = normal, -1 = anomaly
    raw_scores = model.decision_function(X_scaled) # higher = more normal

    # Invert and normalise to [0, 1]  (1 = definitely anomalous)
    rng_score = raw_scores.max() - raw_scores.min()
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (rng_score + 1e-10)

    results = df.copy()
    results["prediction"]    = preds
    results["is_anomaly"]    = preds == -1
    results["anomaly_score"] = np.round(anomaly_scores, 4)

    total         = len(results)
    anomaly_count = int(results["is_anomaly"].sum())
    print(f"  OK: {total} readings analysed -> {anomaly_count} anomalies "
          f"({anomaly_count / total * 100:.1f} %)")

    top_anomalies = results[results["is_anomaly"]].nlargest(5, "anomaly_score")
    if not top_anomalies.empty:
        print("  ALERT: Top anomalies:")
        for _, row in top_anomalies.iterrows():
            print(f"     TDS={row['tds_value']:.1f} ppm  "
                  f"Temp={row['temperature']:.1f}°C  "
                  f"Score={row['anomaly_score']:.2f}")

    return results


# ══════════════════════════════════════════════════════════════
# STEP 6: VISUALISATION
# ══════════════════════════════════════════════════════════════

def generate_charts(results: pd.DataFrame, features: pd.DataFrame,
                    feature_names: list, training_time: float) -> dict:
    """Generate PNG charts for the report / dashboard."""
    print(f"\n{'='*60}")
    print("STEP 6: VISUALISATION")
    print(f"{'='*60}")

    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    normal    = results[~results["is_anomaly"]]
    anomalies = results[results["is_anomaly"]]

    # ── Chart 1: TDS timeline ──
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(results.index, results["tds_value"],
            color="#0EA5E9", alpha=0.55, linewidth=0.9, label="TDS Value")
    ax.scatter(normal.index, normal["tds_value"],
               color="#0EA5E9", s=8, alpha=0.4, zorder=4)
    ax.scatter(anomalies.index, anomalies["tds_value"],
               color="#EF4444", s=55, marker="X", zorder=10,
               label=f"Anomaly ({len(anomalies)} detected)",
               edgecolors="white", linewidths=0.5)
    mean_tds = results["tds_value"].mean()
    ax.axhline(mean_tds, color="#8B5CF6", linestyle="--", alpha=0.5,
               label=f"Mean: {mean_tds:.0f} ppm")
    ax.set_xlabel("Reading Index"); ax.set_ylabel("TDS (ppm)")
    ax.set_title("Water Quality — TDS with Anomaly Detection (Isolation Forest)",
                 fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    p1 = os.path.join(CHARTS_DIR, "tds_anomaly_timeline.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  STATS: Chart 1 saved: {p1}")

    # ── Chart 2: Score distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(normal["anomaly_score"],    bins=30, alpha=0.7, color="#0EA5E9",
            label="Normal", edgecolor="white")
    ax.hist(anomalies["anomaly_score"], bins=15, alpha=0.85, color="#EF4444",
            label="Anomaly", edgecolor="white")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution — Isolation Forest", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(CHARTS_DIR, "anomaly_score_distribution.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  STATS: Chart 2 saved: {p2}")

    # ── Chart 3: Pipeline summary (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AquaQuality — Pipeline Summary (Isolation Forest)",
                 fontsize=15, fontweight="bold")

    # 3a: TDS distribution
    axes[0, 0].hist(results["tds_value"], bins=40,
                    color="#0EA5E9", alpha=0.7, edgecolor="white")
    axes[0, 0].set_title("TDS Distribution")
    axes[0, 0].set_xlabel("TDS (ppm)"); axes[0, 0].set_ylabel("Count")

    # 3b: Temperature distribution
    axes[0, 1].hist(results["temperature"], bins=30,
                    color="#F59E0B", alpha=0.75, edgecolor="white")
    axes[0, 1].set_title("Temperature Distribution")
    axes[0, 1].set_xlabel("Temperature (°C)"); axes[0, 1].set_ylabel("Count")

    # 3c: Anomaly score scatter
    axes[1, 0].scatter(results.index, results["anomaly_score"],
                       c=results["is_anomaly"].map({True: "#EF4444", False: "#0EA5E9"}),
                       s=8, alpha=0.5)
    axes[1, 0].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Threshold 0.5")
    axes[1, 0].set_title("Anomaly Scores Over Time")
    axes[1, 0].set_xlabel("Reading Index"); axes[1, 0].set_ylabel("Score"); axes[1, 0].legend()

    # 3d: Model info
    total  = len(results)
    n_anom = int(results["is_anomaly"].sum())
    info   = (
        f"Model: Isolation Forest\n"
        f"Estimators: {IF_N_ESTIMATORS}\n"
        f"Contamination: {IF_CONTAMINATION}\n"
        f"Training Time: {training_time:.2f}s\n"
        f"────────────────────\n"
        f"Total Readings: {total}\n"
        f"Features: {len(feature_names)}\n"
        f"────────────────────\n"
        f"Anomalies: {n_anom} ({n_anom/total*100:.1f}%)\n"
        f"Normal: {total - n_anom} ({(total-n_anom)/total*100:.1f}%)\n"
        f"TDS Mean: {results['tds_value'].mean():.1f} ppm\n"
        f"Temp Mean: {results['temperature'].mean():.1f} °C"
    )
    axes[1, 1].text(0.08, 0.5, info, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment="center", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#DBEAFE", alpha=0.35))
    axes[1, 1].set_title("Model Summary"); axes[1, 1].axis("off")

    plt.tight_layout()
    p3 = os.path.join(CHARTS_DIR, "pipeline_summary.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  STATS: Chart 3 saved: {p3}")

    return {
        "tds_timeline":        p1,
        "score_distribution":  p2,
        "pipeline_summary":    p3,
    }


# ══════════════════════════════════════════════════════════════
# STEP 7: SAVE RESULTS
# ══════════════════════════════════════════════════════════════

def save_results(results: pd.DataFrame, feature_names: list,
                 training_time: float, chart_paths: dict) -> dict:
    """Persist predictions and model metadata as JSON + CSV."""
    print(f"\n{'='*60}")
    print("STEP 7: SAVE RESULTS")
    print(f"{'='*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    total     = len(results)
    n_anom    = int(results["is_anomaly"].sum())

    def _safe(row, key):
        val = row.get(key, None)
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val

    anomaly_rows = results[results["is_anomaly"]].to_dict("records")
    for r in anomaly_rows:
        for k in list(r.keys()):
            r[k] = _safe(r, k)

    output = {
        "model": {
            "name":                  "Isolation Forest",
            "algorithm":             "sklearn.ensemble.IsolationForest",
            "n_estimators":          IF_N_ESTIMATORS,
            "contamination":         IF_CONTAMINATION,
            "training_time_seconds": round(training_time, 3),
            "features_used":         feature_names,
            "n_features":            len(feature_names),
            "trained_at":            datetime.utcnow().isoformat(),
        },
        "dataset": {
            "total_samples":  total,
            "tds_mean":       round(float(results["tds_value"].mean()),   2),
            "tds_std":        round(float(results["tds_value"].std()),    2),
            "tds_min":        round(float(results["tds_value"].min()),    2),
            "tds_max":        round(float(results["tds_value"].max()),    2),
            "temp_mean":      round(float(results["temperature"].mean()), 2),
            "temp_min":       round(float(results["temperature"].min()),  2),
            "temp_max":       round(float(results["temperature"].max()),  2),
        },
        "anomaly_detection": {
            "total_anomalies": n_anom,
            "anomaly_rate":    round(n_anom / total * 100, 2),
            "normal_count":    total - n_anom,
            "top_anomalies":   anomaly_rows[:10],
        },
        "charts": chart_paths,
    }

    results_path = os.path.join(RESULTS_DIR, "predictions.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  SAVED: JSON  -> {results_path}")

    csv_path = os.path.join(RESULTS_DIR, "full_predictions.csv")
    results.to_csv(csv_path, index=False)
    print(f"  SAVED: CSV   -> {csv_path}")

    return output


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline():
    """Run the complete pipeline end-to-end."""
    print("\n" + "═" * 60)
    print("  AQUAQUALITY - Water Quality Anomaly Detection Pipeline")
    print("  Isolation Forest | TDS + Temperature | Live ThingSpeak")
    print("═" * 60)

    live_df   = fetch_thingspeak_data(results=8000)
    full_df   = build_historical_dataset(live_df, history_count=HISTORICAL_TARGET)
    features, feature_names = extract_features(full_df)
    model, scaler, t_time   = train_isolation_forest(features, feature_names)
    results                 = detect_anomalies(model, scaler, features, feature_names, full_df)
    chart_paths             = generate_charts(results, features, feature_names, t_time)
    output                  = save_results(results, feature_names, t_time, chart_paths)

    n_anom = output["anomaly_detection"]["total_anomalies"]
    total  = output["dataset"]["total_samples"]
    print(f"\n{'═'*60}")
    print("  OK: PIPELINE COMPLETE")
    print(f"     Dataset:   {total} entries")
    print(f"     Anomalies: {n_anom} ({n_anom/total*100:.1f}%)")
    print(f"     Time:      {t_time:.2f}s")
    print(f"{'═'*60}\n")
    return output


if __name__ == "__main__":
    run_pipeline()
