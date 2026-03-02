import { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, Cell,
} from 'recharts';
import {
  Droplets, Cpu, Cloud, Server, BrainCircuit, ShieldAlert,
  Bell, ChevronRight, Activity, Gauge, AlertTriangle,
  CheckCircle2, Wifi, Zap, Clock, TrendingUp, Thermometer,
} from 'lucide-react';
import {
  fetchLiveReadings, fetchModelInfo, fetchPredictions, fetchFullDataset,
  checkMLHealth, classifyWaterQuality, formatTimestamp,
  retrainModel,
  type SensorReading, type ModelInfo, type PredictResponse, type FullDatasetResponse,
} from '../services/api';

/* ═══════════════════════════════════════
   DASHBOARD PAGE
   Shows: Pipeline → Status → KPIs → Charts
   ═══════════════════════════════════════ */

export default function Dashboard() {
  const [liveData,    setLiveData]    = useState<SensorReading[]>([]);
  const [modelInfo,   setModelInfo]   = useState<ModelInfo | null>(null);
  const [predictions, setPredictions] = useState<PredictResponse | null>(null);
  const [fullDataset, setFullDataset] = useState<FullDatasetResponse | null>(null);
  const [mlOnline,    setMlOnline]    = useState(false);
  const [loading,     setLoading]     = useState(true);
  const [chartTab,    setChartTab]    = useState<'full' | 'live'>('full');
  const [retraining,  setRetraining]  = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const live = await fetchLiveReadings(100);
      setLiveData(live);

      try {
        await checkMLHealth();
        setMlOnline(true);

        const [info, preds, full] = await Promise.all([
          fetchModelInfo(),
          fetchPredictions(),
          fetchFullDataset(),
        ]);
        setModelInfo(info);
        setPredictions(preds);
        setFullDataset(full);
      } catch {
        setMlOnline(false);
      }
    } catch (err) {
      console.error('Data load error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 60_000); // refresh every 60 s
    return () => clearInterval(interval);
  }, [loadData]);

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      await retrainModel();
      await loadData();
    } catch (e) {
      console.error('Retrain failed:', e);
    } finally {
      setRetraining(false);
    }
  };

  // Derived
  const latestReading  = liveData.length > 0 ? liveData[liveData.length - 1] : null;
  const waterQuality   = latestReading ? classifyWaterQuality(latestReading.tdsValue) : null;

  const fullChartData  = (fullDataset?.predictions || []).map((p, i) => ({
    index:        i,
    tds:          p.tds_value,
    temperature:  p.temperature,
    score:        p.anomaly_score,
    isAnomaly:    p.is_anomaly,
  }));

  const liveChartData  = liveData.map((r, i) => ({
    index:       i,
    name:        `#${r.entryId}`,
    tds:         r.tdsValue,
    voltage:     r.voltage,
    temperature: r.temperature,
  }));

  const latestAnomaly  = predictions?.predictions?.filter(p => p.is_anomaly).slice(-1)[0];
  const avgTemp        = latestReading?.temperature ?? null;

  const isDeviceOnline = (() => {
    if (!latestReading) return false;
    const minsSince = (Date.now() - new Date(latestReading.timestamp).getTime()) / 60000;
    return minsSince < 1; // Offline if no ping for 1 min
  })();

  if (loading && liveData.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: 16 }}>
        <Activity className="text-accent" size={40} style={{ animation: 'pulse-glow 1.5s infinite' }} />
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>Loading sensor data from ThingSpeak…</p>
      </div>
    );
  }

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* ── Page Header ── */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1>⚡ AquaQuality Dashboard</h1>
          <p>Real-time IoT water quality monitoring with ML-powered anomaly detection</p>
        </div>
        {mlOnline && (
          <button
            className="btn primary"
            onClick={handleRetrain}
            disabled={retraining}
            style={{ fontSize: 12, padding: '7px 14px', opacity: retraining ? 0.6 : 1 }}
          >
            <BrainCircuit size={13} />
            {retraining ? 'Retraining…' : 'Retrain Model'}
          </button>
        )}
      </div>



      {/* ── Status Alert ── */}
      {latestReading && (
        <div className={`alert-banner ${
          latestAnomaly ? 'danger' :
          waterQuality && (waterQuality.label === 'Good' || waterQuality.label === 'Excellent') ? 'success' : 'warning'
        }`}>
          {latestAnomaly ? (
            <>
              <AlertTriangle size={20} />
              <span>
                <strong>ANOMALY DETECTED</strong> — TDS: {latestAnomaly.tds_value} ppm |
                Temp: {latestAnomaly.temperature?.toFixed(1)}°C |
                Score: {(latestAnomaly.anomaly_score * 100).toFixed(0)}% |
                {formatTimestamp(latestAnomaly.timestamp)}
              </span>
            </>
          ) : waterQuality ? (
            <>
              <CheckCircle2 size={20} />
              <span>
                Water Quality: <strong>{waterQuality.label}</strong> — {waterQuality.description}
                &nbsp;(TDS: {latestReading.tdsValue.toFixed(1)} ppm | Temp: {latestReading.temperature.toFixed(1)}°C)
              </span>
            </>
          ) : null}
        </div>
      )}

      {/* ── KPI Cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 16 }}>
        {/* Live TDS */}
        <div className="glass-card accent">
          <div className="stat-label">Live TDS Value</div>
          <div className="stat-value" style={{ color: waterQuality?.color || 'var(--accent-text)' }}>
            {latestReading?.tdsValue.toFixed(1) || '—'}
            <span style={{ fontSize: 14, fontWeight: 400, marginLeft: 4, color: 'var(--text-muted)' }}>ppm</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Gauge size={14} />
            {waterQuality?.label || 'N/A'} • {latestReading?.voltage.toFixed(3) || '0'} V
          </div>
        </div>

        {/* Temperature */}
        <div className="glass-card" style={{ borderColor: '#f59e0b30' }}>
          <div className="stat-label">Water Temperature</div>
          <div className="stat-value" style={{ color: '#f59e0b' }}>
            {avgTemp !== null ? avgTemp.toFixed(1) : '—'}
            <span style={{ fontSize: 14, fontWeight: 400, marginLeft: 4, color: 'var(--text-muted)' }}>°C</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Thermometer size={14} />
            Live from ESP32
          </div>
        </div>

        {/* Total Readings */}
        <div className="glass-card">
          <div className="stat-label">Total Readings</div>
          <div className="stat-value text-accent">
            {fullDataset?.total_readings || liveData.length}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <TrendingUp size={14} />
            Continuous monitoring feed
          </div>
        </div>

        {/* Anomalies */}
        <div className={`glass-card ${(fullDataset?.anomalies_found || 0) > 0 ? 'danger' : 'success'}`}>
          <div className="stat-label">Anomalies Detected</div>
          <div className="stat-value text-red">
            {fullDataset?.anomalies_found || predictions?.anomalies_found || 0}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <ShieldAlert size={14} />
            Rate: {fullDataset?.anomaly_rate || predictions?.anomaly_rate || 0}%
          </div>
        </div>

        {/* ML Model */}
        <div className="glass-card">
          <div className="stat-label">ML Model</div>
          <div className="stat-value" style={{ fontSize: 18, color: '#f59e0b' }}>
            {mlOnline ? 'Isolation Forest' : 'Offline'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <BrainCircuit size={14} />
            {modelInfo?.model
              ? `${modelInfo.model.n_estimators} trees • ${modelInfo.model.training_time_seconds}s train`
              : 'Start python ml/api.py'}
          </div>
        </div>
      </div>

      {/* ── Main Content Grid ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20 }}>
        {/* LEFT: Chart Area */}
        <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
          {/* Tab bar */}
          <div style={{ padding: '16px 20px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="custom-tabs">
              <button className={`custom-tab ${chartTab === 'full' ? 'active' : ''}`} onClick={() => setChartTab('full')}>
                All Readings ({fullDataset?.total_readings || 0})
              </button>
              <button className={`custom-tab ${chartTab === 'live' ? 'active' : ''}`} onClick={() => setChartTab('live')}>
                Live ThingSpeak ({liveData.length})
              </button>
            </div>
          </div>

          <div style={{ padding: 20 }}>
            {chartTab === 'full' && fullChartData.length > 0 ? (
              <>
                <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>
                  TDS Readings — All {fullChartData.length} Entries
                  <span style={{ fontSize: 12, fontWeight: 400, color: '#fca5a5', marginLeft: 8 }}>
                    {fullDataset?.anomalies_found} anomalies
                  </span>
                </h3>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 16 }}>
                  Historical + live ThingSpeak readings with Isolation Forest predictions
                </p>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={fullChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="index" tick={{ fontSize: 10, fill: '#64748b' }}
                      label={{ value: 'Reading Index', position: 'insideBottom', offset: -5, style: { fontSize: 11, fill: '#64748b' } }} />
                    <YAxis tick={{ fontSize: 10, fill: '#64748b' }}
                      label={{ value: 'TDS (ppm)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }}
                      labelStyle={{ color: '#94a3b8' }}
                      itemStyle={{ color: '#f1f5f9' }}
                      labelFormatter={(label) => {
                        const pt = fullChartData[label as number];
                        return pt ? `#${label}${pt.isAnomaly ? ' | ⚠ ANOMALY' : ''}` : `#${label}`;
                      }}
                      formatter={(val: number, name: string) => {
                        if (name === 'TDS') return [`${val.toFixed(2)} ppm`, name];
                        if (name === 'Temp') return [`${val.toFixed(1)} °C`, name];
                        return [val, name];
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="tds"         stroke="#06b6d4" strokeWidth={1} dot={false} name="TDS" />
                    <Line type="monotone" dataKey="temperature" stroke="#f59e0b" strokeWidth={1} dot={false} name="Temp" />
                  </LineChart>
                </ResponsiveContainer>

                {/* Anomaly scatter */}
                <h4 style={{ fontSize: 13, fontWeight: 600, marginTop: 20, marginBottom: 8 }}>Anomaly Score Map</h4>
                <ResponsiveContainer width="100%" height={180}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="index" type="number" tick={{ fontSize: 10, fill: '#64748b' }}
                      label={{ value: 'Index', position: 'insideBottom', offset: -5, style: { fontSize: 11, fill: '#64748b' } }} />
                    <YAxis dataKey="tds" tick={{ fontSize: 10, fill: '#64748b' }}
                      label={{ value: 'TDS', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }}
                      formatter={(val: number, name: string) => [`${typeof val === 'number' ? val.toFixed(2) : val}`, name]}
                    />
                    <Scatter data={fullChartData} name="TDS">
                      {fullChartData.map((entry, i) => (
                        <Cell key={i} fill={entry.isAnomaly ? '#ef4444' : '#06b6d4'}
                          r={entry.isAnomaly ? 5 : 2} fillOpacity={entry.isAnomaly ? 1 : 0.3} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </>
            ) : chartTab === 'live' ? (
              <>
                <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>Live ThingSpeak Readings</h3>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 16 }}>
                  Latest {liveData.length} readings from ESP32 via ThingSpeak (HTTP/JSON)
                </p>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={liveChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#64748b' }}
                      label={{ value: 'TDS (ppm)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }}
                      formatter={(val: number, name: string) => {
                        if (name === 'TDS')   return [`${val.toFixed(2)} ppm`, name];
                        if (name === 'Temp')  return [`${val.toFixed(1)} °C`,  name];
                        return [`${val.toFixed(4)} V`, name];
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="tds"         stroke="#06b6d4" strokeWidth={2} dot={false} name="TDS" />
                    <Line type="monotone" dataKey="temperature" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Temp" />
                    <Line type="monotone" dataKey="voltage"     stroke="#8b5cf6" strokeWidth={1}   dot={false} name="Voltage" />
                  </LineChart>
                </ResponsiveContainer>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
                <BrainCircuit size={40} style={{ opacity: 0.3, marginBottom: 12 }} />
                <p style={{ fontSize: 13 }}>Start Python API to load full dataset</p>
                <code style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 8, display: 'block' }}>
                  python ml/api.py
                </code>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT: Sidebar Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* System Status */}
          <div className="glass-card">
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>System Status</h3>
            {[
              { icon: <Droplets size={16} />,    name: 'TDS Sensor',   status: isDeviceOnline ? 'Active' : 'Offline',    ok: isDeviceOnline },
              { icon: <Thermometer size={16} />, name: 'Temp Sensor',  status: isDeviceOnline ? 'Active' : 'Offline',    ok: isDeviceOnline },
              { icon: <Wifi size={16} />,        name: 'ESP32 WiFi',   status: isDeviceOnline ? 'Connected' : 'Offline', ok: isDeviceOnline },
              { icon: <Cloud size={16} />,       name: 'ThingSpeak',   status: liveData.length > 0 ? 'Online' : 'Offline',   ok: liveData.length > 0 },
              { icon: <BrainCircuit size={16} />,name: 'ML Model',     status: mlOnline ? 'Active' : 'Offline',         ok: mlOnline },
            ].map((item, i, arr) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 13 }}>
                  {item.icon}
                  <span>{item.name}</span>
                </div>
                <span className={`badge ${item.ok ? 'green' : 'orange'}`}>{item.status}</span>
              </div>
            ))}
            {latestReading && (
              <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-dim)' }}>
                <div>Last: {formatTimestamp(latestReading.timestamp)}</div>
                <div>Voltage: {latestReading.voltage.toFixed(3)} V</div>
                <div>Temperature: {latestReading.temperature.toFixed(1)} °C</div>
              </div>
            )}
          </div>

          {/* Dataset Stats */}
          {fullDataset?.dataset_stats && (
            <div className="glass-card">
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Dataset Statistics</h3>
              {[
                { name: 'TDS Mean',   value: `${fullDataset.dataset_stats.tds_mean} ppm`,  color: '#06b6d4' },
                { name: 'TDS Std',    value: `${fullDataset.dataset_stats.tds_std} ppm`,   color: '#8b5cf6' },
                { name: 'TDS Min',    value: `${fullDataset.dataset_stats.tds_min} ppm`,   color: '#10b981' },
                { name: 'TDS Max',    value: `${fullDataset.dataset_stats.tds_max} ppm`,   color: '#ef4444' },
                { name: 'Temp Mean',  value: `${fullDataset.dataset_stats.temp_mean} °C`,  color: '#f59e0b' },
                { name: 'Temp Range', value: `${fullDataset.dataset_stats.temp_min}–${fullDataset.dataset_stats.temp_max} °C`, color: '#f97316' },
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0' }}>
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{item.name}</span>
                  <span className="mono" style={{ fontSize: 12, color: item.color, fontWeight: 600 }}>{item.value}</span>
                </div>
              ))}
            </div>
          )}

          {/* Model Details */}
          {modelInfo?.model && (
            <div className="glass-card">
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <BrainCircuit size={16} style={{ color: '#f59e0b' }} />
                  Model Info
                </span>
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {[
                  { label: 'Algorithm',   value: 'Isolation Forest' },
                  { label: 'Estimators',  value: modelInfo.model.n_estimators },
                  { label: 'Features',    value: modelInfo.model.n_features },
                  { label: 'Train Time',  value: `${modelInfo.model.training_time_seconds}s` },
                  { label: 'TDS Mean',    value: `${modelInfo.dataset?.tds_mean}` },
                  { label: 'Temp Mean',   value: `${modelInfo.dataset?.temp_mean} °C` },
                ].map((item, i) => (
                  <div key={i}>
                    <div style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{item.label}</div>
                    <div className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{item.value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Anomalies */}
          {predictions?.predictions && predictions.predictions.filter(p => p.is_anomaly).length > 0 && (
            <div className="glass-card danger">
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: '#fca5a5', display: 'flex', alignItems: 'center', gap: 8 }}>
                <AlertTriangle size={16} />
                Recent Anomalies
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {predictions.predictions.filter(p => p.is_anomaly).slice(-4).map((a, i) => (
                  <div key={i} style={{ fontSize: 11, padding: '6px 10px', background: 'rgba(239,68,68,0.08)', borderRadius: 6, border: '1px solid rgba(239,68,68,0.15)' }}>
                    <span className="mono" style={{ color: '#fca5a5' }}>TDS: {a.tds_value} ppm</span>
                    <span style={{ color: '#f59e0b', marginLeft: 8 }}>T: {a.temperature?.toFixed(1)}°C</span>
                    <span style={{ color: 'var(--text-dim)', marginLeft: 8 }}>Score: {(a.anomaly_score * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Communication Protocol Summary ── */}
      <div className="glass-card" style={{ marginTop: 0 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Communication Protocol & Data Flow</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
          {[
            { stage: 'Sensor → MCU',     protocol: 'ADC (Analog)',       detail: 'TDS/Temp sensors → ESP32 GPIO (12-bit ADC)', icon: <Zap size={16} /> },
            { stage: 'MCU → Cloud',      protocol: 'HTTP POST / JSON',   detail: 'ESP32 WiFi → ThingSpeak REST API',           icon: <Cloud size={16} /> },
            { stage: 'Cloud → Backend',  protocol: 'HTTP GET / JSON',    detail: 'ThingSpeak feeds → React Dashboard',         icon: <Server size={16} /> },
            { stage: 'Backend → ML',     protocol: 'REST API / JSON',    detail: 'Flask API → Isolation Forest (sklearn)',      icon: <BrainCircuit size={16} /> },
          ].map((item, i) => (
            <div key={i} style={{ padding: 16, background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, color: 'var(--accent-text)', fontSize: 12, fontWeight: 600 }}>
                {item.icon}
                {item.stage}
              </div>
              <div className="mono" style={{ fontSize: 12, color: 'var(--text-primary)', marginBottom: 4 }}>{item.protocol}</div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>{item.detail}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
