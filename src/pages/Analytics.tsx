import { useState, useEffect } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, BarChart, Bar, Legend,
  LineChart, Line,
} from 'recharts';
import {
  BrainCircuit, BarChart3, Activity, TrendingUp,
  ShieldAlert, Thermometer,
} from 'lucide-react';
import {
  fetchFullDataset, fetchModelInfo,
  type FullDatasetResponse, type ModelInfo,
} from '../services/api';

/* ═══════════════════════════════════════
   ANALYTICS PAGE
   Full dataset analysis: anomaly scores,
   TDS distribution, temperature trends,
   and statistical summary
   ═══════════════════════════════════════ */

export default function Analytics() {
  const [fullDataset, setFullDataset] = useState<FullDatasetResponse | null>(null);
  const [modelInfo,   setModelInfo]   = useState<ModelInfo | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [viewTab, setViewTab] = useState<'scores' | 'distribution' | 'temperature'>('scores');

  useEffect(() => {
    async function load() {
      try {
        const [full, info] = await Promise.all([fetchFullDataset(), fetchModelInfo()]);
        setFullDataset(full);
        setModelInfo(info);
      } catch (err) {
        console.error('Analytics load error:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: 16 }}>
        <Activity className="text-accent" size={40} style={{ animation: 'pulse-glow 1.5s infinite' }} />
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>Loading analytics data…</p>
      </div>
    );
  }

  if (!fullDataset) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: 16 }}>
        <BrainCircuit size={48} style={{ opacity: 0.3, color: 'var(--text-muted)' }} />
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>Python ML API is offline</p>
        <code style={{ fontSize: 12, color: 'var(--text-dim)' }}>Run: python ml/api.py</code>
      </div>
    );
  }

  // ── Chart data ──
  const scoreData = fullDataset.predictions.map((p, i) => ({
    index:     i,
    score:     p.anomaly_score,
    isAnomaly: p.is_anomaly,
    tds:       p.tds_value,
    temp:      p.temperature,
  }));

  // TDS histogram
  const bins      = 20;
  const allTds    = fullDataset.predictions.map(p => p.tds_value);
  const minTds    = Math.min(...allTds);
  const maxTds    = Math.max(...allTds);
  const binWidth  = (maxTds - minTds) / bins || 1;
  const histogram = Array.from({ length: bins }, (_, i) => {
    const lo        = minTds + i * binWidth;
    const hi        = lo + binWidth;
    const count     = allTds.filter(v => v >= lo && v < hi).length;
    const anomCount = fullDataset.predictions.filter(p => p.tds_value >= lo && p.tds_value < hi && p.is_anomaly).length;
    return {
      range:    `${lo.toFixed(0)}–${hi.toFixed(0)}`,
      total:    count,
      anomalies: anomCount,
      normal:   count - anomCount,
    };
  });

  // Temperature trend
  const tempData = fullDataset.predictions
    .filter((_, i) => i % Math.max(1, Math.floor(fullDataset.predictions.length / 200)) === 0)
    .map((p, i) => ({
      index:     i,
      temp:      p.temperature,
      isAnomaly: p.is_anomaly,
    }));

  const anomalyPreds = fullDataset.predictions.filter(p =>  p.is_anomaly);
  const normalPreds  = fullDataset.predictions.filter(p => !p.is_anomaly);

  const stats = fullDataset.dataset_stats;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Page Header */}
      <div className="page-header">
        <h1>📊 Data Analytics</h1>
        <p>Comprehensive analysis of {fullDataset.total_readings} sensor readings with Isolation Forest predictions</p>
      </div>

      {/* Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16 }}>
        <div className="glass-card">
          <div className="stat-label">Total Samples</div>
          <div className="stat-value text-accent">{fullDataset.total_readings}</div>
        </div>
        <div className="glass-card">
          <div className="stat-label">TDS Mean</div>
          <div className="stat-value" style={{ color: '#06b6d4' }}>{stats?.tds_mean ?? '—'} <span style={{ fontSize: 13, fontWeight: 400 }}>ppm</span></div>
        </div>
        <div className="glass-card">
          <div className="stat-label">Temp Mean</div>
          <div className="stat-value" style={{ color: '#f59e0b' }}>{stats?.temp_mean ?? '—'} <span style={{ fontSize: 13, fontWeight: 400 }}>°C</span></div>
        </div>
        <div className="glass-card danger">
          <div className="stat-label">Anomalies</div>
          <div className="stat-value text-red">{fullDataset.anomalies_found}</div>
        </div>
        <div className="glass-card">
          <div className="stat-label">Anomaly Rate</div>
          <div className="stat-value text-orange">{fullDataset.anomaly_rate}%</div>
        </div>
      </div>

      {/* Chart Tabs */}
      <div className="custom-tabs">
        <button className={`custom-tab ${viewTab === 'scores' ? 'active' : ''}`} onClick={() => setViewTab('scores')}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><ShieldAlert size={14} /> Anomaly Scores</span>
        </button>
        <button className={`custom-tab ${viewTab === 'distribution' ? 'active' : ''}`} onClick={() => setViewTab('distribution')}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><BarChart3 size={14} /> TDS Distribution</span>
        </button>
        <button className={`custom-tab ${viewTab === 'temperature' ? 'active' : ''}`} onClick={() => setViewTab('temperature')}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Thermometer size={14} /> Temperature Trend</span>
        </button>
      </div>

      {/* Chart Content */}
      <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: 20 }}>

          {viewTab === 'scores' && (
            <>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>
                Anomaly Score Distribution
                <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--text-muted)', marginLeft: 8 }}>threshold = 0.5</span>
              </h3>
              <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 16 }}>
                Points above 0.5 are classified as anomalies by Isolation Forest. Red = anomaly, blue = normal.
              </p>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="index" type="number" tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'Reading Index', position: 'insideBottom', offset: -5, style: { fontSize: 11, fill: '#64748b' } }} />
                  <YAxis dataKey="score" domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'Anomaly Score', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }}
                    formatter={(val: number, name: string) => {
                      if (name === 'score') return [`${(val * 100).toFixed(1)}%`, 'Score'];
                      return [val, name];
                    }}
                    labelFormatter={(label) => {
                      const pt = scoreData[label as number];
                      return `#${label} | TDS: ${pt?.tds?.toFixed(1)} ppm | Temp: ${pt?.temp?.toFixed(1)}°C | ${pt?.isAnomaly ? 'ANOMALY' : 'Normal'}`;
                    }}
                  />
                  <Scatter data={scoreData} name="Score">
                    {scoreData.map((entry, i) => (
                      <Cell key={i} fill={entry.isAnomaly ? '#ef4444' : '#06b6d4'}
                        r={entry.isAnomaly ? 5 : 2} fillOpacity={entry.isAnomaly ? 0.9 : 0.35} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </>
          )}

          {viewTab === 'distribution' && (
            <>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>TDS Value Distribution</h3>
              <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 16 }}>
                Histogram of TDS readings — blue = normal, red = anomalous
              </p>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={histogram}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="range" tick={{ fontSize: 9, fill: '#64748b' }} angle={-30} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }}
                  />
                  <Legend />
                  <Bar dataKey="normal"    stackId="a" fill="#06b6d4" name="Normal"   radius={[0, 0, 0, 0]} />
                  <Bar dataKey="anomalies" stackId="a" fill="#ef4444" name="Anomaly"  radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}

          {viewTab === 'temperature' && (
            <>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>Water Temperature Trend</h3>
              <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 16 }}>
                Temperature readings across all {fullDataset.total_readings} entries &mdash; Diurnal variation based on time of day
              </p>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={tempData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="index" tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'Sample Index', position: 'insideBottom', offset: -5, style: { fontSize: 11, fill: '#64748b' } }} />
                  <YAxis domain={[24, 32]} tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }}
                    formatter={(val: number) => [`${val.toFixed(2)} °C`, 'Temperature']}
                  />
                  <Line type="monotone" dataKey="temp" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Temperature" />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}

        </div>
      </div>

      {/* Statistical Summary */}
      <div className="glass-card">
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <TrendingUp size={16} style={{ color: 'var(--accent-text)' }} />
          Statistical Summary
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16 }}>
          {[
            { label: 'TDS Mean',           value: `${stats?.tds_mean ?? '—'} ppm`,  color: '#06b6d4' },
            { label: 'TDS Std Dev',         value: `${stats?.tds_std  ?? '—'} ppm`,  color: '#8b5cf6' },
            { label: 'TDS Min',             value: `${stats?.tds_min  ?? '—'} ppm`,  color: '#10b981' },
            { label: 'TDS Max',             value: `${stats?.tds_max  ?? '—'} ppm`,  color: '#ef4444' },
            { label: 'Temp Mean',           value: `${stats?.temp_mean ?? '—'} °C`,  color: '#f59e0b' },
            { label: 'Temp Range',          value: stats ? `${stats.temp_min}–${stats.temp_max} °C` : '—', color: '#f97316' },
            { label: 'Normal Count',        value: normalPreds.length.toString(),     color: '#06b6d4' },
            { label: 'Anomaly Count',       value: anomalyPreds.length.toString(),    color: '#ef4444' },
            { label: 'Avg Anomaly Score',   value: `${(anomalyPreds.reduce((s, p) => s + p.anomaly_score, 0) / (anomalyPreds.length || 1) * 100).toFixed(1)}%`, color: '#f59e0b' },
            { label: 'Avg Normal Score',    value: `${(normalPreds.reduce((s,  p) => s + p.anomaly_score, 0) / (normalPreds.length  || 1) * 100).toFixed(1)}%`, color: '#10b981' },
          ].map((item, i) => (
            <div key={i} style={{ padding: 16, background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>{item.label}</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: item.color }}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
