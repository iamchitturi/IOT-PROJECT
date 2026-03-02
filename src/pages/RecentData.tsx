import { useState, useEffect } from 'react';
import {
  Activity, Search, ArrowUpDown, AlertTriangle,
  CheckCircle2, Download, RefreshCw, Thermometer,
} from 'lucide-react';
import {
  fetchLiveReadings, fetchPredictions,
  classifyWaterQuality, formatTimestamp,
  type SensorReading, type PredictResponse,
} from '../services/api';

/* ═══════════════════════════════════════
   RECENT DATA PAGE
   Tabular view of sensor readings with
   temperature, TDS, voltage, and
   Isolation Forest anomaly status
   ═══════════════════════════════════════ */

export default function RecentData() {
  const [liveData,    setLiveData]    = useState<SensorReading[]>([]);
  const [predictions, setPredictions] = useState<PredictResponse | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField,   setSortField]   = useState<'entryId' | 'tdsValue' | 'timestamp' | 'temperature'>('entryId');
  const [sortDir,     setSortDir]     = useState<'asc' | 'desc'>('desc');
  const [filterType,  setFilterType]  = useState<'all' | 'anomaly' | 'normal'>('all');

  const fetchAll = async () => {
    const [live, preds] = await Promise.all([
      fetchLiveReadings(100),
      fetchPredictions().catch(() => null),
    ]);
    setLiveData(live);
    setPredictions(preds);
  };

  useEffect(() => {
    (async () => {
      setLoading(true);
      try { await fetchAll(); } catch (err) { console.error(err); }
      finally { setLoading(false); }
    })();
  }, []);

  const refresh = async () => {
    setLoading(true);
    try { await fetchAll(); } finally { setLoading(false); }
  };

  // Build anomaly lookup from predictions
  const anomalyMap = new Map<number, { isAnomaly: boolean; score: number; temperature: number }>();
  if (predictions?.predictions) {
    predictions.predictions.forEach(p => {
      anomalyMap.set(p.entry_id, {
        isAnomaly:   p.is_anomaly,
        score:       p.anomaly_score,
        temperature: p.temperature,
      });
    });
  }

  // Filter + sort
  let filtered = [...liveData];

  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    filtered = filtered.filter(r =>
      r.tdsValue.toString().includes(q) ||
      r.entryId.toString().includes(q)  ||
      r.timestamp.toLowerCase().includes(q) ||
      r.temperature.toString().includes(q)
    );
  }

  if (filterType === 'anomaly') filtered = filtered.filter(r =>  anomalyMap.get(r.entryId)?.isAnomaly);
  if (filterType === 'normal')  filtered = filtered.filter(r => !anomalyMap.get(r.entryId)?.isAnomaly);

  filtered.sort((a, b) => {
    let diff = 0;
    if      (sortField === 'entryId')     diff = a.entryId   - b.entryId;
    else if (sortField === 'tdsValue')    diff = a.tdsValue  - b.tdsValue;
    else if (sortField === 'temperature') diff = a.temperature - b.temperature;
    else diff = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    return sortDir === 'asc' ? diff : -diff;
  });

  const toggleSort = (field: typeof sortField) => {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('desc'); }
  };

  // Export CSV
  const exportCSV = () => {
    const headers = 'Entry ID,Timestamp,TDS (ppm),Voltage (V),Temperature (°C),Quality,Anomaly,Score\n';
    const rows = filtered.map(r => {
      const q    = classifyWaterQuality(r.tdsValue);
      const anom = anomalyMap.get(r.entryId);
      return `${r.entryId},${r.timestamp},${r.tdsValue.toFixed(2)},${r.voltage.toFixed(4)},${r.temperature.toFixed(2)},${q.label},${anom?.isAnomaly ? 'Yes' : 'No'},${anom?.score ? (anom.score * 100).toFixed(1) : 'N/A'}`;
    }).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'aquaquality_data.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  if (loading && liveData.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: 16 }}>
        <Activity className="text-accent" size={40} style={{ animation: 'pulse-glow 1.5s infinite' }} />
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>Loading recent data…</p>
      </div>
    );
  }

  const anomalyCount = liveData.filter(r => anomalyMap.get(r.entryId)?.isAnomaly).length;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Page Header */}
      <div className="page-header">
        <h1>📋 Recent Data</h1>
        <p>Latest {liveData.length} live readings from ThingSpeak with Isolation Forest anomaly classification</p>
      </div>

      {/* Controls Bar */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        {/* Search */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '8px 14px', background: 'var(--bg-glass)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)', flex: '1 1 240px', maxWidth: 360,
        }}>
          <Search size={16} style={{ color: 'var(--text-dim)' }} />
          <input
            type="text"
            placeholder="Search by ID, TDS, temp, or timestamp…"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            style={{
              background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text-primary)', fontSize: 13, width: '100%',
              fontFamily: 'inherit',
            }}
          />
        </div>

        {/* Filter */}
        <div className="custom-tabs">
          <button className={`custom-tab ${filterType === 'all'     ? 'active' : ''}`} onClick={() => setFilterType('all')}>
            All ({liveData.length})
          </button>
          <button className={`custom-tab ${filterType === 'anomaly' ? 'active' : ''}`} onClick={() => setFilterType('anomaly')}>
            Anomalies ({anomalyCount})
          </button>
          <button className={`custom-tab ${filterType === 'normal'  ? 'active' : ''}`} onClick={() => setFilterType('normal')}>
            Normal ({liveData.length - anomalyCount})
          </button>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="btn" onClick={refresh}>
            <RefreshCw size={14} /> Refresh
          </button>
          <button className="btn primary" onClick={exportCSV}>
            <Download size={14} /> Export CSV
          </button>
        </div>
      </div>

      {/* Data Table */}
      <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto', maxHeight: '65vh', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => toggleSort('entryId')} style={{ cursor: 'pointer' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}># ID <ArrowUpDown size={12} /></span>
                </th>
                <th onClick={() => toggleSort('timestamp')} style={{ cursor: 'pointer' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>Timestamp <ArrowUpDown size={12} /></span>
                </th>
                <th onClick={() => toggleSort('tdsValue')} style={{ cursor: 'pointer' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>TDS (ppm) <ArrowUpDown size={12} /></span>
                </th>
                <th>Voltage (V)</th>
                <th onClick={() => toggleSort('temperature')} style={{ cursor: 'pointer' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Thermometer size={12} /> Temp (°C) <ArrowUpDown size={12} /></span>
                </th>
                <th>Quality</th>
                <th>Anomaly</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(reading => {
                const quality  = classifyWaterQuality(reading.tdsValue);
                const anomInfo = anomalyMap.get(reading.entryId);
                const isAnom   = anomInfo?.isAnomaly || false;
                // Fall back to reading's own temperature if ML API is offline
                const temp     = anomInfo?.temperature ?? reading.temperature;

                return (
                  <tr key={reading.entryId} className={isAnom ? 'anomaly-row' : ''}>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {reading.entryId}
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {formatTimestamp(reading.timestamp)}
                    </td>
                    <td style={{ fontWeight: 600, color: quality.color }}>
                      {reading.tdsValue.toFixed(2)}
                    </td>
                    <td>{reading.voltage.toFixed(4)}</td>
                    <td style={{ color: '#f59e0b', fontWeight: 500 }}>
                      {temp.toFixed(2)}
                    </td>
                    <td>
                      <span className={`badge ${
                        quality.label === 'Excellent' || quality.label === 'Good' ? 'green' :
                        quality.label === 'Fair' ? 'orange' : 'red'
                      }`}>
                        {quality.label}
                      </span>
                    </td>
                    <td>
                      {isAnom ? (
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#fca5a5' }}>
                          <AlertTriangle size={14} /> Yes
                        </span>
                      ) : (
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#6ee7b7' }}>
                          <CheckCircle2 size={14} /> No
                        </span>
                      )}
                    </td>
                    <td>
                      {anomInfo ? (
                        <span style={{
                          color:      isAnom ? '#fca5a5' : 'var(--text-muted)',
                          fontWeight: isAnom ? 600 : 400,
                        }}>
                          {(anomInfo.score * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-dim)' }}>—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
            <p style={{ fontSize: 14 }}>No readings match your filters</p>
          </div>
        )}
      </div>

      {/* Summary Footer */}
      <div style={{ fontSize: 12, color: 'var(--text-dim)', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <span>Showing {filtered.length} of {liveData.length} readings</span>
        <span>•</span>
        <span>Source: ThingSpeak Channel {`3286342`}</span>
        <span>•</span>
        <span>Protocol: HTTP GET / JSON</span>
        {predictions && <>
          <span>•</span>
          <span><Thermometer size={11} style={{ display: 'inline', verticalAlign: 'middle' }} /> Temp: Diurnal variation (26–33 °C)</span>
          <span>•</span>
          <span>ML: Isolation Forest ({predictions.anomaly_rate}% anomaly rate)</span>
        </>}
      </div>
    </div>
  );
}
