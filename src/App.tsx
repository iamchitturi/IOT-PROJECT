import { useState } from 'react';
import { LayoutDashboard, BarChart3, ClipboardList, Droplets } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import RecentData from './pages/RecentData';

/* ═══════════════════════════════════════
   AQUAQUALITY — Main App Shell
   IoT Water Quality Monitoring System
   ═══════════════════════════════════════ */

type Tab = 'dashboard' | 'analytics' | 'recent';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');

  const navItems: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} /> },
    { id: 'analytics', label: 'Data Analytics', icon: <BarChart3 size={18} /> },
    { id: 'recent', label: 'Recent Data', icon: <ClipboardList size={18} /> },
  ];

  return (
    <div className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <h1>
            <Droplets size={22} />
            AquaQuality
          </h1>
          <p>IoT Water Monitoring System</p>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div style={{ marginBottom: 6 }}>
            <span className="status-dot" />
            System Active
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span>ML: Isolation Forest (Python)</span>
            <span>Data: ThingSpeak HTTP/JSON</span>
          </div>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="app-main">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'analytics' && <Analytics />}
        {activeTab === 'recent' && <RecentData />}
      </main>
    </div>
  );
}
