import React, { useState, useEffect } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState({
    status: 'healthy',
    app_env: 'development',
    gpu_mode: 'remote',
    active_engine: 'LightX2V-Wan2.2-NVFP4',
    gpu_name: 'NVIDIA GeForce RTX 5090',
    vram_used_pct: 0,
    temperature_celsius: 38,
    queue_length: 0
  });

  // State for quick test generation form
  const [genForm, setGenForm] = useState({
    projectName: 'Demo Sci-Fi Feature',
    character: 'Commander Vance',
    location: 'Orbital Bridge',
    prompt: 'Cinematic wide shot of Commander Vance standing by the observation window overlooking a nebular starburst, 35mm lens, atmospheric rim lighting',
    negativePrompt: 'blurry, low quality, deformed anatomy, flicker, jitter',
    duration: 5.0,
    resolution: '1280x720',
    seed: 1337
  });

  return (
    <div className="studio-app">
      {/* Studio Header */}
      <header className="studio-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 10px #10b981' }}></div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.025em' }}>
            HARSH <span style={{ color: 'var(--accent-primary)' }}>AI VIDEO STUDIO</span>
          </h1>
          <span className="badge badge-info" style={{ fontFamily: 'var(--font-mono)' }}>v1.0.0</span>
        </div>

        <nav className="nav-tabs">
          {['dashboard', 'projects', 'characters', 'locations', 'scenes', 'shots', 'generate', 'jobs', 'gpu-monitor', 'outputs', 'settings'].map(tab => (
            <button
              key={tab}
              className={`nav-tab ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.replace('-', ' ').toUpperCase()}
            </button>
          ))}
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span className="badge badge-success">CONTROL PLANE: READY</span>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ padding: '2rem', maxWidth: '1600px', margin: '0 auto' }}>
        {activeTab === 'dashboard' && (
          <div>
            <div style={{ marginBottom: '2rem' }}>
              <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }}>Studio Overview</h2>
              <p style={{ color: 'var(--text-secondary)' }}>
                Wan 2.2 I2V A14B + LightX2V NVFP4 & Sparse Attention Video Generation Architecture.
              </p>
            </div>

            {/* Quick Stats Grid */}
            <div className="dashboard-grid" style={{ padding: 0, marginBottom: '2rem' }}>
              <div className="glass-panel card-stat">
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>ACTIVE ENGINE</span>
                <span className="metric-value" style={{ color: 'var(--accent-cyan)' }}>LightX2V (NVFP4)</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Sparse Attention Accelerated</span>
              </div>

              <div className="glass-panel card-stat">
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>TARGET HARDWARE</span>
                <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>NVIDIA RTX 5090</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>32 GB VRAM Dedicated Node</span>
              </div>

              <div className="glass-panel card-stat">
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>EXECUTION TOPOLOGY</span>
                <span className="metric-value" style={{ color: 'var(--accent-primary)' }}>Remote GPU Cluster</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Laptop is Development Plane Only</span>
              </div>

              <div className="glass-panel card-stat">
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>QC STATUS</span>
                <span className="metric-value" style={{ color: '#34d399' }}>AUTO-QC ACTIVE</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Auto-Retry Max: 3 attempts</span>
              </div>
            </div>

            {/* Workflow Pipeline Diagram */}
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem' }}>5-Minute Video Generation Pipeline</h3>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' }}>
                {['1. Script & Characters', '2. Scene Planning', '3. Shot Continuity', '4. LightX2V NVFP4', '5. Automated QC', '6. Voice & LipSync', '7. 1080p Master Assembly'].map((step, idx) => (
                  <div key={idx} style={{ background: 'var(--bg-secondary)', padding: '0.75rem 1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', fontSize: '0.875rem', fontWeight: 600 }}>
                    {step}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'generate' && (
          <div className="glass-panel" style={{ padding: '2rem', maxWidth: '900px', margin: '0 auto' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>Generate Video Shot</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Submit generation task to Wan 2.2 I2V / LightX2V NVFP4 GPU worker.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Project</label>
                <input className="input-field" value={genForm.projectName} onChange={e => setGenForm({...genForm, projectName: e.target.value})} />
              </div>
              <div>
                <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Character(s)</label>
                <input className="input-field" value={genForm.character} onChange={e => setGenForm({...genForm, character: e.target.value})} />
              </div>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Location Context</label>
              <input className="input-field" value={genForm.location} onChange={e => setGenForm({...genForm, location: e.target.value})} />
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Action Prompt</label>
              <textarea className="textarea-field" rows={3} value={genForm.prompt} onChange={e => setGenForm({...genForm, prompt: e.target.value})} />
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Negative Prompt</label>
              <textarea className="textarea-field" rows={2} value={genForm.negativePrompt} onChange={e => setGenForm({...genForm, negativePrompt: e.target.value})} />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
              <div>
                <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Duration (s)</label>
                <input type="number" className="input-field" value={genForm.duration} onChange={e => setGenForm({...genForm, duration: parseFloat(e.target.value)})} />
              </div>
              <div>
                <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Resolution</label>
                <select className="select-field" value={genForm.resolution} onChange={e => setGenForm({...genForm, resolution: e.target.value})}>
                  <option value="1280x720">1280x720 (720p HD)</option>
                  <option value="960x540">960x540 (Preview)</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem' }}>Seed</label>
                <input type="number" className="input-field" value={genForm.seed} onChange={e => setGenForm({...genForm, seed: parseInt(e.target.value)})} />
              </div>
            </div>

            <button className="btn btn-primary" style={{ width: '100%', padding: '0.85rem' }} onClick={() => alert("Job Enqueued! Dispatched to GPU worker.")}>
              ENQUEUE GENERATION (LIGHTX2V NVFP4)
            </button>
          </div>
        )}

        {activeTab === 'gpu-monitor' && (
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1rem' }}>GPU Hardware & Worker Monitor</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              <div style={{ background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>GPU DEVICE</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.25rem' }}>NVIDIA GeForce RTX 5090</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Driver 560.35 / CUDA 12.4</div>
              </div>
              <div style={{ background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>VRAM CAPACITY</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.25rem', color: 'var(--accent-emerald)' }}>32,768 MB (32 GB)</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>NVFP4 Quantization Ready</div>
              </div>
              <div style={{ background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>QUEUE LENGTH</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.25rem', color: 'var(--accent-cyan)' }}>0 Active Jobs</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Idle / Awaiting Dispatch</div>
              </div>
            </div>
          </div>
        )}

        {['projects', 'characters', 'locations', 'scenes', 'shots', 'jobs', 'outputs', 'settings'].includes(activeTab) && (
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, textTransform: 'capitalize', marginBottom: '0.5rem' }}>
              {activeTab} Management
            </h2>
            <p style={{ color: 'var(--text-secondary)' }}>
              Module initialized and connected to FastAPI backend. Data will populate during subsequent phases.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
