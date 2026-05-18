import React from 'react';
import { AQI_CAT } from '../utils/aqiHelpers';

function LiveDemoTab({
  activeTab,
  citiesList,
  selectedCityDemo,
  setSelectedCityDemo,
  demoSample,
  demoLoading,
  fetchDemoSample
}) {
  return (
    <div className={`page-section ${activeTab === 'livedemo' ? 'active' : ''}`}>
      <div className="main">
        <div className="card">
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'14px',flexWrap:'wrap',gap:'10px'}}>
            <div className="card-title" style={{margin:0}}>Live prediction demo — {demoSample?.label || "Anand Vihar"}, {demoSample?.station || "Delhi"}</div>
            <div className="demo-controls">
              <span className="sample-info">Test sample #{demoSample?.sample_index !== undefined ? demoSample.sample_index + 1 : 1} of {demoSample?.total_samples ? demoSample.total_samples.toLocaleString() : "2,310"}</span>
              <button className="ctrl-btn" onClick={() => fetchDemoSample(selectedCityDemo, 'index', Math.max(0, (demoSample?.sample_index || 1) - 1))} disabled={demoLoading}>← prev</button>
              <button className="ctrl-btn" onClick={() => fetchDemoSample(selectedCityDemo, 'index', (demoSample?.sample_index || 0) + 1)} disabled={demoLoading}>next →</button>
              <button className="ctrl-btn primary" onClick={() => fetchDemoSample(selectedCityDemo, 'random')} disabled={demoLoading}>↻ random</button>
            </div>
          </div>

          <div className="actual-box">
            <div>
              <div style={{fontSize:'10px',color:'var(--text-secondary)',fontFamily:'sans-serif',marginBottom:'4px'}}>
                Actual AQI &nbsp;·&nbsp; {demoSample?.timestamp || "Nov 14, 2023 at 02:00 PM"}
              </div>
              <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
                <div className="actual-aqi" style={{color: demoSample?.actual_color || "#993C1D"}}>{demoSample?.actual_aqi || 241}</div>
                <span className="c-badge" style={{background: AQI_CAT(demoSample?.actual_aqi || 241).style.split(';')[0].split(':')[1], color: AQI_CAT(demoSample?.actual_aqi || 241).style.split(';')[1].split(':')[1], fontSize:'11px', padding:'4px 12px'}}>
                  {demoSample?.actual_category || "Poor"}
                </span>
              </div>
            </div>
            <div style={{fontSize:'11px',color:'var(--text-secondary)',textAlign:'right',maxWidth:'220px',lineHeight:'1.6',fontFamily:'sans-serif'}}>
              Model was given the preceding 48 hours of real sensor data to predict this timestep
            </div>
          </div>

          <div className="three-col">
            {demoSample?.predictions ? demoSample.predictions.map(p => {
              const isWinner = p.model === demoSample.best_model;
              return (
                <div key={p.model} className={`pred-card ${isWinner ? 'best' : ''}`}>
                  <div className="pred-model"><div className="pred-dot" style={{background: p.color}}></div>{p.model}</div>
                  <div className="pred-val" style={{color: p.color}}>{p.predicted}</div>
                  <div className="pred-err">Error: {typeof p.error === 'number' ? p.error.toFixed(1) : p.error} AQI &nbsp;({typeof p.pct_error === 'number' ? p.pct_error.toFixed(1) : p.pct_error}%)</div>
                  {isWinner && <div className="winner-chip">best this sample</div>}
                  <div className="pred-note">{p.advice}</div>
                </div>
              );
            }) : (
              <div className="loading-box" style={{gridColumn:'span 3'}}>Loading prediction models...</div>
            )}
          </div>

          <div className="info-box">
            <i className="ti ti-info-circle" style={{fontSize:'13px',marginTop:'1px',flexShrink:0}} aria-hidden="true"></i>
            ARIMA and SARIMAX are daily-horizon models (next-day prediction) and are not included in the hourly live demo. Their metrics appear in the comparison table above.
          </div>
        </div>

        {/* Demo city selector */}
        <div className="card" style={{padding:'14px 18px'}}>
          <div className="section-label" style={{marginBottom:'8px'}}>Station for demo</div>
          <div className="models-city-tabs">
            {citiesList.map(c => (
              <div key={c.key} className={`models-city-tab ${selectedCityDemo === c.key ? 'active' : ''}`} onClick={() => setSelectedCityDemo(c.key)}>
                {c.label} ({c.station})
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default LiveDemoTab;
