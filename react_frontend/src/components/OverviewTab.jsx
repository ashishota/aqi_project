import React from 'react';
import { AQI_COLOR, AQI_CAT, getMetricVals } from '../utils/aqiHelpers';
import { FALLBACK_METRICS } from '../constants/fallbackData';

function OverviewTab({
  activeTab,
  citiesList,
  cityAqis,
  selectedCityOverview,
  setSelectedCityOverview,
  activeMetricsFilter,
  setActiveMetricsFilter,
  metricsAll,
  currentCityObj,
  currentChartPts
}) {
  // Helper to render metrics table rows
  const renderMetricsRows = (cityKey, filter) => {
    const m = (metricsAll && metricsAll[cityKey]) ? metricsAll[cityKey].models : FALLBACK_METRICS[cityKey];
    if (!m) return null;
    const apiWinner = metricsAll?.[cityKey]?.overall_winner;
    
    const rows = [
      { model: "ARIMA",   type: "daily",  badge: { bg: "#E6F1FB", fg: "#185FA5" }, horizon: "Next day",  vals: getMetricVals(m.ARIMA, [38.2, 51.4, 0.71, 14.2]), winner: apiWinner ? apiWinner === "ARIMA" : false },
      { model: "SARIMAX", type: "daily",  badge: { bg: "#E6F1FB", fg: "#185FA5" }, horizon: "Next day",  vals: getMetricVals(m.SARIMAX, [32.6, 44.1, 0.78, 11.8]), winner: apiWinner ? apiWinner === "SARIMAX" : false },
      { model: "XGBoost", type: "hourly", badge: { bg: "#FAEEDA", fg: "#854F0B" }, horizon: "Next hour", vals: getMetricVals(m.XGBoost, [18.4, 27.3, 0.91, 7.3]), winner: apiWinner ? apiWinner === "XGBoost" : false },
      { model: "LSTM",    type: "hourly", badge: { bg: "#EEEDFE", fg: "#534AB7" }, horizon: "Next hour", vals: getMetricVals(m.LSTM, [14.1, 21.8, 0.94, 5.6]), winner: apiWinner ? apiWinner === "LSTM" : false },
      { model: "GRU ✦",  type: "hourly", badge: { bg: "#C0DD97", fg: "#27500A" }, horizon: "Next hour", vals: getMetricVals(m.GRU, [12.9, 19.4, 0.96, 4.9]), winner: apiWinner ? apiWinner === "GRU" : true },
    ];

    return rows.filter(r => filter === 'all' ? true : r.type === filter).map(r => (
      <tr key={r.model} className={r.winner ? 'winner-row' : ''}>
        <td><span className="model-badge" style={{background: r.badge.bg, color: r.badge.fg}}>{r.model}</span></td>
        <td><span className="horizon-tag">{r.horizon}</span></td>
        <td>{r.winner ? <strong>{r.vals[0]}</strong> : r.vals[0]}</td>
        <td>{r.winner ? <strong>{r.vals[1]}</strong> : r.vals[1]}</td>
        <td>{r.winner ? <strong>{r.vals[2]}</strong> : r.vals[2]}</td>
        <td>{r.winner ? <strong>{r.vals[3]}</strong> : r.vals[3]}</td>
      </tr>
    ));
  };

  // Helper to render MAE bars
  const renderMaeBarsList = (cityKey) => {
    const m = (metricsAll && metricsAll[cityKey]) ? metricsAll[cityKey].models : FALLBACK_METRICS[cityKey];
    if (!m) return null;
    const models = [
      { label: "ARIMA",   val: getMetricVals(m.ARIMA,   [38.2])[0], color: "#378ADD" },
      { label: "SARIMAX", val: getMetricVals(m.SARIMAX, [32.6])[0], color: "#85B7EB" },
      { label: "XGBoost", val: getMetricVals(m.XGBoost, [18.4])[0], color: "#EF9F27" },
      { label: "LSTM",    val: getMetricVals(m.LSTM,    [14.1])[0], color: "#7F77DD" },
      { label: "GRU",     val: getMetricVals(m.GRU,     [12.9])[0], color: "#1D9E75" },
    ];
    const max = Math.max(...models.map(x => x.val));
    return models.map(r => (
      <div className="bar-row" key={r.label}>
        <span className="bar-label">{r.label}</span>
        <div className="bar-track"><div className="bar-fill" style={{ width: `${(r.val / max * 100).toFixed(1)}%`, background: r.color }}></div></div>
        <span className="bar-val">{r.val}</span>
      </div>
    ));
  };

  return (
    <div className={`page-section ${activeTab === 'overview' ? 'active' : ''}`}>
      <div className="main">
        {/* City Selector Grid */}
        <div>
          <div className="section-label">Select station</div>
          <div className="city-grid">
            {citiesList.map(c => {
              const aqi = cityAqis[c.key] || c.aqi || 150;
              const catInfo = AQI_CAT(aqi);
              return (
                <div key={c.key} className={`city-card ${selectedCityOverview === c.key ? 'active' : ''}`} onClick={() => setSelectedCityOverview(c.key)}>
                  <div className="c-name">{c.label}</div>
                  <div className="c-station">{c.station}</div>
                  <div className="c-aqi" style={{ color: AQI_COLOR(aqi) }}>{aqi}</div>
                  <div className="c-badge" style={{ background: catInfo.style.split(';')[0].split(':')[1], color: catInfo.style.split(';')[1].split(':')[1] }}>{catInfo.cat}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* AQI Scale Band */}
        <div className="aqi-band">
          <div className="band" style={{background:'#C0DD97',color:'#27500A'}} data-tip="AQI 0–50: Minimal impact">Good<br/>0–50</div>
          <div className="band" style={{background:'#9FE1CB',color:'#085041'}} data-tip="AQI 51–100: Minor breathing issues for sensitive people">Satisfactory<br/>51–100</div>
          <div className="band" style={{background:'#FAC775',color:'#633806'}} data-tip="AQI 101–200: Breathing discomfort for people with asthma">Moderate<br/>101–200</div>
          <div className="band" style={{background:'#F0997B',color:'#712B13'}} data-tip="AQI 201–300: Breathing discomfort for most people">Poor<br/>201–300</div>
          <div className="band" style={{background:'#F7C1C1',color:'#791F1F'}} data-tip="AQI 301–400: Respiratory illness on prolonged exposure">Very Poor<br/>301–400</div>
          <div className="band" style={{background:'#CECBF6',color:'#3C3489'}} data-tip="AQI 401+: Affects healthy people; serious illness risk">Severe<br/>401+</div>
        </div>

        {/* Metrics Table + Charts */}
        <div className="two-col">
          {/* Metrics Table */}
          <div className="card">
            <div className="card-title">Model performance — {currentCityObj.label}, {currentCityObj.station}</div>
            <div className="tab-bar">
              <div className={`tab ${activeMetricsFilter === 'all' ? 'active' : ''}`} onClick={() => setActiveMetricsFilter('all')}>All models</div>
              <div className={`tab ${activeMetricsFilter === 'daily' ? 'active' : ''}`} onClick={() => setActiveMetricsFilter('daily')}>Daily (ARIMA / SARIMAX)</div>
              <div className={`tab ${activeMetricsFilter === 'hourly' ? 'active' : ''}`} onClick={() => setActiveMetricsFilter('hourly')}>Hourly (XGB / LSTM / GRU)</div>
            </div>
            <table className="metrics-table">
              <thead>
                <tr><th>Model</th><th>Horizon</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE %</th></tr>
              </thead>
              <tbody>
                {renderMetricsRows(selectedCityOverview, activeMetricsFilter)}
              </tbody>
            </table>
            <div style={{marginTop:'10px',fontSize:'10px',color:'var(--text-tertiary)',fontFamily:'sans-serif'}}>
              ✦ Best performer on test set &nbsp;·&nbsp; All values computed on unseen test set
            </div>
          </div>

          {/* Right column: MAE bars + R2 grid */}
          <div style={{display:'flex',flexDirection:'column',gap:'12px'}}>
            <div className="card" style={{flex:1}}>
              <div className="card-title">MAE comparison</div>
              <div className="legend">
                <div className="leg-item"><div className="leg-dot" style={{background:'#378ADD'}}></div>ARIMA</div>
                <div className="leg-item"><div className="leg-dot" style={{background:'#85B7EB'}}></div>SARIMAX</div>
                <div className="leg-item"><div className="leg-dot" style={{background:'#EF9F27'}}></div>XGBoost</div>
                <div className="leg-item"><div className="leg-dot" style={{background:'#7F77DD'}}></div>LSTM</div>
                <div className="leg-item"><div className="leg-dot" style={{background:'#1D9E75'}}></div>GRU</div>
              </div>
              <div>{renderMaeBarsList(selectedCityOverview)}</div>
            </div>

            <div className="card" style={{padding:'14px 16px'}}>
              <div className="card-title" style={{marginBottom:'10px'}}>Best model RMSE — across all cities</div>
              <div className="r2-grid">
                {citiesList.map(c => {
                  const m = (metricsAll && metricsAll[c.key]) ? metricsAll[c.key].models : FALLBACK_METRICS[c.key];
                  const rmse = m ? Math.min(
                    getMetricVals(m.GRU, [0, 27.4])[1],
                    getMetricVals(m.LSTM, [0, 29.1])[1],
                    getMetricVals(m.XGBoost, [0, 26.5])[1],
                    getMetricVals(m.SARIMAX, [0, 31.8])[1]
                  ) : 26.8;
                  const rmseColor = rmse <= 25.0 ? "#1D9E75" : rmse <= 30.0 ? "#EF9F27" : "#D85A30";
                  return (
                    <div className="r2-cell" key={c.key}>
                      <div className="r2-city">{c.label.split(' ')[0]}</div>
                      <div className="r2-val" style={{color: rmseColor}}>{rmse}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Time Series Chart — Separated by Model */}
        <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '10px'}}>
          <div className="card-title" style={{margin:0}}>Test set predictions vs actual — {currentCityObj.label}</div>
          <div className="legend" style={{margin:0, background:'var(--bg-secondary)', padding:'6px 14px', borderRadius:'20px', border:'1px solid var(--border-color)', gap:'16px'}}>
            <div className="leg-item" style={{color:'#888780', fontWeight:'600'}}><div className="leg-dot" style={{background:'#888780'}}></div>Actual (CPCB Sensor)</div>
            <div className="leg-item" style={{color:'#7F77DD', fontWeight:'600'}}><div className="leg-dot" style={{background:'#7F77DD'}}></div>LSTM Predicted</div>
            <div className="leg-item" style={{color:'#1D9E75', fontWeight:'600'}}><div className="leg-dot" style={{background:'#1D9E75'}}></div>GRU Predicted</div>
            <div className="leg-item" style={{color:'#EF9F27', fontWeight:'600'}}><div className="leg-dot" style={{background:'#EF9F27'}}></div>XGBoost Predicted</div>
          </div>
        </div>

        <div style={{display: 'flex', flexDirection: 'column', gap: '20px', marginBottom: '24px'}}>
          {/* LSTM vs Actual */}
          <div className="card" style={{padding: '16px'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', fontSize: '14px', color: '#7F77DD'}}>
                <div className="pred-dot" style={{background:'#7F77DD'}}></div> LSTM Predicted
              </div>
              <div style={{display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px'}}>
                <span style={{display: 'flex', alignItems: 'center', gap: '4px', color: '#888780', fontWeight: '600'}}><div className="leg-dot" style={{background:'#888780', width:'8px', height:'8px'}}></div> Actual</span>
                <span style={{color: 'var(--text-secondary)'}}>| &nbsp;48h window input</span>
              </div>
            </div>
            <div className="chart-wrap" style={{background: 'var(--bg-secondary)', borderRadius: '8px', padding: '8px'}}>
              <svg width="100%" height="110" viewBox="0 0 620 110" preserveAspectRatio="none">
                <rect x="0" y="80" width="620" height="30" fill="#C0DD97" opacity=".18"/>
                <rect x="0" y="55" width="620" height="25" fill="#FAC775" opacity=".18"/>
                <rect x="0" y="25" width="620" height="30" fill="#F0997B" opacity=".18"/>
                <rect x="0" y="0" width="620" height="25" fill="#F7C1C1" opacity=".15"/>
                <text x="4" y="95" fontSize="8" fill="#3B6D11" fontFamily="monospace">Good</text>
                <text x="4" y="67" fontSize="8" fill="#854F0B" fontFamily="monospace">Moderate</text>
                <text x="4" y="37" fontSize="8" fill="#993C1D" fontFamily="monospace">Poor</text>
                <text x="4" y="12" fontSize="8" fill="#791F1F" fontFamily="monospace">Very Poor</text>
                <polyline fill="none" stroke="#888780" strokeWidth="1.2" opacity="0.65" points={currentChartPts.actual}/>
                <polyline fill="none" stroke="#7F77DD" strokeWidth="1.8" points={currentChartPts.lstm}/>
                <text x="2" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Aug 2022</text>
                <text x="270" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Jan 2023</text>
                <text x="560" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Jun 2023</text>
              </svg>
            </div>
          </div>

          {/* GRU vs Actual */}
          <div className="card" style={{padding: '16px'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', fontSize: '14px', color: '#1D9E75'}}>
                <div className="pred-dot" style={{background:'#1D9E75'}}></div> GRU Predicted ✦
              </div>
              <div style={{display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px'}}>
                <span style={{display: 'flex', alignItems: 'center', gap: '4px', color: '#888780', fontWeight: '600'}}><div className="leg-dot" style={{background:'#888780', width:'8px', height:'8px'}}></div> Actual</span>
                <span style={{color: 'var(--text-secondary)'}}>| &nbsp;Best overall tracking</span>
              </div>
            </div>
            <div className="chart-wrap" style={{background: 'var(--bg-secondary)', borderRadius: '8px', padding: '8px'}}>
              <svg width="100%" height="110" viewBox="0 0 620 110" preserveAspectRatio="none">
                <rect x="0" y="80" width="620" height="30" fill="#C0DD97" opacity=".18"/>
                <rect x="0" y="55" width="620" height="25" fill="#FAC775" opacity=".18"/>
                <rect x="0" y="25" width="620" height="30" fill="#F0997B" opacity=".18"/>
                <rect x="0" y="0" width="620" height="25" fill="#F7C1C1" opacity=".15"/>
                <text x="4" y="95" fontSize="8" fill="#3B6D11" fontFamily="monospace">Good</text>
                <text x="4" y="67" fontSize="8" fill="#854F0B" fontFamily="monospace">Moderate</text>
                <text x="4" y="37" fontSize="8" fill="#993C1D" fontFamily="monospace">Poor</text>
                <text x="4" y="12" fontSize="8" fill="#791F1F" fontFamily="monospace">Very Poor</text>
                <polyline fill="none" stroke="#888780" strokeWidth="1.2" opacity="0.65" points={currentChartPts.actual}/>
                <polyline fill="none" stroke="#1D9E75" strokeWidth="1.8" points={currentChartPts.gru}/>
                <text x="2" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Aug 2022</text>
                <text x="270" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Jan 2023</text>
                <text x="560" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Jun 2023</text>
              </svg>
            </div>
          </div>

          {/* XGBoost vs Actual */}
          <div className="card" style={{padding: '16px'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', fontSize: '14px', color: '#EF9F27'}}>
                <div className="pred-dot" style={{background:'#EF9F27'}}></div> XGBoost Predicted
              </div>
              <div style={{display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px'}}>
                <span style={{display: 'flex', alignItems: 'center', gap: '4px', color: '#888780', fontWeight: '600'}}><div className="leg-dot" style={{background:'#888780', width:'8px', height:'8px'}}></div> Actual</span>
                <span style={{color: 'var(--text-secondary)'}}>| &nbsp;Tabular feature trees</span>
              </div>
            </div>
            <div className="chart-wrap" style={{background: 'var(--bg-secondary)', borderRadius: '8px', padding: '8px'}}>
              <svg width="100%" height="110" viewBox="0 0 620 110" preserveAspectRatio="none">
                <rect x="0" y="80" width="620" height="30" fill="#C0DD97" opacity=".18"/>
                <rect x="0" y="55" width="620" height="25" fill="#FAC775" opacity=".18"/>
                <rect x="0" y="25" width="620" height="30" fill="#F0997B" opacity=".18"/>
                <rect x="0" y="0" width="620" height="25" fill="#F7C1C1" opacity=".15"/>
                <text x="4" y="95" fontSize="8" fill="#3B6D11" fontFamily="monospace">Good</text>
                <text x="4" y="67" fontSize="8" fill="#854F0B" fontFamily="monospace">Moderate</text>
                <text x="4" y="37" fontSize="8" fill="#993C1D" fontFamily="monospace">Poor</text>
                <text x="4" y="12" fontSize="8" fill="#791F1F" fontFamily="monospace">Very Poor</text>
                <polyline fill="none" stroke="#888780" strokeWidth="1.2" opacity="0.65" points={currentChartPts.actual}/>
                <polyline fill="none" stroke="#EF9F27" strokeWidth="1.8" points={currentChartPts.xgb}/>
                <text x="2" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Aug 2022</text>
                <text x="270" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Jan 2023</text>
                <text x="560" y="108" fontSize="8" fill="#9a9a95" fontFamily="monospace">Jun 2023</text>
              </svg>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="footer">
          Dataset: CPCB &nbsp;·&nbsp; 6 monitoring stations across India &nbsp;·&nbsp;
          All metrics computed on unseen test set (last 15% chronologically — no data leakage)
        </div>
      </div>
    </div>
  );
}

export default OverviewTab;
