import React from 'react';
import { getMetricVals } from '../utils/aqiHelpers';
import { FALLBACK_METRICS } from '../constants/fallbackData';

function ModelsTab({
  activeTab,
  citiesList,
  selectedCityModels,
  setSelectedCityModels,
  metricsAll
}) {
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

  // Helper to render RMSE bars for Models page
  const renderRmseBarsList = (cityKey) => {
    const m = (metricsAll && metricsAll[cityKey]) ? metricsAll[cityKey].models : FALLBACK_METRICS[cityKey];
    if (!m) return null;
    const models = [
      { label: "ARIMA",   val: getMetricVals(m.ARIMA,   [0, 51.4])[1], color: "#378ADD" },
      { label: "SARIMAX", val: getMetricVals(m.SARIMAX, [0, 44.1])[1], color: "#85B7EB" },
      { label: "XGBoost", val: getMetricVals(m.XGBoost, [0, 27.3])[1], color: "#EF9F27" },
      { label: "LSTM",    val: getMetricVals(m.LSTM,    [0, 21.8])[1], color: "#7F77DD" },
      { label: "GRU",     val: getMetricVals(m.GRU,     [0, 19.4])[1], color: "#1D9E75" },
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
    <div className={`page-section ${activeTab === 'models' ? 'active' : ''}`}>
      <div className="main">
        <div className="card">
          <div className="card-title">Model Comparison — All Stations</div>
          <div className="section-label" style={{marginBottom:'12px'}}>Select city</div>
          <div className="models-city-tabs">
            {citiesList.map(c => (
              <div key={c.key} className={`models-city-tab ${selectedCityModels === c.key ? 'active' : ''}`} onClick={() => setSelectedCityModels(c.key)}>
                {c.label} ({c.station})
              </div>
            ))}
          </div>
          <table className="metrics-table">
            <thead>
              <tr><th>Model</th><th>Horizon</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE %</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {(() => {
                const m = (metricsAll && metricsAll[selectedCityModels]) ? metricsAll[selectedCityModels].models : FALLBACK_METRICS[selectedCityModels];
                if (!m) return null;
                const apiWinner = metricsAll?.[selectedCityModels]?.overall_winner;
                const rows = [
                  { model: "ARIMA",   badge: { bg: "#E6F1FB", fg: "#185FA5" }, horizon: "Next day",  vals: getMetricVals(m.ARIMA,   [38.2, 51.4, 0.71, 14.2]), note: "Classical AR+MA", winner: apiWinner ? apiWinner === "ARIMA" : false },
                  { model: "SARIMAX", badge: { bg: "#E6F1FB", fg: "#185FA5" }, horizon: "Next day",  vals: getMetricVals(m.SARIMAX, [32.6, 44.1, 0.78, 11.8]), note: "Seasonal + exogenous", winner: apiWinner ? apiWinner === "SARIMAX" : false },
                  { model: "XGBoost", badge: { bg: "#FAEEDA", fg: "#854F0B" }, horizon: "Next hour", vals: getMetricVals(m.XGBoost, [18.4, 27.3, 0.91, 7.3]), note: "Gradient boosted trees", winner: apiWinner ? apiWinner === "XGBoost" : false },
                  { model: "LSTM",    badge: { bg: "#EEEDFE", fg: "#534AB7" }, horizon: "Next hour", vals: getMetricVals(m.LSTM,    [14.1, 21.8, 0.94, 5.6]), note: "48h window, 2-layer", winner: apiWinner ? apiWinner === "LSTM" : false },
                  { model: "GRU ✦",  badge: { bg: "#C0DD97", fg: "#27500A" }, horizon: "Next hour", vals: getMetricVals(m.GRU,     [12.9, 19.4, 0.96, 4.9]), note: "Best overall model", winner: apiWinner ? apiWinner === "GRU" : true },
                ];
                return rows.map(r => (
                  <tr key={r.model} className={r.winner ? 'winner-row' : ''}>
                    <td><span className="model-badge" style={{background: r.badge.bg, color: r.badge.fg}}>{r.model}</span></td>
                    <td><span className="horizon-tag">{r.horizon}</span></td>
                    <td>{r.vals[0]}</td><td>{r.vals[1]}</td><td>{r.vals[2]}</td><td>{r.vals[3]}</td>
                    <td style={{fontSize:'11px',color:'var(--text-secondary)'}}>{r.note}</td>
                  </tr>
                ));
              })()}
            </tbody>
          </table>
        </div>

        <div className="two-col">
          <div className="card">
            <div className="card-title">MAE by model — selected city</div>
            <div>{renderMaeBarsList(selectedCityModels)}</div>
          </div>
          <div className="card">
            <div className="card-title">RMSE by model — selected city</div>
            <div>{renderRmseBarsList(selectedCityModels)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelsTab;
