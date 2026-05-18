import React from 'react';

function AboutTab({ activeTab, citiesList }) {
  return (
    <div className={`page-section ${activeTab === 'about' ? 'active' : ''}`}>
      <div className="main">
        <div className="card">
          <div className="card-title">About this Dashboard</div>
          <div className="about-grid">
            <div className="about-section">
              <h3>Project Overview</h3>
              <p>This dashboard presents a comparative study of five forecasting models for predicting Air Quality Index (AQI) across six major Indian cities. All models were trained on CPCB sensor data and evaluated on a chronologically held-out test set (last 15%) to prevent data leakage.</p>
              <br/>
              <h3>Dataset</h3>
              <ul>
                <li>Source: Central Pollution Control Board (CPCB)</li>
                <li>Stations: 6 monitoring locations across India</li>
                <li>Features: PM2.5, PM10, NO₂, SO₂, CO, O₃, temperature, humidity</li>
                <li>Test split: Last 15% chronologically (no leakage)</li>
              </ul>
              <br/>
              <h3>Dataset Coverage by Station</h3>
              <ul style={{listStyleType:'none',paddingLeft:0,display:'flex',flexDirection:'column',gap:'6px',marginTop:'6px'}}>
                {citiesList.map(c => (
                  <li key={c.key} style={{fontSize:'13px'}}>
                    <strong>{c.label} ({c.station}):</strong> <span style={{fontFamily:'monospace',background:'var(--bg-secondary)',padding:'2px 6px',borderRadius:'4px',color:'var(--text-primary)'}}>{c.date_range || "2022-08-01 → 2023-06-01"}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="about-section">
              <h3>Models</h3>
              <div className="model-card-about">
                <h4><span className="model-badge" style={{background:'#E6F1FB',color:'#185FA5'}}>ARIMA</span></h4>
                <p>Classical time-series model. Captures autoregressive and moving average patterns. Used for next-day AQI prediction.</p>
              </div>
              <div className="model-card-about">
                <h4><span className="model-badge" style={{background:'#E6F1FB',color:'#185FA5'}}>SARIMAX</span></h4>
                <p>Seasonal ARIMA with exogenous variables. Incorporates meteorological features for improved daily forecasting.</p>
              </div>
              <div className="model-card-about">
                <h4><span className="model-badge" style={{background:'#FAEEDA',color:'#854F0B'}}>XGBoost</span></h4>
                <p>Gradient-boosted trees on tabular lag features and meteorological covariates. Strong hourly performance.</p>
              </div>
              <div className="model-card-about">
                <h4><span className="model-badge" style={{background:'#EEEDFE',color:'#534AB7'}}>LSTM</span>&nbsp;<span className="model-badge" style={{background:'#C0DD97',color:'#27500A'}}>GRU</span></h4>
                <p>Recurrent neural networks trained on 48-hour sliding windows. GRU achieves the best overall metrics across all cities and horizons.</p>
              </div>
            </div>
          </div>
          <div style={{marginTop:'16px',paddingTop:'14px',borderTop:'0.5px solid var(--border-light)',fontSize:'11px',color:'var(--text-tertiary)',fontFamily:'sans-serif'}}>
            All metrics (MAE, RMSE, R², MAPE) are computed exclusively on the unseen test set. Higher R² and lower MAE/RMSE/MAPE indicate better performance.
          </div>
        </div>
      </div>
    </div>
  );
}

export default AboutTab;
