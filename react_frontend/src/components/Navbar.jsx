import React from 'react';

function Navbar({ activeTab, setActiveTab }) {
  return (
    <nav className="nav">
      <div className="nav-brand">
        <div className="nav-icon"><i className="ti ti-wind" aria-hidden="true"></i></div>
        <div>
          <div className="nav-title">AQI Forecasting — Multi-City Dashboard</div>
          <div className="nav-sub">Comparative analysis · ARIMA · SARIMAX · XGBoost · LSTM · GRU</div>
        </div>
      </div>
      <div className="nav-pills">
        <div className={`pill ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>Overview</div>
        <div className={`pill ${activeTab === 'models' ? 'active' : ''}`} onClick={() => setActiveTab('models')}>Models</div>
        <div className={`pill ${activeTab === 'livedemo' ? 'active' : ''}`} onClick={() => setActiveTab('livedemo')}>Live Demo</div>
        <div className={`pill ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>History</div>
        <div className={`pill ${activeTab === 'about' ? 'active' : ''}`} onClick={() => setActiveTab('about')}>About</div>
      </div>
    </nav>
  );
}

export default Navbar;
