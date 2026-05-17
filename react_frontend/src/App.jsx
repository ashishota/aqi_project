import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000'

const getHealthRecommendation = (category) => {
  switch(category) {
    case 'Good': return { class: 'badge-good', text: 'Good' };
    case 'Satisfactory': return { class: 'badge-sat', text: 'Satisfactory' };
    case 'Moderate': return { class: 'badge-mod', text: 'Moderate' };
    case 'Poor': return { class: 'badge-poor', text: 'Poor' };
    case 'Very Poor': return { class: 'badge-vp', text: 'Very Poor' };
    case 'Severe': return { class: 'badge-vp', text: 'Severe' };
    default: return { class: 'badge-mod', text: category };
  }
}

function App() {
  const [cities, setCities] = useState([])
  const [formData, setFormData] = useState({
    city: '',
    PM25: 10.0,
    PM10: 10.0,
    NO2: 10.0,
    SO2: 10.0,
    NH3: 10.0,
    CO: 10.0,
    O3: 10.0,
    AT: 25.0,
    RH: 50.0,
    WS: 1.0,
    WD: 180.0,
    SR: 100.0,
    BP: 1000.0
  })
  
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [historyData, setHistoryData] = useState([])

  const WEATHER_PARAMS = [
    { id: 'AT', label: 'Temp (°C)' },
    { id: 'RH', label: 'Humidity (%)' },
    { id: 'WS', label: 'Wind (m/s)' },
    { id: 'WD', label: 'Wind Dir (°)' },
    { id: 'SR', label: 'Solar Rad' },
    { id: 'BP', label: 'Pressure (hPa)' }
  ];

  const fetchHistory = async (city) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/history?city=${city}`)
      setHistoryData(response.data.history)
    } catch (err) {
      console.error("Error fetching history:", err)
    }
  }

  useEffect(() => {
    axios.get(`${API_BASE_URL}/api/cities`)
      .then(response => {
        if (response.data.cities && response.data.cities.length > 0) {
          setCities(response.data.cities)
          const defaultCity = response.data.cities[0]
          setFormData(prev => ({ ...prev, city: defaultCity }))
          fetchHistory(defaultCity)
        }
      })
      .catch(err => {
        console.error("Error fetching cities:", err)
        setError("Could not load cities. Ensure backend is running.")
      })
  }, [])

  const handleChange = (e) => {
    const { name, value } = e.target
    const newValue = name === 'city' ? value : parseFloat(value)
    setFormData(prev => ({ ...prev, [name]: newValue }))
    if (name === 'city') {
      fetchHistory(value)
    }
  }

  const handleAutoFill = () => {
    if (historyData.length > 0) {
      const randomIdx = Math.floor(Math.random() * historyData.length);
      const sample = historyData[randomIdx];
      
      setFormData(prev => ({
        ...prev,
        PM25: sample.PM25 || prev.PM25,
        PM10: sample.PM10 || prev.PM10,
        NO2: sample.NO2 || prev.NO2,
        SO2: sample.SO2 || prev.SO2,
        NH3: sample.NH3 || prev.NH3,
        CO: sample.CO || prev.CO,
        O3: sample.O3 || prev.O3,
        AT: sample.AT || prev.AT,
        RH: sample.RH || prev.RH,
        WS: sample.WS || prev.WS,
        WD: sample.WD || prev.WD,
        SR: sample.SR || prev.SR,
        BP: sample.BP || prev.BP
      }));
    }
  }

  const getPrimaryPollutant = () => {
    const pollutants = ['PM25', 'PM10', 'NO2', 'SO2', 'NH3', 'O3'];
    return pollutants.reduce((max, p) => formData[p] > formData[max] ? p : max, 'PM25');
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict`, formData)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred during prediction.")
    } finally {
      setLoading(false)
    }
  }

  // Derive current AQI from history if available
  let currentAqi = 187;
  let currentCat = "Moderate";
  if (historyData.length > 0) {
    const latest = historyData[historyData.length - 1];
    if (latest.AQI) {
        currentAqi = Math.round(latest.AQI);
        if (currentAqi <= 50) currentCat = "Good";
        else if (currentAqi <= 100) currentCat = "Satisfactory";
        else if (currentAqi <= 200) currentCat = "Moderate";
        else if (currentAqi <= 300) currentCat = "Poor";
        else if (currentAqi <= 400) currentCat = "Very Poor";
        else currentCat = "Severe";
    }
  }
  const currentHealth = getHealthRecommendation(currentCat);

  return (
    <div className="app">
      <nav className="sidebar" role="navigation" aria-label="Main navigation">
        <div className="logo">
          <div className="logo-text"><i className="ti ti-wind" aria-hidden="true"></i> AQI Forecast</div>
          <div className="logo-sub">Research dashboard</div>
        </div>
        <div className="nav-section">Forecast</div>
        <div className="nav-item active"><i className="ti ti-chart-line" aria-hidden="true"></i> Overview</div>
        <div className="nav-item"><i className="ti ti-clock-hour-4" aria-hidden="true"></i> Hourly (XGBoost)</div>
        <div className="nav-item"><i className="ti ti-clock-hour-4" aria-hidden="true"></i> Hourly (LSTM / GRU)</div>
        <div className="nav-item"><i className="ti ti-calendar" aria-hidden="true"></i> Daily (SARIMAX)</div>
        <div className="nav-section">Analysis</div>
        <div className="nav-item"><i className="ti ti-map-pin" aria-hidden="true"></i> Stations</div>
        <div className="nav-item"><i className="ti ti-chart-bar" aria-hidden="true"></i> Model comparison</div>
        <div className="nav-item"><i className="ti ti-settings" aria-hidden="true"></i> Settings</div>
      </nav>

      <main className="main">
        <div className="topbar">
          <div>
            <div className="page-title">Overview</div>
            <div className="page-sub">Last updated: just now</div>
          </div>
          <div className="station-pill">
            <i className="ti ti-map-pin"></i> 
            <select 
              value={formData.city} 
              onChange={handleChange} 
              name="city"
              style={{background: 'transparent', color: 'inherit', border: 'none', outline: 'none', cursor: 'pointer', appearance: 'none', paddingRight: '15px'}}
            >
              {cities.map(city => <option key={city} value={city} style={{background: 'var(--color-background-secondary)'}}>{city}</option>)}
            </select>
            <i className="ti ti-chevron-down" style={{marginLeft: '-15px', pointerEvents: 'none'}}></i>
          </div>
        </div>

        {error && <div style={{padding: '10px', background: 'rgba(255,0,0,0.1)', color: '#ff5555', borderRadius: '6px', marginBottom: '15px'}}>{error}</div>}

        <div className="metrics">
          <div className="metric">
            <div className="metric-label">Current AQI</div>
            <div className="metric-value">{currentAqi}</div>
            <div className={`metric-badge ${currentHealth.class}`}>{currentHealth.text}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Next hour (XGBoost)</div>
            <div className="metric-value">{result ? result.predicted_aqi : '--'}</div>
            <div className={`metric-badge ${result ? getHealthRecommendation(result.category).class : 'badge-mod'}`}>{result ? result.category : 'Pending'}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Next hour (LSTM)</div>
            <div className="metric-value">191</div>
            <div className="metric-badge badge-mod">Moderate</div>
          </div>
          <div className="metric">
            <div className="metric-label">Tomorrow (SARIMAX)</div>
            <div className="metric-value">218</div>
            <div className="metric-badge badge-poor">Poor</div>
          </div>
        </div>

        <div className="cards">
          <div className="card">
            <div className="card-title"><i className="ti ti-bolt" aria-hidden="true"></i> Hourly prediction <span className="model-tag">XGBoost</span></div>
            <div className="card-sub">Enter last hour's readings to predict next hour's AQI</div>
            
            <button type="button" className="predict-btn" onClick={handleAutoFill} disabled={historyData.length === 0} style={{marginBottom: '10px', background: 'var(--color-background-primary)'}}>
               🪄 Load Sample Data
            </button>

            <form onSubmit={handleSubmit}>
              <div className="input-grid">
                {['PM25', 'PM10', 'NO2', 'SO2', 'NH3', 'CO', 'O3'].map(pollutant => (
                  <div className="input-row" key={pollutant}>
                    <div className="input-label">{pollutant} {pollutant !== 'CO' ? '(µg/m³)' : '(mg/m³)'}</div>
                    <div className="real-input-container">
                      <input type="number" id={pollutant} name={pollutant} value={formData[pollutant]} onChange={handleChange} step="0.1" min="0" required />
                    </div>
                  </div>
                ))}
                {WEATHER_PARAMS.map(weather => (
                  <div className="input-row" key={weather.id}>
                    <div className="input-label">{weather.label}</div>
                    <div className="real-input-container">
                      <input type="number" id={weather.id} name={weather.id} value={formData[weather.id]} onChange={handleChange} step="0.1" required />
                    </div>
                  </div>
                ))}
              </div>
              
              <button type="submit" className="predict-btn" disabled={loading}>
                <i className="ti ti-arrow-right" aria-hidden="true"></i> 
                {loading ? "Predicting..." : "Predict next hour"}
              </button>
            </form>

            {result && (
              <div className="result-box">
                <div>
                  <div className="result-cat">Predicted AQI — next hour</div>
                  <div className="result-aqi">{result.predicted_aqi} <span style={{fontSize:'13px',fontWeight:400,color:'var(--color-text-secondary)'}}>{result.category}</span></div>
                </div>
                <div style={{textAlign:'right',fontSize:'12px',color:'var(--color-text-secondary)'}}>Dominant: {getPrimaryPollutant()}</div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-title"><i className="ti ti-brain" aria-hidden="true"></i> Hourly prediction <span className="model-tag">LSTM / GRU</span></div>
            <div className="card-sub">Uses last 48 hours of readings as input sequence</div>
            <div style={{background:'var(--color-background-secondary)',borderRadius:'var(--border-radius-md)',padding:'10px 12px',fontSize:'12px',color:'var(--color-text-secondary)',marginBottom:'10px'}}>
              <i className="ti ti-info-circle" aria-hidden="true"></i> Sequence of 48 h loaded from history — upload a CSV or connect live sensor data
            </div>
            <div className="input-grid" style={{marginBottom:'10px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', overflowY: 'visible', maxHeight: 'none'}}>
              <div style={{background:'var(--color-background-secondary)',borderRadius:'var(--border-radius-md)',padding:'8px 10px'}}>
                <div style={{fontSize:'11px',color:'var(--color-text-secondary)'}}>BiLSTM</div>
                <div style={{fontSize:'20px',fontWeight:500,color:'var(--color-text-primary)'}}>191</div>
                <div style={{fontSize:'11px',color:'var(--color-text-secondary)'}}>RMSE 7.2</div>
              </div>
              <div style={{background:'var(--color-background-secondary)',borderRadius:'var(--border-radius-md)',padding:'8px 10px'}}>
                <div style={{fontSize:'11px',color:'var(--color-text-secondary)'}}>BiGRU</div>
                <div style={{fontSize:'20px',fontWeight:500,color:'var(--color-text-primary)'}}>189</div>
                <div style={{fontSize:'11px',color:'var(--color-text-secondary)'}}>RMSE 7.5</div>
              </div>
            </div>
            <div className="chart-area">
              <svg className="chart" viewBox="0 0 300 120" preserveAspectRatio="none">
                <polyline fill="none" stroke="#378ADD" strokeWidth="1.5"
                  points="0,80 20,75 40,70 60,65 80,68 100,72 120,78 140,74 160,69 180,64 200,60 220,58 240,62 260,65 280,69 300,68"/>
                <polyline fill="none" stroke="#D85A30" strokeWidth="1.5" strokeDasharray="4,3"
                  points="0,82 20,77 40,72 60,67 80,70 100,74 120,80 140,76 160,71 180,66 200,63 220,61 240,65 260,68 280,71 300,70"/>
                <line x1="260" y1="0" x2="260" y2="120" stroke="var(--color-border-secondary)" strokeWidth="0.5" strokeDasharray="3,3"/>
                <text x="262" y="14" fontSize="9" fill="var(--color-text-tertiary)">now</text>
              </svg>
            </div>
            <div style={{display:'flex',gap:'12px',fontSize:'11px',color:'var(--color-text-secondary)',marginTop:'4px'}}>
              <span style={{display:'flex',alignItems:'center',gap:'4px'}}><span style={{width:'16px',height:'2px',background:'#378ADD',display:'inline-block'}}></span>LSTM</span>
              <span style={{display:'flex',alignItems:'center',gap:'4px'}}><span style={{width:'16px',height:'2px',background:'#D85A30',display:'inline-block'}}></span>GRU</span>
            </div>
          </div>
        </div>

        <div className="forecast-card">
          <div className="card-title" style={{marginBottom:'4px'}}><i className="ti ti-calendar-week" aria-hidden="true"></i> 7-day daily forecast <span className="model-tag">SARIMAX</span></div>
          <div className="card-sub">Rolling 1-step-ahead — conditions on last observed daily AQI</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(7,1fr)',gap:'8px',marginTop:'8px'}}>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Today</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>187</div>
              <div className="metric-badge badge-mod" style={{fontSize:'10px'}}>Moderate</div>
            </div>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Tue</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>218</div>
              <div className="metric-badge badge-poor" style={{fontSize:'10px'}}>Poor</div>
            </div>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Wed</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>234</div>
              <div className="metric-badge badge-poor" style={{fontSize:'10px'}}>Poor</div>
            </div>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Thu</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>201</div>
              <div className="metric-badge badge-poor" style={{fontSize:'10px'}}>Poor</div>
            </div>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Fri</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>178</div>
              <div className="metric-badge badge-mod" style={{fontSize:'10px'}}>Moderate</div>
            </div>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Sat</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>155</div>
              <div className="metric-badge badge-mod" style={{fontSize:'10px'}}>Moderate</div>
            </div>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:'11px',color:'var(--color-text-secondary)',marginBottom:'4px'}}>Sun</div>
              <div style={{fontSize:'16px',fontWeight:500,color:'var(--color-text-primary)'}}>132</div>
              <div className="metric-badge badge-sat" style={{fontSize:'10px'}}>Satisfactory</div>
            </div>
          </div>
        </div>

        <div className="bottom-cards">
          <div className="card">
            <div className="card-title"><i className="ti ti-chart-bar" aria-hidden="true"></i> Model performance</div>
            <div style={{marginTop:'8px'}}>
              <div className="day-row">
                <span className="day-label">XGBoost (hourly)</span>
                <span className="day-aqi">RMSE 6.8 · MAE 4.9</span>
              </div>
              <div className="day-row">
                <span className="day-label">BiLSTM (hourly)</span>
                <span className="day-aqi">RMSE 7.2 · MAE 5.1</span>
              </div>
              <div className="day-row">
                <span className="day-label">BiGRU (hourly)</span>
                <span className="day-aqi">RMSE 7.5 · MAE 5.3</span>
              </div>
              <div className="day-row">
                <span className="day-label">SARIMAX rolling (daily)</span>
                <span className="day-aqi">RMSE 18.4 · MAE 13.2</span>
              </div>
            </div>
          </div>
          <div className="card">
            <div className="card-title"><i className="ti ti-map-2" aria-hidden="true"></i> Station coverage</div>
            <div style={{marginTop:'8px'}}>
              <div className="day-row"><span className="day-label"><span className="cat-dot dot-good"></span>Anand Vihar</span><span className="day-aqi">187 AQI</span></div>
              <div className="day-row"><span className="day-label"><span className="cat-dot dot-mod"></span>IITG, Guwahati</span><span className="day-aqi">142 AQI</span></div>
              <div className="day-row"><span className="day-label"><span className="cat-dot dot-sat"></span>Jayanagar</span><span className="day-aqi">98 AQI</span></div>
              <div className="day-row"><span className="day-label"><span className="cat-dot dot-mod"></span>Kodungaiyur</span><span className="day-aqi">163 AQI</span></div>
              <div className="day-row"><span className="day-label"><span className="cat-dot dot-poor"></span>Patia</span><span className="day-aqi">211 AQI</span></div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
