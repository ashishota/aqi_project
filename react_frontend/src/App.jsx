import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// Update this to 'http://localhost:5000' for local testing before pushing
const API_BASE_URL = 'https://aqi-backend-api-q9as.onrender.com'

function App() {
  const [cities, setCities] = useState([])
  const [formData, setFormData] = useState({
    city: '',
    previous_aqi: 150.0,
    PM25: 10.0,
    PM10: 10.0,
    NO2: 10.0,
    SO2: 10.0,
    NH3: 10.0,
    CO: 10.0,
    O3: 10.0
  })
  
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [historyData, setHistoryData] = useState([])

  const fetchHistory = async (city) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/history?city=${city}`)
      setHistoryData(response.data.history)
    } catch (err) {
      console.error("Error fetching history:", err)
      // Don't set main error state so form still works even if history fails
    }
  }

  useEffect(() => {
    // Fetch available cities
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
    
    // Fetch new history if city changes
    if (name === 'city') {
      fetchHistory(value)
    }
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

  // Custom Tooltip for the chart
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="label">{`${label}`}</p>
          <p className="intro">{`AQI : ${payload[0].value}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="app-container">
      <h1>AQI Forecaster</h1>
      <p className="subtitle">Predict Air Quality Index for Indian Cities using XGBoost</p>

      {error && <div className="error-msg">{error}</div>}

      <div className="main-content">
        <div className="form-section">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="city">Select City</label>
              <select 
                id="city" 
                name="city" 
                value={formData.city} 
                onChange={handleChange}
              >
                {cities.map(city => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </select>
            </div>

            <div className="grid">
              {['PM25', 'PM10', 'NO2', 'SO2', 'NH3', 'CO', 'O3'].map(pollutant => (
                <div className="form-group" key={pollutant}>
                  <label htmlFor={pollutant}>{pollutant}</label>
                  <input
                    type="number"
                    id={pollutant}
                    name={pollutant}
                    value={formData[pollutant]}
                    onChange={handleChange}
                    step="0.1"
                    min="0"
                    required
                  />
                </div>
              ))}
              <div className="form-group">
                <label htmlFor="previous_aqi">Previous Day AQI</label>
                <input
                  type="number"
                  id="previous_aqi"
                  name="previous_aqi"
                  value={formData.previous_aqi}
                  onChange={handleChange}
                  step="0.1"
                  min="0"
                  required
                />
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? "Predicting..." : "Predict AQI"}
            </button>
          </form>

          {result && (
            <div className="result-container">
              <div className="result-title">Predicted Air Quality Index</div>
              <div className="result-aqi">{result.predicted_aqi}</div>
              <div 
                className="result-category" 
                style={{ backgroundColor: result.color }}
              >
                {result.category}
              </div>
            </div>
          )}
        </div>

        {historyData.length > 0 && (
          <div className="chart-section">
            <h2 className="chart-title">Last 30 Days Trend ({formData.city})</h2>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historyData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="date" stroke="#aaaaaa" tick={{fontSize: 12}} />
                  <YAxis stroke="#aaaaaa" tick={{fontSize: 12}} domain={['auto', 'auto']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="aqi" 
                    stroke="#42a5f5" 
                    strokeWidth={3}
                    dot={{ fill: '#7e57c2', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, fill: '#fff' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
