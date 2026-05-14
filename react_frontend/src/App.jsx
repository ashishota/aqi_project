import { useState, useEffect } from 'react'
import axios from 'axios'

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

  useEffect(() => {
    // Fetch available cities
    axios.get('http://localhost:5000/api/cities')
      .then(response => {
        if (response.data.cities && response.data.cities.length > 0) {
          setCities(response.data.cities)
          setFormData(prev => ({ ...prev, city: response.data.cities[0] }))
        }
      })
      .catch(err => {
        console.error("Error fetching cities:", err)
        setError("Could not load cities. Ensure backend is running.")
      })
  }, [])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: name === 'city' ? value : parseFloat(value)
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post('http://localhost:5000/api/predict', formData)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred during prediction.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <h1>AQI Forecaster</h1>
      <p className="subtitle">Predict Air Quality Index for Indian Cities using XGBoost</p>

      {error && <div className="error-msg">{error}</div>}

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
              step="1"
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
  )
}

export default App
