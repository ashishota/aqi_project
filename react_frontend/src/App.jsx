import { useState, useEffect } from 'react';
import axios from 'axios';

import { FALLBACK_CITIES, FALLBACK_METRICS, FALLBACK_CHART_POINTS } from './constants/fallbackData';
import { AQI_CAT, AQI_COLOR, pointsToSvg } from './utils/aqiHelpers';

import Navbar from './components/Navbar';
import OverviewTab from './components/OverviewTab';
import ModelsTab from './components/ModelsTab';
import LiveDemoTab from './components/LiveDemoTab';
import HistoryTab from './components/HistoryTab';
import AboutTab from './components/AboutTab';

const API_BASE_URL = 'http://localhost:5000';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [citiesList, setCitiesList] = useState(FALLBACK_CITIES);
  const [metricsAll, setMetricsAll] = useState(null);
  const [cityAqis, setCityAqis] = useState({ anandvihar: 287, colaba: 78, iitg: 142, jayanagar: 63, kodungaiyur: 118, patia: 94 });
  
  // Tab-specific selections
  const [selectedCityOverview, setSelectedCityOverview] = useState('anandvihar');
  const [activeMetricsFilter, setActiveMetricsFilter] = useState('all');
  const [selectedCityModels, setSelectedCityModels] = useState('anandvihar');
  
  // Live Demo state
  const [selectedCityDemo, setSelectedCityDemo] = useState('anandvihar');
  const [demoSample, setDemoSample] = useState(null);
  const [demoLoading, setDemoLoading] = useState(false);
  
  // History state
  const [histCityFilter, setHistCityFilter] = useState('anandvihar');
  const [histModelFilter, setHistModelFilter] = useState('all');
  const [historyDataObj, setHistoryDataObj] = useState(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initial Data Fetch
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [citiesRes, metricsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/cities`),
          axios.get(`${API_BASE_URL}/metrics/all`)
        ]);
        
        if (citiesRes.data && citiesRes.data.length > 0) {
          setCitiesList(citiesRes.data);
          setSelectedCityOverview(citiesRes.data[0].key);
          setSelectedCityModels(citiesRes.data[0].key);
          setSelectedCityDemo(citiesRes.data[0].key);
          setHistCityFilter(citiesRes.data[0].key);

          // Fetch all cities' history to populate real AQI numbers on all city cards instantly
          const historyResults = await Promise.allSettled(
            citiesRes.data.map(c => axios.get(`${API_BASE_URL}/history/${c.key}`))
          );
          const aqiUpdates = {};
          citiesRes.data.forEach((c, i) => {
            const res = historyResults[i];
            if (res.status === 'fulfilled' && res.value?.data?.actual?.length > 0) {
              const arr = res.value.data.actual;
              aqiUpdates[c.key] = Math.round(arr[arr.length - 1]);
            }
          });
          if (Object.keys(aqiUpdates).length > 0) {
            setCityAqis(prev => ({ ...prev, ...aqiUpdates }));
          }
        }
        if (metricsRes.data) {
          setMetricsAll(metricsRes.data);
        }
        setError(null);
      } catch (err) {
        console.warn("Backend fetch failed, using fallback data:", err);
        setError("Could not connect to live Flask backend. Showing static fallback data.");
      } finally {
        setLoading(false);
      }
    };
    fetchInitialData();
  }, []);

  // Fetch Demo Sample
  const fetchDemoSample = async (cityKey, type = 'random', idx = 0) => {
    setDemoLoading(true);
    try {
      const url = type === 'random' 
        ? `${API_BASE_URL}/predict/random/${cityKey}`
        : `${API_BASE_URL}/predict/index/${cityKey}/${idx}`;
      const res = await axios.get(url);
      setDemoSample(res.data);
    } catch (err) {
      console.warn("Demo fetch failed, generating mock sample:", err);
      // Generate beautiful mock sample matching aqi_dashboard (1).html
      const baseAqi = cityAqis[cityKey] || 150;
      const actual = Math.max(10, Math.round(baseAqi + (Math.random() - 0.5) * 40));
      const lstm = Math.max(10, actual + Math.round((Math.random() - 0.5) * 14));
      const gru = Math.max(10, actual + Math.round((Math.random() - 0.5) * 10));
      const xgb = Math.max(10, actual + Math.round((Math.random() - 0.5) * 24));
      
      const errors = { LSTM: Math.abs(actual - lstm), GRU: Math.abs(actual - gru), XGBoost: Math.abs(actual - xgb) };
      const winner = Object.keys(errors).reduce((a, b) => errors[a] < errors[b] ? a : b);
      
      setDemoSample({
        city: cityKey,
        label: citiesList.find(c => c.key === cityKey)?.label || cityKey,
        station: citiesList.find(c => c.key === cityKey)?.station || "",
        sample_index: idx || Math.floor(Math.random() * 2000),
        timestamp: "2023-11-14 14:00:00",
        actual_aqi: actual,
        actual_category: AQI_CAT(actual).cat,
        actual_color: AQI_COLOR(actual),
        best_model: winner,
        predictions: [
          { model: "LSTM", predicted: lstm, error: Math.abs(actual - lstm), pct_error: ((Math.abs(actual - lstm)/actual)*100).toFixed(1), category: AQI_CAT(lstm).cat, color: "#7F77DD", advice: "Next-hour model · 48h window input" },
          { model: "GRU", predicted: gru, error: Math.abs(actual - gru), pct_error: ((Math.abs(actual - gru)/actual)*100).toFixed(1), category: AQI_CAT(gru).cat, color: "#1D9E75", advice: "Next-hour model · 48h window input" },
          { model: "XGBoost", predicted: xgb, error: Math.abs(actual - xgb), pct_error: ((Math.abs(actual - xgb)/actual)*100).toFixed(1), category: AQI_CAT(xgb).cat, color: "#EF9F27", advice: "Next-hour model · tabular features" },
        ]
      });
    } finally {
      setDemoLoading(false);
    }
  };

  useEffect(() => {
    fetchDemoSample(selectedCityDemo, 'random');
  }, [selectedCityDemo]);

  // Fetch History Data
  useEffect(() => {
    const fetchHistoryData = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/history/${histCityFilter}`);
        setHistoryDataObj(res.data);
        if (res.data && res.data.actual && res.data.actual.length > 0) {
          const latestAqi = res.data.actual[res.data.actual.length - 1];
          setCityAqis(prev => ({ ...prev, [histCityFilter]: Math.round(latestAqi) }));
        }
      } catch (err) {
        console.warn("History fetch failed, using fallback history:", err);
        setHistoryDataObj(null);
      }
    };
    fetchHistoryData();
  }, [histCityFilter]);

  // Fetch History for Overview Chart when selectedCityOverview changes
  const [overviewHistory, setOverviewHistory] = useState(null);
  useEffect(() => {
    const fetchOverviewHist = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/history/${selectedCityOverview}`);
        setOverviewHistory(res.data);
        if (res.data && res.data.actual && res.data.actual.length > 0) {
          const latestAqi = res.data.actual[res.data.actual.length - 1];
          setCityAqis(prev => ({ ...prev, [selectedCityOverview]: Math.round(latestAqi) }));
        }
      } catch (err) {
        setOverviewHistory(null);
      }
    };
    fetchOverviewHist();
  }, [selectedCityOverview]);

  // Get current city object and metrics
  const currentCityObj = citiesList.find(c => c.key === selectedCityOverview) || citiesList[0];
  const currentChartPts = overviewHistory ? {
    actual: pointsToSvg(overviewHistory.actual),
    lstm:   pointsToSvg(overviewHistory.LSTM),
    gru:    pointsToSvg(overviewHistory.GRU),
    xgb:    pointsToSvg(overviewHistory.XGBoost)
  } : FALLBACK_CHART_POINTS[selectedCityOverview];

  if (loading) {
    return <div className="app"><div className="loading-box">Loading Multi-City AQI Dashboard...</div></div>;
  }

  return (
    <div className="app">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {error && <div className="main" style={{paddingBottom:0}}><div className="error-box">{error}</div></div>}

      <OverviewTab
        activeTab={activeTab}
        citiesList={citiesList}
        cityAqis={cityAqis}
        selectedCityOverview={selectedCityOverview}
        setSelectedCityOverview={setSelectedCityOverview}
        activeMetricsFilter={activeMetricsFilter}
        setActiveMetricsFilter={setActiveMetricsFilter}
        metricsAll={metricsAll}
        currentCityObj={currentCityObj}
        currentChartPts={currentChartPts}
      />

      <ModelsTab
        activeTab={activeTab}
        citiesList={citiesList}
        selectedCityModels={selectedCityModels}
        setSelectedCityModels={setSelectedCityModels}
        metricsAll={metricsAll}
      />

      <LiveDemoTab
        activeTab={activeTab}
        citiesList={citiesList}
        selectedCityDemo={selectedCityDemo}
        setSelectedCityDemo={setSelectedCityDemo}
        demoSample={demoSample}
        demoLoading={demoLoading}
        fetchDemoSample={fetchDemoSample}
      />

      <HistoryTab
        activeTab={activeTab}
        citiesList={citiesList}
        histCityFilter={histCityFilter}
        setHistCityFilter={setHistCityFilter}
        histModelFilter={histModelFilter}
        setHistModelFilter={setHistModelFilter}
        historyDataObj={historyDataObj}
        cityAqis={cityAqis}
      />

      <AboutTab
        activeTab={activeTab}
        citiesList={citiesList}
      />
    </div>
  );
}

export default App;
