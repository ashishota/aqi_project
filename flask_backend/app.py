from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for the React frontend

# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────
# Using relative paths so it works on any machine/server
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'XGBOOST', 'saved_models')

MODEL_REGISTRY = {
    "Delhi": os.path.join(MODEL_DIR, "AnandVihar_xgboost_aqi.pkl"),
    "Chennai": os.path.join(MODEL_DIR, "Kodungaiyur_xgboost_aqi.pkl"),
    "Bangalore": os.path.join(MODEL_DIR, "jayanagar_xgboost_aqi.pkl"),
    "Bhubaneswar": os.path.join(MODEL_DIR, "patia_xgboost_aqi.pkl"),
    "Guwahati": os.path.join(MODEL_DIR, "iitg_xgboost_aqi.pkl"),
    "Mumbai": os.path.join(MODEL_DIR, "colaba_xgboost_aqi_hourly.pkl")
}

# ──────────────────────────────────────────────
# Data Registry (for historical charts)
# ──────────────────────────────────────────────
CSV_DIR = os.path.join(os.path.dirname(__file__), 'data')

DATA_REGISTRY = {
    "Delhi": os.path.join(CSV_DIR, "AnandVihar_AQI_cleaned.csv"),
    "Chennai": os.path.join(CSV_DIR, "Kodungaiyur_AQI_cleaned.csv"),
    "Bangalore": os.path.join(CSV_DIR, "Jayanagar_AQI_cleaned.csv"),
    "Bhubaneswar": os.path.join(CSV_DIR, "Patia_AQI_cleaned.csv"),
    "Guwahati": os.path.join(CSV_DIR, "IITG_AQI_cleaned.csv"),
    "Mumbai": os.path.join(CSV_DIR, "Colaba_AQI_cleaned.csv")
}

# ──────────────────────────────────────────────
# Features List
# ──────────────────────────────────────────────
BASE_FEATURES = ["PM25", "PM10", "NO2", "SO2", "NH3", "CO", "O3", "AT", "RH", "WS", "WD", "SR", "BP"]
LAG_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30"]
ROLLING_FEATURES = ["roll_mean_7", "roll_std_7", "roll_mean_30"]
CALENDAR_FEATURES = ["day_of_week", "month", "day_of_year", "quarter"]
CYCLICAL_FEATURES = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

FEATURE_COLUMNS = BASE_FEATURES + LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES + CYCLICAL_FEATURES

# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def get_aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "#00b050"
    elif aqi <= 100:
        return "Satisfactory", "#92d050"
    elif aqi <= 200:
        return "Moderate", "#ffff00"
    elif aqi <= 300:
        return "Poor", "#ff9900"
    elif aqi <= 400:
        return "Very Poor", "#ff0000"
    return "Severe", "#7030a0"

def parse_number(s):
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0

def build_features(df, base_inputs, features_in, target_time=None):
    if target_time is None:
        target_time = pd.to_datetime('now')
        
    new_row = df.iloc[-1].copy()
    for k, v in base_inputs.items():
        if k in new_row:
            new_row[k] = v
    new_row['Timestamp'] = target_time
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    aqi_col = 'AQI'
    out = pd.DataFrame(index=[0])
    
    for f in features_in:
        if f in df.columns:
            out[f] = df[f].iloc[-1]
            continue
            
        if f == 'AT':
            out[f] = base_inputs.get('AT', 25.0)
            continue
            
        if 'lag' in f.lower():
            shift_val = parse_number(f)
            if 'pm25' in f.lower(): out[f] = df['PM25'].shift(shift_val).iloc[-1]
            elif 'pm10' in f.lower(): out[f] = df['PM10'].shift(shift_val).iloc[-1]
            elif 'co_' in f.lower(): out[f] = df['CO'].shift(shift_val).iloc[-1]
            elif 'nox_' in f.lower(): out[f] = df['NOx'].shift(shift_val).iloc[-1]
            elif 'no2_' in f.lower(): out[f] = df['NO2'].shift(shift_val).iloc[-1]
            else: out[f] = df[aqi_col].shift(shift_val).iloc[-1]
            continue
            
        if 'roll_mean' in f.lower():
            win = parse_number(f)
            out[f] = df[aqi_col].iloc[-(win+1):-1].mean()
            continue
        if 'roll_std' in f.lower():
            win = parse_number(f)
            out[f] = df[aqi_col].iloc[-(win+1):-1].std()
            continue
            
        now = target_time
        f_lower = f.lower()
        if f_lower == 'hour': out[f] = now.hour
        elif f_lower in ['dayofweek', 'day_of_week']: out[f] = now.weekday()
        elif f_lower == 'month': out[f] = now.month
        elif f_lower == 'year': out[f] = now.year
        elif f_lower in ['dayofyear', 'day_of_year']: out[f] = now.timetuple().tm_yday
        elif f_lower == 'quarter': out[f] = (now.month - 1) // 3 + 1
        elif f_lower in ['isweekend', 'is_weekend']: out[f] = 1 if now.weekday() >= 5 else 0
        elif 'season' in f_lower:
            month = now.month
            if month in [12, 1, 2]: out[f] = 1
            elif month in [3, 4, 5]: out[f] = 2
            elif month in [6, 7, 8, 9]: out[f] = 3
            else: out[f] = 4
        elif 'cat_enc' in f_lower or 'category_enc' in f_lower:
            out[f] = 2 # Default Moderate
            
        elif 'hour_sin' in f_lower: out[f] = np.sin(2 * np.pi * now.hour / 24)
        elif 'hour_cos' in f_lower: out[f] = np.cos(2 * np.pi * now.hour / 24)
        elif 'month_sin' in f_lower: out[f] = np.sin(2 * np.pi * now.month / 12)
        elif 'month_cos' in f_lower: out[f] = np.cos(2 * np.pi * now.month / 12)
        elif 'dow_sin' in f_lower: out[f] = np.sin(2 * np.pi * now.weekday() / 7)
        elif 'dow_cos' in f_lower: out[f] = np.cos(2 * np.pi * now.weekday() / 7)
        elif 'doy_sin' in f_lower: out[f] = np.sin(2 * np.pi * now.timetuple().tm_yday / 365)
        elif 'doy_cos' in f_lower: out[f] = np.cos(2 * np.pi * now.timetuple().tm_yday / 365)
        else: out[f] = 0.0

    return out

# Dictionary to store loaded models
loaded_models = {}

def get_model(city):
    if city not in loaded_models:
        model_path = MODEL_REGISTRY.get(city)
        if not model_path or not os.path.exists(model_path):
            return None
        loaded_models[city] = joblib.load(model_path)
    return loaded_models[city]

# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify({"cities": list(MODEL_REGISTRY.keys())})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        city = req_data.get('city')
        previous_aqi = float(req_data.get('previous_aqi', 150.0))
        
        # Extract base inputs
        base_inputs = {}
        default_vals = {"AT": 25.0, "RH": 50.0, "WS": 1.0, "WD": 180.0, "SR": 100.0, "BP": 1000.0}
        for feature in BASE_FEATURES:
            base_inputs[feature] = float(req_data.get(feature, default_vals.get(feature, 10.0)))

        # Load Model
        model = get_model(city)
        if model is None:
            return jsonify({"error": f"Model not found for city: {city}"}), 404
            
        features_in = getattr(model, "feature_names_in_", [])
        
        # Load historical data context for lags
        csv_path = DATA_REGISTRY.get(city)
        if not csv_path or not os.path.exists(csv_path):
             return jsonify({"error": f"Data file not found for {city}"}), 404
             
        df = pd.read_csv(csv_path).tail(200).reset_index(drop=True)

        # Feature Engineering
        feature_df = build_features(df, base_inputs, features_in)

        # Predict (Model natively predicts Next Hour AQI due to shifted target)
        prediction = model.predict(feature_df)[0]
        prediction = max(0.0, float(prediction))
        category, color = get_aqi_category(prediction)

        return jsonify({
            "predicted_aqi": round(prediction, 2),
            "category": category,
            "color": color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        city = request.args.get('city')
        if not city or city not in DATA_REGISTRY:
            return jsonify({"error": "Valid city required."}), 400
            
        csv_path = DATA_REGISTRY[city]
        if not os.path.exists(csv_path):
            return jsonify({"error": f"Data file not found for {city}"}), 404
            
        df = pd.read_csv(csv_path)
        
        # Columns to extract for history
        cols_to_extract = ['Timestamp', 'AQI'] + BASE_FEATURES
        
        # Last 168 hours (7 days) of data
        last_168 = df.tail(168)[cols_to_extract].copy()
        last_168['Timestamp'] = pd.to_datetime(last_168['Timestamp']).dt.strftime('%b %d, %H:%M')
        
        # Round values for cleaner display
        last_168['AQI'] = last_168['AQI'].round(0)
        for feature in BASE_FEATURES:
            if feature in last_168.columns:
                last_168[feature] = last_168[feature].round(1)
        
        # Rename Timestamp to date for Recharts
        last_365 = last_168.rename(columns={'Timestamp': 'date'})
        
        # Rename Timestamp to date for Recharts
        last_365 = last_365.rename(columns={'Timestamp': 'date'})
        
        history_data = last_365.to_dict(orient='records')
        
        return jsonify({"history": history_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
