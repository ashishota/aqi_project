from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for the React frontend

# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────
# Using relative paths so it works on any machine/server
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

MODEL_REGISTRY = {
    "Delhi": os.path.join(MODEL_DIR, "delhi_xgboost_aqi.pkl"),
    "Chennai": os.path.join(MODEL_DIR, "chennai_xgboost_aqi.pkl"),
    "Bangalore": os.path.join(MODEL_DIR, "bangalore_xgboost_aqi.pkl"),
    "Bhubaneswar": os.path.join(MODEL_DIR, "Bhubaneswar_xgboost_aqi.pkl"),
    "Guwahati": os.path.join(MODEL_DIR, "Guwahati_xgboost_aqi.pkl"),
    "Mumbai": os.path.join(MODEL_DIR, "mumbai_xgboost_aqi.pkl")
}

# ──────────────────────────────────────────────
# Data Registry (for historical charts)
# ──────────────────────────────────────────────
CSV_DIR = os.path.join(os.path.dirname(__file__), 'data')

DATA_REGISTRY = {
    "Delhi": os.path.join(CSV_DIR, "AnandVihar_AQI_daily_clean.csv"),
    "Chennai": os.path.join(CSV_DIR, "Kodungaiyur_AQI_daily_clean.csv"),
    "Bangalore": os.path.join(CSV_DIR, "Jayanagar_AQI_daily_clean.csv"),
    "Bhubaneswar": os.path.join(CSV_DIR, "Patia_AQI_daily_clean.csv"),
    "Guwahati": os.path.join(CSV_DIR, "IITG_AQI_daily_clean.csv"),
    "Mumbai": os.path.join(CSV_DIR, "Colaba_AQI_daily_clean.csv")
}

# ──────────────────────────────────────────────
# Features List
# ──────────────────────────────────────────────
BASE_FEATURES = ["PM25", "PM10", "NO2", "SO2", "NH3", "CO", "O3"]
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

def create_input_features(base_inputs, previous_aqi):
    now = datetime.now()
    data = base_inputs.copy()

    # Lag Features
    data["lag_1"] = previous_aqi
    data["lag_2"] = previous_aqi
    data["lag_3"] = previous_aqi
    data["lag_7"] = previous_aqi
    data["lag_14"] = previous_aqi
    data["lag_30"] = previous_aqi

    # Rolling Statistics
    data["roll_mean_7"] = previous_aqi
    data["roll_std_7"] = 10.0
    data["roll_mean_30"] = previous_aqi

    # Calendar Features
    data["day_of_week"] = now.weekday()
    data["month"] = now.month
    data["day_of_year"] = now.timetuple().tm_yday
    data["quarter"] = (now.month - 1) // 3 + 1

    # Cyclical Encoding
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    return pd.DataFrame([data])[FEATURE_COLUMNS]

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
        for feature in BASE_FEATURES:
            base_inputs[feature] = float(req_data.get(feature, 10.0))

        # Get Model
        model = get_model(city)
        if model is None:
            return jsonify({"error": f"Model not found for city: {city}"}), 404

        # Feature Engineering
        feature_df = create_input_features(base_inputs, previous_aqi)

        # Predict
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
        last_30 = df.tail(30)[['Timestamp', 'AQI']]
        last_30['Timestamp'] = pd.to_datetime(last_30['Timestamp']).dt.strftime('%b %d')
        
        # Round AQI for cleaner display
        last_30['AQI'] = last_30['AQI'].round(0)
        
        history_data = last_30.rename(columns={'Timestamp': 'date', 'AQI': 'aqi'}).to_dict(orient='records')
        
        return jsonify({"history": history_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
