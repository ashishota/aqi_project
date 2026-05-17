"""
AQI Forecasting System — Flask Backend
========================================
Project: An Intelligent System for AQI Prediction using Supervised Machine Learning
Station: Anand Vihar, Delhi

What this backend does:
  - Loads all 3 trained models (LSTM, GRU, XGBoost)
  - Loads the dataset and reconstructs the test set
  - Picks a random sample from the TEST SET only (unseen data)
  - Feeds the correct 48-hour window to LSTM/GRU
  - Feeds the correct feature row to XGBoost
  - Returns predictions from all 3 models vs actual AQI
  - Returns overall metrics for the comparison dashboard

Endpoints:
  GET  /                    → health check
  GET  /metrics             → overall test set metrics for all 3 models
  GET  /predict/random      → random test sample → all 3 predictions vs actual
  GET  /predict/index/<n>   → specific test sample by index
  GET  /history             → full test set actual vs predicted (for chart)

Run:
  pip install flask flask-cors pandas numpy joblib xgboost scikit-learn tensorflow
  python app.py
"""

import os, json, joblib, warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── File paths — update these to match your saved files ──────────────────────
DATASET_PATH        = "AnandVihar_AQI_cleaned.csv"
LSTM_MODEL_PATH     = "saved_models/anandvihar_lstm_model.keras"
GRU_MODEL_PATH      = "saved_models/anandvihar_gru_model.keras"
FEATURE_SCALER_PATH = "saved_models/anandvihar_feature_scaler.pkl"
TARGET_SCALER_PATH  = "saved_models/anandvihar_target_scaler.pkl"
XGBOOST_MODEL_PATH  = "saved_models/AnandVihar_xgboost_aqi.pkl"
XGBOOST_FEATS_PATH  = "saved_models/AnandVihar_xgboost_features.json"

# ── Constants — must match your notebooks exactly ─────────────────────────────
LOOK_BACK = 48
LSTM_FEATURES = [
    'PM25', 'PM10', 'NO2', 'CO',
    'AQI_lag1', 'AQI_lag2',
    'AQI_roll3_mean', 'AQI_roll6_mean',
    'O3', 'SO2', 'NH3',
    'NOx_total', 'hour_sin', 'hour_cos',
    'AQI_roll12_mean', 'PM25_lag1',
    'RH', 'NO', 'WS', 'AT', 'SR',
    'Toluene_Benzene_ratio',
]
TARGET = 'AQI'

app = Flask(__name__)
CORS(app)

# =============================================================================
# STARTUP — Load everything once when server starts
# =============================================================================

print("=" * 55)
print("  AQI Forecasting System — Loading...")
print("=" * 55)

# ── 1. Load raw dataset ───────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df_raw = pd.read_csv(DATASET_PATH, parse_dates=["Timestamp"])
df_raw = df_raw.set_index("Timestamp").sort_index()
print(f"      Shape: {df_raw.shape} | "
      f"{df_raw.index.min().date()} → {df_raw.index.max().date()}")

# ── 2. Feature engineering (must match LSTM notebook exactly) ─────────────────
print("[2/5] Reproducing LSTM feature engineering...")
fe = df_raw.copy()

fe["AQI_lag1"]  = fe[TARGET].shift(1)
fe["AQI_lag2"]  = fe[TARGET].shift(2)
fe["PM25_lag1"] = fe["PM25"].shift(1)

for w in [3, 6, 12]:
    fe[f"AQI_roll{w}_mean"] = fe[TARGET].shift(1).rolling(w).mean()

fe["hour_sin"] = np.sin(2 * np.pi * fe.index.hour / 24)
fe["hour_cos"] = np.cos(2 * np.pi * fe.index.hour / 24)

fe["NOx_total"]             = fe["NO"] + fe["NO2"]
fe["Toluene_Benzene_ratio"] = fe["Toluene"] / (fe["Benzene"] + 1e-6)

fe.dropna(inplace=True)

available_features = [f for f in LSTM_FEATURES if f in fe.columns]
data = fe[available_features + [TARGET]].copy()
print(f"      Features available: {len(available_features)}/{len(LSTM_FEATURES)}")

# ── 3. Load scalers and scale data ────────────────────────────────────────────
print("[3/5] Loading scalers and scaling...")
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler  = joblib.load(TARGET_SCALER_PATH)

scaled_features = feature_scaler.transform(data[available_features])
scaled_target   = target_scaler.transform(data[[TARGET]])
scaled_data     = np.hstack([scaled_features, scaled_target])

# ── 4. Rebuild sequences ──────────────────────────────────────────────────────
print("[4/5] Rebuilding sequences...")

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, :-1])  # all cols except target
        y.append(data[i, -1])                  # target col
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(scaled_data, LOOK_BACK)

# Reproduce exact 70/15/15 split from notebook
n         = len(X_all)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_test = X_all[val_end:]
y_test = y_all[val_end:]

# Timestamps aligned to sequences
timestamps_all  = data.index[LOOK_BACK:]
test_timestamps = timestamps_all[val_end:]

print(f"      Test set: {len(X_test):,} samples "
      f"({test_timestamps[0].date()} → {test_timestamps[-1].date()})")

# ── 5. Load models ────────────────────────────────────────────────────────────
print("[5/5] Loading models...")
from tensorflow.keras.models import load_model

lstm_model = load_model(LSTM_MODEL_PATH)
gru_model  = load_model(GRU_MODEL_PATH)
print("      ✅ LSTM loaded")
print("      ✅ GRU  loaded")

xgb_model     = joblib.load(XGBOOST_MODEL_PATH)
xgb_feat_cols = json.load(open(XGBOOST_FEATS_PATH))
print("      ✅ XGBoost loaded")

# ── Prepare XGBoost test features ─────────────────────────────────────────────
print("      Preparing XGBoost test feature matrix...")
df_xgb = df_raw.copy()
df_xgb["AQI_next"] = df_xgb[TARGET].shift(-1)

for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
    df_xgb[f"AQI_lag_{lag}h"] = df_xgb[TARGET].shift(lag)

for win in [3, 6, 24]:
    df_xgb[f"AQI_roll_mean_{win}h"] = df_xgb[TARGET].shift(1).rolling(win).mean()
    df_xgb[f"AQI_roll_std_{win}h"]  = df_xgb[TARGET].shift(1).rolling(win).std()

df_xgb["Hour_sin"]  = np.sin(2 * np.pi * df_xgb.index.hour  / 24)
df_xgb["Hour_cos"]  = np.cos(2 * np.pi * df_xgb.index.hour  / 24)
df_xgb["Month_sin"] = np.sin(2 * np.pi * df_xgb.index.month / 12)
df_xgb["Month_cos"] = np.cos(2 * np.pi * df_xgb.index.month / 12)
df_xgb["IsWeekend"] = (df_xgb.index.dayofweek >= 5).astype(int)
doy = df_xgb.index.dayofyear
df_xgb["doy_sin"] = np.sin(2 * np.pi * doy / 365)
df_xgb["doy_cos"] = np.cos(2 * np.pi * doy / 365)

lag_roll_xgb  = [c for c in df_xgb.columns if "lag" in c or "roll" in c]
df_xgb        = df_xgb.dropna(subset=lag_roll_xgb + ["AQI_next"]).copy()

n_xgb         = len(df_xgb)
xgb_val_end   = int(n_xgb * 0.85)
df_xgb_test   = df_xgb.iloc[xgb_val_end:]

available_xgb = [c for c in xgb_feat_cols if c in df_xgb_test.columns]
X_xgb_test    = df_xgb_test[available_xgb]
y_xgb_test    = df_xgb_test["AQI_next"].values
xgb_timestamps = df_xgb_test.index

print("      ✅ XGBoost test features ready")

# ── Pre-compute all test predictions once (for metrics and history chart) ─────
print("\nPre-computing full test set predictions...")

y_true_lstm = target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
y_pred_lstm = target_scaler.inverse_transform(
                  lstm_model.predict(X_test, verbose=0)).flatten()

y_true_gru  = y_true_lstm.copy()
y_pred_gru  = target_scaler.inverse_transform(
                  gru_model.predict(X_test, verbose=0)).flatten()

y_pred_xgb  = xgb_model.predict(X_xgb_test)
y_true_xgb  = y_xgb_test

print(f"{'='*55}")
print("  ✅ All models ready. Backend running on port 5000.")
print(f"{'='*55}\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def categorize_aqi(v):
    if   v <= 50:  return "Good",         "#00e676"
    elif v <= 100: return "Satisfactory", "#00e5ff"
    elif v <= 200: return "Moderate",     "#ffe066"
    elif v <= 300: return "Poor",         "#ff9f43"
    elif v <= 400: return "Very Poor",    "#ff4757"
    else:          return "Severe",       "#9b59b6"

def health_advice(category):
    return {
        "Good":         "Air quality is satisfactory. Safe for all activities.",
        "Satisfactory": "Acceptable. Sensitive individuals should limit prolonged outdoor exertion.",
        "Moderate":     "Sensitive groups may experience effects. Limit outdoor activity.",
        "Poor":         "Everyone may experience health effects. Avoid outdoor activity.",
        "Very Poor":    "Health alert! Everyone should avoid outdoor activity.",
        "Severe":       "Emergency conditions. Stay indoors. Keep windows closed.",
    }.get(category, "")

def inverse_target(val):
    return float(target_scaler.inverse_transform([[val]])[0][0])

def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    return {
        "MAE" : round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2"  : round(r2, 4),
        "MAPE": round(mape, 2),
    }

def model_result(name, predicted, actual):
    predicted   = round(max(0, predicted), 1)
    error       = round(abs(predicted - actual), 1)
    pct_error   = round(abs(predicted - actual) / (actual + 1e-8) * 100, 1)
    cat, color  = categorize_aqi(predicted)
    return {
        "model"    : name,
        "predicted": predicted,
        "error"    : error,
        "pct_error": pct_error,
        "category" : cat,
        "color"    : color,
        "advice"   : health_advice(cat),
    }

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status"       : "running",
        "models_loaded": ["LSTM", "GRU", "XGBoost"],
        "test_samples" : int(len(X_test)),
        "test_period"  : f"{test_timestamps[0].date()} → {test_timestamps[-1].date()}",
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Returns overall test set metrics for all 3 models.
    Powers the comparison dashboard table.
    """
    results = {
        "LSTM"    : compute_metrics(y_true_lstm, y_pred_lstm),
        "GRU"     : compute_metrics(y_true_gru,  y_pred_gru),
        "XGBoost" : compute_metrics(y_true_xgb,  y_pred_xgb),
    }

    # Find winner per metric (lower is better for MAE/RMSE/MAPE, higher for R2)
    models = list(results.keys())
    winners = {
        "MAE" : min(models, key=lambda m: results[m]["MAE"]),
        "RMSE": min(models, key=lambda m: results[m]["RMSE"]),
        "R2"  : max(models, key=lambda m: results[m]["R2"]),
        "MAPE": min(models, key=lambda m: results[m]["MAPE"]),
    }

    # Overall best model — wins most metrics
    from collections import Counter
    overall_winner = Counter(winners.values()).most_common(1)[0][0]

    return jsonify({
        "models"        : results,
        "winners"       : winners,
        "overall_winner": overall_winner,
        "test_set_size" : int(len(X_test)),
        "dataset_range" : f"{df_raw.index.min().date()} → {df_raw.index.max().date()}",
        "note"          : "All metrics on unseen test set (last 15% chronologically — no data leakage)"
    })


@app.route("/predict/random", methods=["GET"])
def predict_random():
    """
    Picks a RANDOM sample from the test set.
    Returns all 3 model predictions vs actual.
    Powers the demo section of the dashboard.
    """
    idx = int(np.random.randint(0, len(X_test)))
    return _predict_at_index(idx)


@app.route("/predict/index/<int:idx>", methods=["GET"])
def predict_at_index(idx):
    """
    Predict at a specific test set index.
    Useful for stepping through samples one by one.
    """
    if idx < 0 or idx >= len(X_test):
        return jsonify({"error": f"Index must be 0 to {len(X_test)-1}"}), 400
    return _predict_at_index(idx)


def _predict_at_index(idx):
    """Core logic: run all 3 models on test sample at given index."""

    # ── LSTM ──────────────────────────────────────────────────────────────────
    x_seq     = X_test[idx:idx+1]                     # (1, 48, features)
    pred_lstm = inverse_target(lstm_model.predict(x_seq, verbose=0)[0][0])
    actual    = inverse_target(y_test[idx])

    # ── GRU (same input sequence) ─────────────────────────────────────────────
    pred_gru  = inverse_target(gru_model.predict(x_seq, verbose=0)[0][0])

    # ── XGBoost (align by timestamp) ──────────────────────────────────────────
    lstm_ts = test_timestamps[idx]
    if lstm_ts in xgb_timestamps:
        xgb_row  = X_xgb_test.loc[[lstm_ts]]
        pred_xgb = float(xgb_model.predict(xgb_row)[0])
    else:
        # Fallback to same relative index
        xgb_idx  = min(idx, len(X_xgb_test) - 1)
        pred_xgb = float(xgb_model.predict(X_xgb_test.iloc[[xgb_idx]])[0])

    actual      = round(actual, 1)
    actual_cat, actual_color = categorize_aqi(actual)

    predictions = [
        model_result("LSTM",    pred_lstm, actual),
        model_result("GRU",     pred_gru,  actual),
        model_result("XGBoost", pred_xgb,  actual),
    ]

    best_model = min(predictions, key=lambda p: p["error"])["model"]

    return jsonify({
        "sample_index"   : idx,
        "total_samples"  : int(len(X_test)),
        "timestamp"      : str(lstm_ts),
        "actual_aqi"     : actual,
        "actual_category": actual_cat,
        "actual_color"   : actual_color,
        "predictions"    : predictions,
        "best_model"     : best_model,
        "context"        : (
            f"Sample from {lstm_ts.strftime('%b %d, %Y at %I:%M %p')}. "
            f"Model was given the preceding 48 hours of data "
            f"and predicted the next hour AQI."
        ),
    })


@app.route("/history", methods=["GET"])
def history():
    """
    Returns actual vs predicted for the full test set for all 3 models.
    Powers the time series chart on the dashboard.
    Downsampled every 6 points to keep response small (~500 points).
    """
    step = 6
    return jsonify({
        "timestamps": [str(t) for t in test_timestamps[::step]],
        "actual"    : [round(float(v), 1) for v in y_true_lstm[::step]],
        "LSTM"      : [round(float(v), 1) for v in y_pred_lstm[::step]],
        "GRU"       : [round(float(v), 1) for v in y_pred_gru[::step]],
        "XGBoost"   : [round(float(v), 1) for v in y_pred_xgb[::step]],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)