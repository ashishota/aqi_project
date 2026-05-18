"""
app.py
======
AQI Forecasting System — Flask entry point.

All heavy lifting lives in separate modules:
  config.py              — city registry, paths, constants
  loader.py              — startup: loads models + test data into CITY_DATA
  utils.py               — metric computation, AQI helpers, route guards
  features/xgb_engineer  — per-city XGBoost feature engineering
  features/lstm_engineer  — LSTM feature lists + build_lstm_features

Endpoints:
  GET /                          → health check; lists loaded cities & models
  GET /cities                    → city metadata (name, station, date range)
  GET /metrics/<city>            → all-model metrics for one city
  GET /metrics/all               → metrics for every city in one response
  GET /predict/random/<city>     → random test sample, all available predictions
  GET /predict/index/<city>/<n>  → specific test sample by index
  GET /history/<city>            → full test-set actual vs predicted (downsampled)

Run locally:
  pip install flask flask-cors pandas numpy joblib xgboost scikit-learn tensorflow
  python app.py
"""

import os
import json
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Windows / Conda environment fixes (TF DLL + OpenMP conflicts)
# ---------------------------------------------------------------------------
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"]              = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"]             = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]              = "3"
    os.environ["KMP_DUPLICATE_LIB_OK"]              = "TRUE"
    os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"

    _conda_base = r"C:\Users\ashis\anaconda3"
    for _d in [
        os.path.join(_conda_base, "Library", "bin"),
        os.path.join(_conda_base, "Library", "lib"),
        os.path.join(_conda_base, "bin"),
        _conda_base,
    ]:
        if os.path.exists(_d):
            try:
                os.add_dll_directory(_d)
            except Exception:
                pass
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

from loader import load_all_cities
from utils  import categorize_aqi, compute_metrics, model_result, city_or_404
from config import HISTORY_DOWNSAMPLE

# ---------------------------------------------------------------------------
# App setup + one-time model loading
# ---------------------------------------------------------------------------

app       = Flask(__name__)
CORS(app)
CITY_DATA = load_all_cities()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    summary = {
        k: {
            "label"       : d["label"],
            "station"     : d["station"],
            "date_range"  : d["date_range"],
            "models_ready": {
                "XGBoost": d["xgb_model"]       is not None,
                "LSTM"   : d["lstm_model"]       is not None,
                "GRU"    : d["gru_model"]        is not None,
                "ARIMA"  : d["arima_metrics"]["MAE"]   is not None,
                "SARIMAX": d["sarimax_metrics"]["MAE"] is not None,
            },
        }
        for k, d in CITY_DATA.items()
    }
    return jsonify({"status": "running", "cities": summary})


@app.route("/cities", methods=["GET"])
def cities():
    """Lightweight city list for the frontend city-switcher."""
    return jsonify([
        {
            "key"       : k,
            "label"     : d["label"],
            "station"   : d["station"],
            "date_range": d["date_range"],
        }
        for k, d in CITY_DATA.items()
    ])


@app.route("/metrics/<city_key>", methods=["GET"])
def metrics(city_key):
    """All-model metrics for one city."""
    d, err = city_or_404(CITY_DATA, city_key)
    if err:
        return err

    results = {}
    if d["xgb_model"]  is not None:
        results["XGBoost"] = compute_metrics(d["y_true_xgb"],  d["y_pred_xgb"])
    if d["lstm_model"] is not None:
        results["LSTM"] = compute_metrics(d["y_true_lstm"], d["y_pred_lstm"])
        results["GRU"]  = compute_metrics(d["y_true_lstm"], d["y_pred_gru"])
    if d["arima_metrics"]["MAE"]   is not None:
        results["ARIMA"]   = d["arima_metrics"]
    if d["sarimax_metrics"]["MAE"] is not None:
        results["SARIMAX"] = d["sarimax_metrics"]

    models  = list(results.keys())
    winners = {}
    if models:
        winners["MAE"]  = min(models, key=lambda m: results[m]["MAE"])
        winners["RMSE"] = min(models, key=lambda m: results[m]["RMSE"])
        winners["R2"]   = max(models, key=lambda m: results[m]["R2"])
        winners["MAPE"] = min(models, key=lambda m: results[m]["MAPE"])
        from collections import Counter
        overall = Counter(winners.values()).most_common(1)[0][0]
    else:
        overall = None

    return jsonify({
        "city"          : city_key,
        "label"         : d["label"],
        "station"       : d["station"],
        "date_range"    : d["date_range"],
        "models"        : results,
        "winners"       : winners,
        "overall_winner": overall,
        "note"          : (
            "All ML metrics on unseen test set (last 15% chronologically). "
            "ARIMA/SARIMAX metrics hardcoded from notebook runs."
        ),
    })


@app.route("/metrics/all", methods=["GET"])
def metrics_all():
    """Metrics for every city — powers the cross-city comparison view."""
    out = {k: json.loads(metrics(k).get_data(as_text=True)) for k in CITY_DATA}
    return jsonify(out)


@app.route("/predict/random/<city_key>", methods=["GET"])
def predict_random(city_key):
    d, err = city_or_404(CITY_DATA, city_key)
    if err:
        return err
    n = len(d["X_test"]) if d["X_test"] is not None else len(d["X_xgb_test"])
    if n == 0:
        return jsonify({"error": "No test data available for this city"}), 500
    return _predict_at(city_key, d, int(np.random.randint(0, n)))


@app.route("/predict/index/<city_key>/<int:idx>", methods=["GET"])
def predict_at_index(city_key, idx):
    d, err = city_or_404(CITY_DATA, city_key)
    if err:
        return err
    n = len(d["X_test"]) if d["X_test"] is not None else len(d["X_xgb_test"])
    if idx < 0 or idx >= n:
        return jsonify({"error": f"Index must be 0 to {n - 1}"}), 400
    return _predict_at(city_key, d, idx)


@app.route("/history/<city_key>", methods=["GET"])
def history(city_key):
    """
    Returns actual vs predicted for the full test set.
    Downsampled every HISTORY_DOWNSAMPLE points to keep payload small (~500 pts).
    """
    d, err = city_or_404(CITY_DATA, city_key)
    if err:
        return err

    s   = HISTORY_DOWNSAMPLE
    out = {"city": city_key, "label": d["label"], "station": d["station"]}

    if d["lstm_model"] is not None:
        out["timestamps"] = [str(t) for t in d["test_timestamps"][::s]]
        out["actual"]     = [round(float(v), 1) for v in d["y_true_lstm"][::s]]
        out["LSTM"]       = [round(float(v), 1) for v in d["y_pred_lstm"][::s]]
        out["GRU"]        = [round(float(v), 1) for v in d["y_pred_gru"][::s]]
    elif d["xgb_model"] is not None:
        out["timestamps"] = [str(t) for t in d["xgb_timestamps"][::s]]
        out["actual"]     = [round(float(v), 1) for v in d["y_true_xgb"][::s]]

    if d["xgb_model"] is not None:
        out["XGBoost"] = [round(float(v), 1) for v in d["y_pred_xgb"][::s]]

    return jsonify(out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _predict_at(city_key: str, d: dict, idx: int):
    predictions = []
    actual      = None
    timestamp   = None

    # LSTM / GRU
    if d["lstm_model"] is not None:
        tscl      = d["tgt_scaler"]
        x_seq     = d["X_test"][idx:idx + 1]
        actual    = float(tscl.inverse_transform([[d["y_test"][idx]]])[0][0])
        timestamp = str(d["test_timestamps"][idx])

        pred_lstm = float(tscl.inverse_transform(
                        d["lstm_model"].predict(x_seq, verbose=0))[0][0])
        pred_gru  = float(tscl.inverse_transform(
                        d["gru_model"].predict(x_seq, verbose=0))[0][0])

        predictions.append(model_result("LSTM", pred_lstm, actual))
        predictions.append(model_result("GRU",  pred_gru,  actual))

    # XGBoost
    if d["xgb_model"] is not None:
        xgb_idx    = min(idx, len(d["X_xgb_test"]) - 1)
        row        = d["X_xgb_test"].iloc[[xgb_idx]]
        pred_xgb   = float(d["xgb_model"].predict(row)[0])
        xgb_actual = float(d["y_xgb_test"][xgb_idx])

        if actual is None:
            actual    = xgb_actual
            timestamp = str(d["xgb_timestamps"][xgb_idx])

        predictions.append(model_result("XGBoost", pred_xgb, actual))

    if actual is None:
        return jsonify({"error": "No models loaded for this city"}), 500

    actual      = round(actual, 1)
    actual_cat, actual_color = categorize_aqi(actual)
    best_model  = (
        min(predictions, key=lambda p: p["error"])["model"]
        if predictions else None
    )

    return jsonify({
        "city"           : city_key,
        "label"          : d["label"],
        "station"        : d["station"],
        "sample_index"   : idx,
        "timestamp"      : timestamp,
        "actual_aqi"     : actual,
        "actual_category": actual_cat,
        "actual_color"   : actual_color,
        "predictions"    : predictions,
        "best_model"     : best_model,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)
