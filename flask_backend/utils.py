"""
utils.py
========
Pure helper functions shared across routes and the loader.
No Flask or model imports here — keeps this module fast to test in isolation.
"""

import numpy as np
from flask import jsonify
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# AQI categorisation
# ---------------------------------------------------------------------------

def categorize_aqi(v: float):
    """Return (category_label, hex_colour) for a given AQI value."""
    if   v <= 50:  return "Good",         "#00e676"
    elif v <= 100: return "Satisfactory", "#00e5ff"
    elif v <= 200: return "Moderate",     "#ffe066"
    elif v <= 300: return "Poor",         "#ff9f43"
    elif v <= 400: return "Very Poor",    "#ff4757"
    else:          return "Severe",       "#9b59b6"


def health_advice(cat: str) -> str:
    return {
        "Good"        : "Air quality is satisfactory. Safe for all activities.",
        "Satisfactory": "Acceptable. Sensitive individuals should limit prolonged outdoor exertion.",
        "Moderate"    : "Sensitive groups may experience effects. Limit outdoor activity.",
        "Poor"        : "Everyone may experience health effects. Avoid outdoor activity.",
        "Very Poor"   : "Health alert! Everyone should avoid outdoor activity.",
        "Severe"      : "Emergency conditions. Stay indoors. Keep windows closed.",
    }.get(cat, "")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {
        "MAE" : round(mae,  3),
        "RMSE": round(rmse, 3),
        "R2"  : round(r2,   4),
        "MAPE": round(mape, 2),
    }


# ---------------------------------------------------------------------------
# Single-prediction result builder
# ---------------------------------------------------------------------------

def model_result(name: str, predicted: float, actual: float) -> dict:
    predicted  = round(max(0.0, predicted), 1)
    error      = round(abs(predicted - actual), 1)
    pct_error  = round(abs(predicted - actual) / (actual + 1e-8) * 100, 1)
    cat, color = categorize_aqi(predicted)
    return {
        "model"    : name,
        "predicted": predicted,
        "error"    : error,
        "pct_error": pct_error,
        "category" : cat,
        "color"    : color,
        "advice"   : health_advice(cat),
    }


# ---------------------------------------------------------------------------
# Route guard
# ---------------------------------------------------------------------------

def city_or_404(city_data: dict, city_key: str):
    """
    Look up city_key (case-insensitive) in city_data.
    Returns (entry_dict, None) on success or (None, error_response) on failure.
    """
    key = city_key.lower()
    if key not in city_data:
        err = jsonify({
            "error": f"Unknown city '{city_key}'. "
                     f"Valid keys: {list(city_data.keys())}"
        }), 404
        return None, err
    return city_data[key], None
