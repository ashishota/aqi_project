"""
config.py
=========
Central configuration: directory paths, city registry, and shared constants.
"""

import os

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
XGB_MODELS_DIR = os.path.join(BASE_DIR, "..", "XGBOOST", "saved_models")
XGB_FEATS_DIR  = os.path.join(BASE_DIR, "..", "XGBOOST", "outputs")
LSTM_BASE_DIR  = os.path.join(BASE_DIR, "..", "LSTM")

# ---------------------------------------------------------------------------
# Shared model constants
# ---------------------------------------------------------------------------

TARGET             = "AQI"
LOOK_BACK_DEFAULT  = 48
TEST_SPLIT         = 0.85      # last (1 - TEST_SPLIT) fraction used for testing
HISTORY_DOWNSAMPLE = 6         # step size when returning full test-set history

# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------

CITY_CONFIG = {
    "anandvihar": {
        "label"   : "Anand Vihar",
        "station" : "Delhi",
        "csv"     : "AnandVihar_AQI_cleaned.csv",
        # XGBoost
        "xgb_model"   : "AnandVihar_xgboost_aqi.pkl",
        "xgb_feats"   : "AnandVihar_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        # LSTM / GRU
        "lstm_model"  : "anandvihar_lstm_model.keras",
        "gru_model"   : "anandvihar_gru_model.keras",
        "feat_scaler" : "anandvihar_feature_scaler.pkl",
        "tgt_scaler"  : "anandvihar_target_scaler.pkl",
        # ARIMA / SARIMAX — metrics hardcoded from notebooks
        "arima_metrics"  : {"MAE": 38.2, "RMSE": 51.4, "R2": 0.71, "MAPE": 14.2},
        "sarimax_metrics": {"MAE": 32.6, "RMSE": 44.1, "R2": 0.78, "MAPE": 11.8},
    },
    "colaba": {
        "label"   : "Colaba",
        "station" : "Mumbai",
        "csv"     : "Colaba_AQI_cleaned.csv",
        "xgb_model"   : "colaba_xgboost_aqi_hourly.pkl",
        "xgb_feats"   : "Colaba_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        "lstm_model"  : "colaba_lstm_model.keras",
        "gru_model"   : "colaba_gru_model.keras",
        "feat_scaler" : "colaba_feature_scaler.pkl",
        "tgt_scaler"  : "colaba_target_scaler.pkl",
        "arima_metrics"  : {"MAE": 29.1, "RMSE": 40.2, "R2": 0.68, "MAPE": 13.4},
        "sarimax_metrics": {"MAE": 24.8, "RMSE": 35.6, "R2": 0.74, "MAPE": 10.9},
    },
    "iitg": {
        "label"   : "IITG",
        "station" : "Guwahati",
        "csv"     : "IITG_AQI_cleaned.csv",
        "xgb_model"   : "iitg_xgboost_aqi.pkl",
        "xgb_feats"   : "IITG_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        "lstm_model"  : "iitg_lstm_model.keras",
        "gru_model"   : "iitg_gru_model.keras",
        "feat_scaler" : "iitg_feature_scaler.pkl",
        "tgt_scaler"  : "iitg_target_scaler.pkl",
        "arima_metrics"  : {"MAE": 35.4, "RMSE": 47.8, "R2": 0.69, "MAPE": 15.1},
        "sarimax_metrics": {"MAE": 30.2, "RMSE": 42.3, "R2": 0.74, "MAPE": 12.6},
    },
    "jayanagar": {
        "label"   : "Jayanagar",
        "station" : "Bangalore",
        "csv"     : "Jayanagar_AQI_cleaned.csv",
        "xgb_model"   : "jayanagar_xgboost_aqi.pkl",
        "xgb_feats"   : "Jayanagar_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        "lstm_model"  : "best_lstm.keras",
        "gru_model"   : "best_gru.keras",
        "feat_scaler" : "jayanagar_feature_scaler.pkl",
        "tgt_scaler"  : "jayanagar_target_scaler.pkl",
        "arima_metrics"  : {"MAE": 22.8, "RMSE": 31.4, "R2": 0.72, "MAPE": 12.3},
        "sarimax_metrics": {"MAE": 19.4, "RMSE": 27.1, "R2": 0.78, "MAPE": 10.2},
    },
    "kodungaiyur": {
        "label"   : "Kodungaiyur",
        "station" : "Chennai",
        "csv"     : "Kodungaiyur_AQI_cleaned.csv",
        "xgb_model"   : "Kodungaiyur_xgboost_aqi.pkl",
        "xgb_feats"   : "Kodungaiyur_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        "lstm_model"  : "kodungaiyur_lstm_model.keras",
        "gru_model"   : "kodungaiyur_gru_model.keras",
        "feat_scaler" : "kodungaiyur_feature_scaler.pkl",
        "tgt_scaler"  : "kodungaiyur_target_scaler.pkl",
        "arima_metrics"  : {"MAE": 31.6, "RMSE": 43.2, "R2": 0.70, "MAPE": 13.8},
        "sarimax_metrics": {"MAE": 27.3, "RMSE": 38.4, "R2": 0.75, "MAPE": 11.4},
    },
    "patia": {
        "label"   : "Patia",
        "station" : "Bhubaneswar",
        "csv"     : "Patia_AQI_cleaned.csv",
        "xgb_model"   : "patia_xgboost_aqi.pkl",
        "xgb_feats"   : "Patia_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        "lstm_model"  : "patia_lstm_model.keras",
        "gru_model"   : "patia_gru_model.keras",
        "feat_scaler" : "patia_feature_scaler.pkl",
        "tgt_scaler"  : "patia_target_scaler.pkl",
        "arima_metrics"  : {"MAE": 26.4, "RMSE": 36.1, "R2": 0.73, "MAPE": 11.9},
        "sarimax_metrics": {"MAE": 22.7, "RMSE": 31.8, "R2": 0.79, "MAPE": 10.0},
    },
}

# Maps city_key → subfolder name inside LSTM_BASE_DIR
LSTM_CITY_FOLDER = {
    "anandvihar" : "Anand_Vihar",
    "colaba"     : "Colaba",
    "iitg"       : "IITG",
    "jayanagar"  : "Jayanagar",
    "kodungaiyur": "Kodungaiyur",
    "patia"      : "Patia",
}
