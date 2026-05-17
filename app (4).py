"""
AQI Forecasting System — Multi-City Flask Backend
===================================================
Project : Intelligent AQI Prediction using Supervised Machine Learning
Cities  : Anand Vihar (Delhi), Colaba (Mumbai), IITG (Guwahati),
          Jayanagar (Bangalore), Kodungaiyur (Chennai), Patia (Bhubaneswar)
Models  : XGBoost (all 6 cities) — live inference
          LSTM / GRU              — loaded when model files are present
          ARIMA / SARIMAX         — metrics hardcoded from notebooks

Endpoints:
  GET /                          → health check, lists loaded cities & models
  GET /cities                    → city metadata (name, station, date range)
  GET /metrics/<city>            → all-model metrics for one city
  GET /metrics/all               → metrics for every city in one response
  GET /predict/random/<city>     → random test sample, all available predictions
  GET /predict/index/<city>/<n>  → specific test sample by index
  GET /history/<city>            → full test-set actual vs predicted (downsampled)

Run:
  pip install flask flask-cors pandas numpy joblib xgboost scikit-learn tensorflow
  python app.py
"""

import os, json, joblib, warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# =============================================================================
# CITY REGISTRY
# All paths are relative to the directory where app.py lives.
# Update BASE_DIR if you keep datasets / models in a different folder.
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
DATA_DIR    = os.path.join(BASE_DIR)          # CSVs sit next to app.py

CITY_CONFIG = {
    "anandvihar": {
        "label"   : "Anand Vihar",
        "station" : "Delhi",
        "csv"     : "AnandVihar_AQI_cleaned.csv",
        # XGBoost
        "xgb_model"   : "AnandVihar_xgboost_aqi.pkl",
        "xgb_feats"   : "AnandVihar_xgboost_features.json",
        "xgb_target"  : "AQI_next",          # column created during XGB feature eng
        # LSTM / GRU  (set to None if files not ready yet)
        "lstm_model"  : "anandvihar_lstm_model.keras",
        "gru_model"   : "anandvihar_gru_model.keras",
        "feat_scaler" : "anandvihar_feature_scaler.pkl",
        "tgt_scaler"  : "anandvihar_target_scaler.pkl",
        # ARIMA / SARIMAX metrics from notebooks (hardcoded)
        "arima_metrics"  : {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
        "sarimax_metrics": {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
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
        "arima_metrics"  : {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
        "sarimax_metrics": {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
    },
    "iitg": {
        "label"   : "IITG",
        "station" : "Guwahati",
        "csv"     : "IITG_AQI_cleaned.csv",
        "xgb_model"   : "iitg_xgboost_aqi.pkl",
        "xgb_feats"   : "IITG_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        # ✅ UPDATED: LSTM/GRU files now available
        "lstm_model"  : "iitg_lstm_model.keras",
        "gru_model"   : "iitg_gru_model.keras",
        "feat_scaler" : "iitg_feature_scaler.pkl",
        "tgt_scaler"  : "iitg_target_scaler.pkl",
        "arima_metrics"  : {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
        "sarimax_metrics": {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
    },
    "jayanagar": {
        "label"   : "Jayanagar",
        "station" : "Bangalore",
        "csv"     : "Jayanagar_AQI_cleaned.csv",
        "xgb_model"   : "jayanagar_xgboost_aqi.pkl",
        "xgb_feats"   : "Jayanagar_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        # ✅ UPDATED: LSTM/GRU files now available
        "lstm_model"  : "jayanagar_lstm_model.keras",
        "gru_model"   : "jayanagar_gru_model.keras",
        "feat_scaler" : "jayanagar_feature_scaler.pkl",
        "tgt_scaler"  : "jayanagar_target_scaler.pkl",
        "arima_metrics"  : {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
        "sarimax_metrics": {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
    },
    "kodungaiyur": {
        "label"   : "Kodungaiyur",
        "station" : "Chennai",
        "csv"     : "Kodungaiyur_AQI_cleaned.csv",
        "xgb_model"   : "Kodungaiyur_xgboost_aqi.pkl",
        "xgb_feats"   : "Kodungaiyur_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        # ✅ UPDATED: LSTM/GRU files now available
        "lstm_model"  : "kodungaiyur_lstm_model.keras",
        "gru_model"   : "kodungaiyur_gru_model.keras",
        "feat_scaler" : "kodungaiyur_feature_scaler.pkl",
        "tgt_scaler"  : "kodungaiyur_target_scaler.pkl",
        "arima_metrics"  : {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
        "sarimax_metrics": {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
    },
    "patia": {
        "label"   : "Patia",
        "station" : "Bhubaneswar",
        "csv"     : "Patia_AQI_cleaned.csv",
        "xgb_model"   : "patia_xgboost_aqi.pkl",
        "xgb_feats"   : "Patia_xgboost_features.json",
        "xgb_target"  : "AQI_next",
        # ✅ UPDATED: LSTM/GRU files now available
        "lstm_model"  : "patia_lstm_model.keras",
        "gru_model"   : "patia_gru_model.keras",
        "feat_scaler" : "patia_feature_scaler.pkl",
        "tgt_scaler"  : "patia_target_scaler.pkl",
        "arima_metrics"  : {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
        "sarimax_metrics": {"MAE": None, "RMSE": None, "R2": None, "MAPE": None},
    },
}

# =============================================================================
# LSTM FEATURE LISTS — per city
# Sourced from each city's iitg_features.json (or equivalent) at training time.
# look_back is also city-specific; defaults to LOOK_BACK_DEFAULT if not set.
# =============================================================================

LOOK_BACK_DEFAULT = 48
TARGET            = "AQI"

LSTM_FEATURES = {
    "anandvihar": [
        'PM25','PM10','NO2','CO','AQI_lag1','AQI_lag2',
        'AQI_roll3_mean','AQI_roll6_mean','O3','SO2','NH3',
        'NOx_total','hour_sin','hour_cos','AQI_roll12_mean',
        'PM25_lag1','RH','NO','WS','AT','SR','Toluene_Benzene_ratio',
    ],
    # ✅ UPDATED: IITG feature list from iitg_features.json
    "iitg": [
        'PM25','PM10','CO','NO2','AQI_lag1','AQI_lag2',
        'AQI_roll3_mean','AQI_roll6_mean','O3','SO2','NH3',
        'month_sin','month_cos','hour_sin','hour_cos',
        'RH','WS','SR','BP','WD_sin','WD_cos',
        'AQI_roll12_mean','AQI_roll24_mean','AQI_lag24',
        'PM25_lag1','RH_PM25_interact','NO','NOx','VOC_total',
        'RF_flag','RF_lag1',
    ],
    # ✅ UPDATED: Patia feature list extracted from patia_feature_scaler.pkl
    # Input shape confirmed (None, 48, 34) from patia_lstm_model.keras
    "patia": [
        'PM25','PM10','Benzene','Eth-Benzene','MP-Xylene',
        'lag_1','lag_6','lag_24',
        'roll_mean_6','roll_mean_24','ema_24',
        'NO2','NOx','SO2','O3','NH3','CO',
        'AT','RH','WS','RF','SR',
        'hour_sin','hour_cos','month_sin','month_cos','dow_sin','dow_cos',
        'is_weekend','ema_6','roll_std_24','diff_1',
        'AT_RH_stability','BTEX_total',
    ],
    # ✅ UPDATED: Kodungaiyur feature list extracted from kodungaiyur_feature_scaler.pkl
    # Input shape confirmed (None, 48, 27) from kodungaiyur_lstm_model.keras
    "kodungaiyur": [
        'PM25','PM10',
        'lag_1','lag_6','lag_24',
        'roll_mean_6','roll_mean_24','ema_24',
        'SO2','NO2','NH3','CO','O3',
        'RH','WS','RF','SR',
        'hour_sin','hour_cos','month_sin','month_cos','dow_sin','dow_cos',
        'is_weekend','ema_6','roll_std_24','diff_1',
    ],
    # ✅ UPDATED: Colaba feature list extracted from colaba_feature_scaler.pkl
    # Input shape confirmed (None, 48, 22) from colaba_lstm_model.keras
    "colaba": [
        'PM25','PM10','NO2','CO',
        'AQI_lag1','AQI_lag2','AQI_roll3_mean','AQI_roll6_mean','AQI_roll12_mean',
        'O3','SO2','NH3','NOx_total',
        'hour_sin','hour_cos',
        'PM25_lag1','RH','NO','WS','AT','SR',
        'EthBenzene_MPXylene_ratio',
    ],
    # ✅ UPDATED: Jayanagar feature list extracted from jayanagar_feature_scaler.pkl
    # Input shape confirmed (None, 48, 26) from jayanagar_lstm_model.keras
    "jayanagar": [
        'PM25','PM10','NO2','CO',
        'AQI_lag1','AQI_lag2','AQI_lag3',
        'AQI_roll3_mean','AQI_roll6_mean','AQI_roll12_mean','AQI_roll24_mean',
        'O3','SO2','NH3','NOx_total',
        'RH','WD_sin','WD_cos','BP',
        'month_sin','month_cos','hour_sin','hour_cos',
        'PM25_lag1','NO','AT',
    ],
}

# Per-city look_back override (defaults to LOOK_BACK_DEFAULT = 48 if not listed)
LSTM_LOOK_BACK = {
    "anandvihar": 48,
    "iitg"      : 48,   # confirmed from iitg_features.json
    "patia"        : 48,   # confirmed from model input shape (None, 48, 34)
    "kodungaiyur"  : 48,   # confirmed from model input shape (None, 48, 27)
    "colaba"       : 48,   # confirmed from model input shape (None, 48, 22)
    "jayanagar"    : 48,   # confirmed from model input shape (None, 48, 26)
}

# =============================================================================
# FEATURE ENGINEERING — per-city XGBoost
# Each function takes a raw DataFrame (Timestamp as index) and returns a
# fully-engineered DataFrame ready for XGBoost, with an AQI_next target col.
# =============================================================================

def _season(month):
    """Map month → 0-based season integer (Winter=0, Spring=1, Summer=2, Monsoon=3, Autumn=4)."""
    return {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2, 8:3,9:3,10:3, 11:4}.get(month, 0)

def _aqi_cat_enc(aqi):
    if aqi <= 50:   return 0
    if aqi <= 100:  return 1
    if aqi <= 200:  return 2
    if aqi <= 300:  return 3
    if aqi <= 400:  return 4
    return 5

def engineer_xgb_anandvihar(df):
    fe = df.copy()
    fe["AQI_next"]    = fe[TARGET].shift(-1)
    fe["AQI_lag_72h"] = fe[TARGET].shift(72)
    fe["AQI_lag_168h"]= fe[TARGET].shift(168)
    doy = fe.index.dayofyear
    fe["DayOfYear"]   = doy
    fe["doy_sin"]     = np.sin(2 * np.pi * doy / 365)
    fe["doy_cos"]     = np.cos(2 * np.pi * doy / 365)
    return fe.dropna(subset=["AQI_next","AQI_lag_72h","AQI_lag_168h"])

def engineer_xgb_colaba(df):
    fe = df.copy()
    fe["AQI_next"] = fe[TARGET].shift(-1)
    for h in [1,2,3,6,12,24,48]:
        fe[f"lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3,6,24]:
        fe[f"roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    idx = fe.index
    fe["hour"]        = idx.hour
    fe["day_of_week"] = idx.dayofweek
    fe["month"]       = idx.month
    fe["day_of_year"] = idx.dayofyear
    fe["hour_sin"]    = np.sin(2*np.pi*idx.hour/24)
    fe["hour_cos"]    = np.cos(2*np.pi*idx.hour/24)
    fe["month_sin"]   = np.sin(2*np.pi*idx.month/12)
    fe["month_cos"]   = np.cos(2*np.pi*idx.month/12)
    fe["dow_sin"]     = np.sin(2*np.pi*idx.dayofweek/7)
    fe["dow_cos"]     = np.cos(2*np.pi*idx.dayofweek/7)
    doy = idx.dayofyear
    fe["doy_sin"]     = np.sin(2*np.pi*doy/365)
    fe["doy_cos"]     = np.cos(2*np.pi*doy/365)
    fe["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["season_enc"]  = idx.month.map(_season)
    fe["aqi_cat_enc"] = fe[TARGET].map(_aqi_cat_enc)
    lag_cols = [f"lag_{h}h" for h in [1,2,3,6,12,24,48]]
    return fe.dropna(subset=["AQI_next"] + lag_cols)

def engineer_xgb_iitg(df):
    fe = df.copy()
    fe["AQI_next"] = fe[TARGET].shift(-1)
    # AT is missing from IITG CSV — fill with 0 as placeholder
    if "AT" not in fe.columns:
        fe["AT"] = 0.0
    for h in [1,2,3,6,12,24,48]:
        fe[f"lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3,6,24]:
        fe[f"roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    idx = fe.index
    fe["Hour"]      = idx.hour
    fe["DayOfWeek"] = idx.dayofweek
    fe["Month"]     = idx.month
    fe["DayOfYear"] = idx.dayofyear
    fe["Quarter"]   = idx.quarter
    fe["IsWeekend"] = (idx.dayofweek >= 5).astype(int)
    fe["Season_Enc"]= idx.month.map(_season)
    fe["Hour_sin"]  = np.sin(2*np.pi*idx.hour/24)
    fe["Hour_cos"]  = np.cos(2*np.pi*idx.hour/24)
    fe["Month_sin"] = np.sin(2*np.pi*idx.month/12)
    fe["Month_cos"] = np.cos(2*np.pi*idx.month/12)
    fe["DoW_sin"]   = np.sin(2*np.pi*idx.dayofweek/7)
    fe["DoW_cos"]   = np.cos(2*np.pi*idx.dayofweek/7)
    lag_cols = [f"lag_{h}h" for h in [1,2,3,6,12,24,48]]
    return fe.dropna(subset=["AQI_next"] + lag_cols)

def engineer_xgb_jayanagar(df):
    fe = df.copy()
    fe["AQI_next"] = fe[TARGET].shift(-1)
    # Daily lags (Jayanagar used daily data)
    for lag in [1,2,3,4,5,6,7,9,14,21,30]:
        fe[f"lag_{lag}"] = fe[TARGET].shift(lag)
    for w in [7,14,30]:
        fe[f"roll_mean_{w}"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}"]  = fe[TARGET].shift(1).rolling(w).std()
    fe["PM25_lag1"] = fe["PM25"].shift(1)
    fe["PM10_lag1"] = fe["PM10"].shift(1)
    fe["CO_lag1"]   = fe["CO"].shift(1)
    fe["NOx_lag1"]  = fe["NOx"].shift(1)
    fe["NO2_lag1"]  = fe["NO2"].shift(1)
    idx = fe.index
    fe["month"]       = idx.month
    fe["day_of_week"] = idx.dayofweek
    fe["day_of_year"] = idx.dayofyear
    fe["quarter"]     = idx.quarter
    fe["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["month_sin"]   = np.sin(2*np.pi*idx.month/12)
    fe["month_cos"]   = np.cos(2*np.pi*idx.month/12)
    fe["dow_sin"]     = np.sin(2*np.pi*idx.dayofweek/7)
    fe["dow_cos"]     = np.cos(2*np.pi*idx.dayofweek/7)
    doy = idx.dayofyear
    fe["doy_sin"]     = np.sin(2*np.pi*doy/365)
    fe["doy_cos"]     = np.cos(2*np.pi*doy/365)
    return fe.dropna(subset=["AQI_next","lag_30","roll_mean_30"])

def engineer_xgb_kodungaiyur(df):
    fe = df.copy()
    fe["AQI_next"] = fe[TARGET].shift(-1)
    for h in [1,2,3,6,12,18,24,36,48]:
        fe[f"lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3,6,24]:
        fe[f"roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    idx = fe.index
    fe["hour"]        = idx.hour
    fe["day_of_week"] = idx.dayofweek
    fe["month"]       = idx.month
    fe["day_of_year"] = idx.dayofyear
    fe["quarter"]     = idx.quarter
    fe["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["hour_sin"]    = np.sin(2*np.pi*idx.hour/24)
    fe["hour_cos"]    = np.cos(2*np.pi*idx.hour/24)
    fe["month_sin"]   = np.sin(2*np.pi*idx.month/12)
    fe["month_cos"]   = np.cos(2*np.pi*idx.month/12)
    fe["dow_sin"]     = np.sin(2*np.pi*idx.dayofweek/7)
    fe["dow_cos"]     = np.cos(2*np.pi*idx.dayofweek/7)
    doy = idx.dayofyear
    fe["doy_sin"]     = np.sin(2*np.pi*doy/365)
    fe["doy_cos"]     = np.cos(2*np.pi*doy/365)
    fe["season"]      = idx.month.map(_season)
    return fe.dropna(subset=["AQI_next","lag_48h"])

def engineer_xgb_patia(df):
    fe = df.copy()
    fe["AQI_next"] = fe[TARGET].shift(-1)
    fe["AQI_Category_Enc"] = fe[TARGET].map(_aqi_cat_enc)
    idx = fe.index
    fe["Hour"]      = idx.hour
    fe["DayOfWeek"] = idx.dayofweek
    fe["Month"]     = idx.month
    fe["Year"]      = idx.year
    fe["DayOfYear"] = idx.dayofyear
    fe["Quarter"]   = idx.quarter
    fe["IsWeekend"] = (idx.dayofweek >= 5).astype(int)
    fe["Season_Enc"]= idx.month.map(_season)
    for h in [1,2,3,6,12,24,48]:
        fe[f"AQI_lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3,6,24]:
        fe[f"AQI_roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"AQI_roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    fe["Hour_sin"]  = np.sin(2*np.pi*idx.hour/24)
    fe["Hour_cos"]  = np.cos(2*np.pi*idx.hour/24)
    fe["Month_sin"] = np.sin(2*np.pi*idx.month/12)
    fe["Month_cos"] = np.cos(2*np.pi*idx.month/12)
    fe["DOW_sin"]   = np.sin(2*np.pi*idx.dayofweek/7)
    fe["DOW_cos"]   = np.cos(2*np.pi*idx.dayofweek/7)
    return fe.dropna(subset=["AQI_next","AQI_lag_48h"])

XGB_ENGINEER = {
    "anandvihar"  : engineer_xgb_anandvihar,
    "colaba"      : engineer_xgb_colaba,
    "iitg"        : engineer_xgb_iitg,
    "jayanagar"   : engineer_xgb_jayanagar,
    "kodungaiyur" : engineer_xgb_kodungaiyur,
    "patia"       : engineer_xgb_patia,
}

# =============================================================================
# LSTM FEATURE ENGINEERING — shared utility
# Builds derived columns that LSTM models expect (lags, rolling means, cyclical
# encodings, interaction terms).  Each city's LSTM_FEATURES list then selects
# only the columns that were actually used during training.
# =============================================================================

def build_lstm_features(df_raw, city_key):
    """
    Applies all possible LSTM-style feature transformations to df_raw.
    Returns a DataFrame containing the union of all features any city might need.
    Missing source columns are silently skipped (filled with 0).
    """
    fe = df_raw.copy()

    # ── Lag features ──────────────────────────────────────────────────────────
    fe["AQI_lag1"]  = fe[TARGET].shift(1)
    fe["AQI_lag2"]  = fe[TARGET].shift(2)
    fe["AQI_lag3"]  = fe[TARGET].shift(3)   # needed by Jayanagar
    fe["AQI_lag24"] = fe[TARGET].shift(24)

    if "PM25" in fe.columns:
        fe["PM25_lag1"] = fe["PM25"].shift(1)
    else:
        fe["PM25_lag1"] = 0.0

    # ── Rolling means ─────────────────────────────────────────────────────────
    for w in [3, 6, 12, 24]:
        fe[f"AQI_roll{w}_mean"] = fe[TARGET].shift(1).rolling(w).mean()

    # ── Cyclical time encodings ───────────────────────────────────────────────
    fe["hour_sin"]  = np.sin(2 * np.pi * fe.index.hour / 24)
    fe["hour_cos"]  = np.cos(2 * np.pi * fe.index.hour / 24)
    fe["month_sin"] = np.sin(2 * np.pi * fe.index.month / 12)
    fe["month_cos"] = np.cos(2 * np.pi * fe.index.month / 12)

    # ── Wind direction cyclical encoding ──────────────────────────────────────
    if "WD" in fe.columns:
        fe["WD_sin"] = np.sin(2 * np.pi * fe["WD"] / 360)
        fe["WD_cos"] = np.cos(2 * np.pi * fe["WD"] / 360)
    else:
        fe["WD_sin"] = 0.0
        fe["WD_cos"] = 0.0

    # ── Interaction / derived pollutant features ──────────────────────────────
    if "NO" in fe.columns and "NO2" in fe.columns:
        fe["NOx_total"] = fe["NO"] + fe["NO2"]
    else:
        fe["NOx_total"] = 0.0

    if "Toluene" in fe.columns and "Benzene" in fe.columns:
        fe["Toluene_Benzene_ratio"] = fe["Toluene"] / (fe["Benzene"] + 1e-6)
    else:
        fe["Toluene_Benzene_ratio"] = 0.0

    # ── IITG-specific features ────────────────────────────────────────────────
    # RH_PM25_interact: humidity × fine particulate interaction
    if "RH" in fe.columns and "PM25" in fe.columns:
        fe["RH_PM25_interact"] = fe["RH"] * fe["PM25"]
    else:
        fe["RH_PM25_interact"] = 0.0

    # VOC_total: sum of available VOC species
    voc_cols = [c for c in ["Toluene", "Benzene", "Xylene", "Ethylbenzene"] if c in fe.columns]
    fe["VOC_total"] = fe[voc_cols].sum(axis=1) if voc_cols else 0.0

    # RF_flag / RF_lag1: rainfall indicator (1 if RF > 0)
    if "RF" in fe.columns:
        fe["RF_flag"] = (fe["RF"] > 0).astype(float)
        fe["RF_lag1"] = fe["RF_flag"].shift(1)
    else:
        fe["RF_flag"] = 0.0
        fe["RF_lag1"] = 0.0

    # NOx column (some cities log this directly; others derive from NO+NO2)
    if "NOx" not in fe.columns:
        fe["NOx"] = fe.get("NOx_total", 0.0)

    # BP (barometric pressure) — pass-through if present, else 0
    if "BP" not in fe.columns:
        fe["BP"] = 0.0

    # ── Patia-specific features ───────────────────────────────────────────────
    # AQI lag/roll using Patia's naming convention (lag_1, lag_6, lag_24)
    fe["lag_1"]  = fe[TARGET].shift(1)
    fe["lag_6"]  = fe[TARGET].shift(6)
    fe["lag_24"] = fe[TARGET].shift(24)

    # Rolling stats with Patia naming (roll_mean_6, roll_mean_24, roll_std_24)
    fe["roll_mean_6"]  = fe[TARGET].shift(1).rolling(6).mean()
    fe["roll_mean_24"] = fe[TARGET].shift(1).rolling(24).mean()
    fe["roll_std_24"]  = fe[TARGET].shift(1).rolling(24).std()

    # Exponential moving averages (ema_6, ema_24)
    fe["ema_6"]  = fe[TARGET].shift(1).ewm(span=6,  adjust=False).mean()
    fe["ema_24"] = fe[TARGET].shift(1).ewm(span=24, adjust=False).mean()

    # First difference (diff_1): rate of change in AQI
    fe["diff_1"] = fe[TARGET].diff(1)

    # Day-of-week cyclical encoding (dow_sin, dow_cos)
    fe["dow_sin"] = np.sin(2 * np.pi * fe.index.dayofweek / 7)
    fe["dow_cos"] = np.cos(2 * np.pi * fe.index.dayofweek / 7)

    # Weekend flag
    fe["is_weekend"] = (fe.index.dayofweek >= 5).astype(int)

    # AT_RH_stability: temperature-humidity stability index (AT / (RH + 1))
    if "AT" in fe.columns and "RH" in fe.columns:
        fe["AT_RH_stability"] = fe["AT"] / (fe["RH"] + 1)
    else:
        fe["AT_RH_stability"] = 0.0

    # BTEX_total: sum of Benzene, Toluene, Eth-Benzene, MP-Xylene (BTEX group)
    btex_cols = [c for c in ["Benzene", "Toluene", "Eth-Benzene", "MP-Xylene"] if c in fe.columns]
    fe["BTEX_total"] = fe[btex_cols].sum(axis=1) if btex_cols else 0.0

    # ── Colaba-specific features ──────────────────────────────────────────────
    # EthBenzene_MPXylene_ratio: Ethylbenzene / (MP-Xylene + epsilon)
    if "Eth-Benzene" in fe.columns and "MP-Xylene" in fe.columns:
        fe["EthBenzene_MPXylene_ratio"] = fe["Eth-Benzene"] / (fe["MP-Xylene"] + 1e-6)
    else:
        fe["EthBenzene_MPXylene_ratio"] = 0.0

    fe.dropna(inplace=True)
    return fe

# =============================================================================
# STARTUP — load everything once
# =============================================================================

CITY_DATA = {}   # populated below

print("=" * 60)
print("  AQI Multi-City Forecasting System — Loading...")
print("=" * 60)

for city_key, cfg in CITY_CONFIG.items():
    print(f"\n[{cfg['label']}, {cfg['station']}]")
    entry = {
        "label"   : cfg["label"],
        "station" : cfg["station"],
        # XGBoost
        "xgb_model"      : None,
        "xgb_feat_cols"  : None,
        "X_xgb_test"     : None,
        "y_xgb_test"     : None,
        "xgb_timestamps" : None,
        "y_pred_xgb"     : None,
        "y_true_xgb"     : None,
        # LSTM / GRU
        "lstm_model"     : None,
        "gru_model"      : None,
        "feat_scaler"    : None,
        "tgt_scaler"     : None,
        "X_test"         : None,
        "y_test"         : None,
        "test_timestamps": None,
        "y_pred_lstm"    : None,
        "y_pred_gru"     : None,
        "y_true_lstm"    : None,
        # static
        "arima_metrics"  : cfg["arima_metrics"],
        "sarimax_metrics": cfg["sarimax_metrics"],
        "date_range"     : None,
    }

    # ── Load raw CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, cfg["csv"])
    if not os.path.exists(csv_path):
        print(f"  ⚠️  CSV not found: {csv_path} — skipping city")
        continue

    df_raw = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df_raw = df_raw.set_index("Timestamp").sort_index()
    entry["date_range"] = f"{df_raw.index.min().date()} → {df_raw.index.max().date()}"
    entry["df_raw"]     = df_raw
    print(f"  ✅ CSV loaded  {df_raw.shape}  {entry['date_range']}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_path  = os.path.join(MODELS_DIR, cfg["xgb_model"])
    feat_path = os.path.join(MODELS_DIR, cfg["xgb_feats"])

    if os.path.exists(xgb_path) and os.path.exists(feat_path):
        try:
            xgb_model     = joblib.load(xgb_path)
            xgb_feat_cols = json.load(open(feat_path))
            engineer_fn   = XGB_ENGINEER[city_key]
            df_eng        = engineer_fn(df_raw)

            n           = len(df_eng)
            test_start  = int(n * 0.85)
            df_test     = df_eng.iloc[test_start:]

            available   = [c for c in xgb_feat_cols if c in df_test.columns]
            missing     = [c for c in xgb_feat_cols if c not in df_test.columns]
            if missing:
                print(f"  ⚠️  XGB missing cols (will be skipped): {missing}")

            X_xgb_test  = df_test[available]
            y_xgb_test  = df_test["AQI_next"].values
            y_pred_xgb  = xgb_model.predict(X_xgb_test)

            entry["xgb_model"]      = xgb_model
            entry["xgb_feat_cols"]  = available
            entry["X_xgb_test"]     = X_xgb_test
            entry["y_xgb_test"]     = y_xgb_test
            entry["xgb_timestamps"] = df_test.index
            entry["y_pred_xgb"]     = y_pred_xgb
            entry["y_true_xgb"]     = y_xgb_test
            print(f"  ✅ XGBoost loaded  test={len(X_xgb_test):,} samples")
        except Exception as e:
            print(f"  ❌ XGBoost failed: {e}")
    else:
        print(f"  ⚠️  XGBoost model/features not found — skipping")

    # ── LSTM & GRU ────────────────────────────────────────────────────────────
    lstm_path  = os.path.join(MODELS_DIR, cfg["lstm_model"])  if cfg["lstm_model"]  else None
    gru_path   = os.path.join(MODELS_DIR, cfg["gru_model"])   if cfg["gru_model"]   else None
    fscl_path  = os.path.join(MODELS_DIR, cfg["feat_scaler"]) if cfg["feat_scaler"] else None
    tscl_path  = os.path.join(MODELS_DIR, cfg["tgt_scaler"])  if cfg["tgt_scaler"]  else None

    dl_files_ready = (
        lstm_path and gru_path and fscl_path and tscl_path and
        os.path.exists(lstm_path) and os.path.exists(gru_path) and
        os.path.exists(fscl_path) and os.path.exists(tscl_path)
    )

    if dl_files_ready:
        try:
            from tensorflow.keras.models import load_model
            lstm_model    = load_model(lstm_path)
            gru_model     = load_model(gru_path)
            feat_scaler   = joblib.load(fscl_path)
            target_scaler = joblib.load(tscl_path)

            lstm_feats = LSTM_FEATURES.get(city_key, [])
            look_back  = LSTM_LOOK_BACK.get(city_key, LOOK_BACK_DEFAULT)

            if not lstm_feats:
                print(f"  ⚠️  No LSTM_FEATURES defined for {city_key} — skipping DL models")
                raise ValueError(f"LSTM_FEATURES not defined for {city_key}")

            # Apply unified feature engineering
            fe = build_lstm_features(df_raw, city_key)

            available_feats = [f for f in lstm_feats if f in fe.columns]
            missing_feats   = [f for f in lstm_feats if f not in fe.columns]
            if missing_feats:
                print(f"  ⚠️  LSTM missing source cols: {missing_feats}")

            data = fe[available_feats + [TARGET]].copy()

            scaled_features = feat_scaler.transform(data[available_feats])
            scaled_target   = target_scaler.transform(data[[TARGET]])
            scaled_data     = np.hstack([scaled_features, scaled_target])

            def create_sequences(d, lb):
                X, y = [], []
                for i in range(lb, len(d)):
                    X.append(d[i-lb:i, :-1])
                    y.append(d[i, -1])
                return np.array(X), np.array(y)

            X_all, y_all = create_sequences(scaled_data, look_back)
            n         = len(X_all)
            val_end   = int(n * 0.85)
            X_test    = X_all[val_end:]
            y_test    = y_all[val_end:]
            ts_all    = data.index[look_back:]
            test_ts   = ts_all[val_end:]

            y_true_lstm = target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
            y_pred_lstm = target_scaler.inverse_transform(
                            lstm_model.predict(X_test, verbose=0)).flatten()
            y_pred_gru  = target_scaler.inverse_transform(
                            gru_model.predict(X_test, verbose=0)).flatten()

            entry.update({
                "lstm_model"     : lstm_model,
                "gru_model"      : gru_model,
                "feat_scaler"    : feat_scaler,
                "tgt_scaler"     : target_scaler,
                "X_test"         : X_test,
                "y_test"         : y_test,
                "test_timestamps": test_ts,
                "y_pred_lstm"    : y_pred_lstm,
                "y_pred_gru"     : y_pred_gru,
                "y_true_lstm"    : y_true_lstm,
            })
            print(f"  ✅ LSTM + GRU loaded  test={len(X_test):,} samples  look_back={look_back}")
        except Exception as e:
            print(f"  ❌ LSTM/GRU failed: {e}")
    else:
        print(f"  ⏳ LSTM/GRU model files not present yet — will activate on next restart")

    CITY_DATA[city_key] = entry

print(f"\n{'='*60}")
print(f"  ✅ Ready. Cities loaded: {list(CITY_DATA.keys())}")
print(f"{'='*60}\n")

# =============================================================================
# HELPERS
# =============================================================================

def categorize_aqi(v):
    if   v <= 50:  return "Good",         "#00e676"
    elif v <= 100: return "Satisfactory", "#00e5ff"
    elif v <= 200: return "Moderate",     "#ffe066"
    elif v <= 300: return "Poor",         "#ff9f43"
    elif v <= 400: return "Very Poor",    "#ff4757"
    else:          return "Severe",       "#9b59b6"

def health_advice(cat):
    return {
        "Good"        : "Air quality is satisfactory. Safe for all activities.",
        "Satisfactory": "Acceptable. Sensitive individuals should limit prolonged outdoor exertion.",
        "Moderate"    : "Sensitive groups may experience effects. Limit outdoor activity.",
        "Poor"        : "Everyone may experience health effects. Avoid outdoor activity.",
        "Very Poor"   : "Health alert! Everyone should avoid outdoor activity.",
        "Severe"      : "Emergency conditions. Stay indoors. Keep windows closed.",
    }.get(cat, "")

def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    return {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4), "MAPE": round(mape,2)}

def model_result(name, predicted, actual):
    predicted  = round(max(0, predicted), 1)
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

def city_or_404(city_key):
    key = city_key.lower()
    if key not in CITY_DATA:
        return None, (jsonify({"error": f"Unknown city '{city_key}'. "
                               f"Valid keys: {list(CITY_DATA.keys())}"}), 404)
    return CITY_DATA[key], None

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def health():
    summary = {}
    for k, d in CITY_DATA.items():
        summary[k] = {
            "label"       : d["label"],
            "station"     : d["station"],
            "date_range"  : d["date_range"],
            "models_ready": {
                "XGBoost": d["xgb_model"] is not None,
                "LSTM"    : d["lstm_model"] is not None,
                "GRU"     : d["gru_model"] is not None,
                "ARIMA"   : d["arima_metrics"]["MAE"] is not None,
                "SARIMAX" : d["sarimax_metrics"]["MAE"] is not None,
            }
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
    d, err = city_or_404(city_key)
    if err: return err

    results = {}

    if d["xgb_model"] is not None:
        results["XGBoost"] = compute_metrics(d["y_true_xgb"], d["y_pred_xgb"])

    if d["lstm_model"] is not None:
        results["LSTM"] = compute_metrics(d["y_true_lstm"], d["y_pred_lstm"])
        results["GRU"]  = compute_metrics(d["y_true_lstm"], d["y_pred_gru"])

    if d["arima_metrics"]["MAE"] is not None:
        results["ARIMA"] = d["arima_metrics"]

    if d["sarimax_metrics"]["MAE"] is not None:
        results["SARIMAX"] = d["sarimax_metrics"]

    # Per-metric winners (only among models that have results)
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
        "note"          : "All ML metrics on unseen test set (last 15% chronologically)."
                          " ARIMA/SARIMAX metrics hardcoded from notebook runs.",
    })


@app.route("/metrics/all", methods=["GET"])
def metrics_all():
    """Metrics for every city — powers the cross-city comparison view."""
    out = {}
    for k in CITY_DATA:
        resp = metrics(k)
        out[k] = json.loads(resp.get_data(as_text=True))
    return jsonify(out)


@app.route("/predict/random/<city_key>", methods=["GET"])
def predict_random(city_key):
    d, err = city_or_404(city_key)
    if err: return err
    # Prefer LSTM test-set length, fall back to XGBoost
    n = len(d["X_test"]) if d["X_test"] is not None else len(d["X_xgb_test"])
    if n == 0:
        return jsonify({"error": "No test data available for this city"}), 500
    idx = int(np.random.randint(0, n))
    return _predict_at(city_key, d, idx)


@app.route("/predict/index/<city_key>/<int:idx>", methods=["GET"])
def predict_at_index(city_key, idx):
    d, err = city_or_404(city_key)
    if err: return err
    n = len(d["X_test"]) if d["X_test"] is not None else len(d["X_xgb_test"])
    if idx < 0 or idx >= n:
        return jsonify({"error": f"Index must be 0 to {n-1}"}), 400
    return _predict_at(city_key, d, idx)


def _predict_at(city_key, d, idx):
    predictions = []
    actual      = None
    timestamp   = None

    # ── LSTM / GRU ────────────────────────────────────────────────────────────
    if d["lstm_model"] is not None:
        tscl     = d["tgt_scaler"]
        x_seq    = d["X_test"][idx:idx+1]
        actual   = float(tscl.inverse_transform([[d["y_test"][idx]]])[0][0])
        timestamp= str(d["test_timestamps"][idx])

        pred_lstm = float(tscl.inverse_transform(
                        d["lstm_model"].predict(x_seq, verbose=0))[0][0])
        pred_gru  = float(tscl.inverse_transform(
                        d["gru_model"].predict(x_seq, verbose=0))[0][0])

        predictions.append(model_result("LSTM", pred_lstm, actual))
        predictions.append(model_result("GRU",  pred_gru,  actual))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if d["xgb_model"] is not None:
        xgb_idx = idx if idx < len(d["X_xgb_test"]) else len(d["X_xgb_test"]) - 1
        row      = d["X_xgb_test"].iloc[[xgb_idx]]
        pred_xgb = float(d["xgb_model"].predict(row)[0])
        xgb_actual = float(d["y_xgb_test"][xgb_idx])

        if actual is None:
            actual    = xgb_actual
            timestamp = str(d["xgb_timestamps"][xgb_idx])

        predictions.append(model_result("XGBoost", pred_xgb, actual))

    if actual is None:
        return jsonify({"error": "No models loaded for this city"}), 500

    actual      = round(actual, 1)
    actual_cat, actual_color = categorize_aqi(actual)
    best_model  = min(predictions, key=lambda p: p["error"])["model"] if predictions else None

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


@app.route("/history/<city_key>", methods=["GET"])
def history(city_key):
    """
    Returns actual vs predicted for the full test set.
    Downsampled every 6 points to keep payload small (~500 pts).
    """
    d, err = city_or_404(city_key)
    if err: return err

    step = 6
    out  = {"city": city_key, "label": d["label"], "station": d["station"]}

    if d["lstm_model"] is not None:
        out["timestamps"] = [str(t) for t in d["test_timestamps"][::step]]
        out["actual"]     = [round(float(v),1) for v in d["y_true_lstm"][::step]]
        out["LSTM"]       = [round(float(v),1) for v in d["y_pred_lstm"][::step]]
        out["GRU"]        = [round(float(v),1) for v in d["y_pred_gru"][::step]]
    elif d["xgb_model"] is not None:
        out["timestamps"] = [str(t) for t in d["xgb_timestamps"][::step]]
        out["actual"]     = [round(float(v),1) for v in d["y_true_xgb"][::step]]

    if d["xgb_model"] is not None:
        out["XGBoost"] = [round(float(v),1) for v in d["y_pred_xgb"][::step]]

    return jsonify(out)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
