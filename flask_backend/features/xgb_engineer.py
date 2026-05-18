"""
features/xgb_engineer.py
========================
Per-city XGBoost feature-engineering functions and the dispatch map XGB_ENGINEER.

Each function receives a raw DataFrame (DatetimeIndex) and returns a new
DataFrame with the engineered features needed by that city's XGBoost model.
"""

import numpy as np
from config import TARGET


# ---------------------------------------------------------------------------
# Shared encoding helpers
# ---------------------------------------------------------------------------

def _season(month: int) -> int:
    """Encode month → season integer (0=Winter … 4=Post-Monsoon)."""
    return {12: 0, 1: 0, 2: 0,
             3: 1, 4: 1, 5: 1,
             6: 2, 7: 2,
             8: 3, 9: 3, 10: 3,
            11: 4}.get(month, 0)


def _aqi_cat_enc(aqi: float) -> int:
    """Encode AQI value → integer category (0–5)."""
    if aqi <= 50:  return 0
    if aqi <= 100: return 1
    if aqi <= 200: return 2
    if aqi <= 300: return 3
    if aqi <= 400: return 4
    return 5


# ---------------------------------------------------------------------------
# Per-city engineers
# ---------------------------------------------------------------------------

def engineer_xgb_anandvihar(df):
    fe = df.copy()
    fe["AQI_next"]     = fe[TARGET].shift(-1)
    fe["AQI_lag_72h"]  = fe[TARGET].shift(72)
    fe["AQI_lag_168h"] = fe[TARGET].shift(168)
    doy = fe.index.dayofyear
    fe["DayOfYear"] = doy
    fe["doy_sin"]   = np.sin(2 * np.pi * doy / 365)
    fe["doy_cos"]   = np.cos(2 * np.pi * doy / 365)
    return fe.dropna(subset=["AQI_next", "AQI_lag_72h", "AQI_lag_168h"])


def engineer_xgb_colaba(df):
    fe  = df.copy()
    idx = fe.index
    fe["AQI_next"] = fe[TARGET].shift(-1)
    for h in [1, 2, 3, 6, 12, 24, 48]:
        fe[f"lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3, 6, 24]:
        fe[f"roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    fe["hour"]        = idx.hour
    fe["day_of_week"] = idx.dayofweek
    fe["month"]       = idx.month
    fe["day_of_year"] = idx.dayofyear
    fe["hour_sin"]    = np.sin(2 * np.pi * idx.hour / 24)
    fe["hour_cos"]    = np.cos(2 * np.pi * idx.hour / 24)
    fe["month_sin"]   = np.sin(2 * np.pi * idx.month / 12)
    fe["month_cos"]   = np.cos(2 * np.pi * idx.month / 12)
    fe["dow_sin"]     = np.sin(2 * np.pi * idx.dayofweek / 7)
    fe["dow_cos"]     = np.cos(2 * np.pi * idx.dayofweek / 7)
    doy = idx.dayofyear
    fe["doy_sin"]     = np.sin(2 * np.pi * doy / 365)
    fe["doy_cos"]     = np.cos(2 * np.pi * doy / 365)
    fe["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["season_enc"]  = idx.month.map(_season)
    fe["aqi_cat_enc"] = fe[TARGET].map(_aqi_cat_enc)
    lag_cols = [f"lag_{h}h" for h in [1, 2, 3, 6, 12, 24, 48]]
    return fe.dropna(subset=["AQI_next"] + lag_cols)


def engineer_xgb_iitg(df):
    fe  = df.copy()
    idx = fe.index
    fe["AQI_next"] = fe[TARGET].shift(-1)
    if "AT" not in fe.columns:
        fe["AT"] = 0.0
    for h in [1, 2, 3, 6, 12, 24, 48]:
        fe[f"lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3, 6, 24]:
        fe[f"roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    fe["Hour"]       = idx.hour
    fe["DayOfWeek"]  = idx.dayofweek
    fe["Month"]      = idx.month
    fe["DayOfYear"]  = idx.dayofyear
    fe["Quarter"]    = idx.quarter
    fe["IsWeekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["Season_Enc"] = idx.month.map(_season)
    fe["Hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    fe["Hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
    fe["Month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    fe["Month_cos"]  = np.cos(2 * np.pi * idx.month / 12)
    fe["DoW_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    fe["DoW_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
    lag_cols = [f"lag_{h}h" for h in [1, 2, 3, 6, 12, 24, 48]]
    return fe.dropna(subset=["AQI_next"] + lag_cols)


def engineer_xgb_jayanagar(df):
    fe  = df.copy()
    idx = fe.index
    fe["AQI_next"] = fe[TARGET].shift(-1)
    for lag in [1, 2, 3, 4, 5, 6, 7, 9, 14, 21, 30]:
        fe[f"lag_{lag}"] = fe[TARGET].shift(lag)
    for w in [7, 14, 30]:
        fe[f"roll_mean_{w}"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}"]  = fe[TARGET].shift(1).rolling(w).std()
    for col in ["PM25", "PM10", "CO", "NOx", "NO2"]:
        fe[f"{col}_lag1"] = fe[col].shift(1) if col in fe.columns else 0.0
    fe["month"]       = idx.month
    fe["day_of_week"] = idx.dayofweek
    fe["day_of_year"] = idx.dayofyear
    fe["quarter"]     = idx.quarter
    fe["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["month_sin"]   = np.sin(2 * np.pi * idx.month / 12)
    fe["month_cos"]   = np.cos(2 * np.pi * idx.month / 12)
    fe["dow_sin"]     = np.sin(2 * np.pi * idx.dayofweek / 7)
    fe["dow_cos"]     = np.cos(2 * np.pi * idx.dayofweek / 7)
    doy = idx.dayofyear
    fe["doy_sin"]     = np.sin(2 * np.pi * doy / 365)
    fe["doy_cos"]     = np.cos(2 * np.pi * doy / 365)
    return fe.dropna(subset=["AQI_next", "lag_30", "roll_mean_30"])


def engineer_xgb_kodungaiyur(df):
    fe  = df.copy()
    idx = fe.index
    fe["AQI_next"] = fe[TARGET].shift(-1)
    for h in [1, 2, 3, 6, 12, 18, 24, 36, 48]:
        fe[f"lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3, 6, 24]:
        fe[f"roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    fe["hour"]        = idx.hour
    fe["day_of_week"] = idx.dayofweek
    fe["month"]       = idx.month
    fe["day_of_year"] = idx.dayofyear
    fe["quarter"]     = idx.quarter
    fe["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    fe["hour_sin"]    = np.sin(2 * np.pi * idx.hour / 24)
    fe["hour_cos"]    = np.cos(2 * np.pi * idx.hour / 24)
    fe["month_sin"]   = np.sin(2 * np.pi * idx.month / 12)
    fe["month_cos"]   = np.cos(2 * np.pi * idx.month / 12)
    fe["dow_sin"]     = np.sin(2 * np.pi * idx.dayofweek / 7)
    fe["dow_cos"]     = np.cos(2 * np.pi * idx.dayofweek / 7)
    doy = idx.dayofyear
    fe["doy_sin"]     = np.sin(2 * np.pi * doy / 365)
    fe["doy_cos"]     = np.cos(2 * np.pi * doy / 365)
    fe["season"]      = idx.month.map(_season)
    return fe.dropna(subset=["AQI_next", "lag_48h"])


def engineer_xgb_patia(df):
    fe  = df.copy()
    idx = fe.index
    fe["AQI_next"]         = fe[TARGET].shift(-1)
    fe["AQI_Category_Enc"] = fe[TARGET].map(_aqi_cat_enc)
    fe["Hour"]      = idx.hour
    fe["DayOfWeek"] = idx.dayofweek
    fe["Month"]     = idx.month
    fe["Year"]      = idx.year
    fe["DayOfYear"] = idx.dayofyear
    fe["Quarter"]   = idx.quarter
    fe["IsWeekend"] = (idx.dayofweek >= 5).astype(int)
    fe["Season_Enc"]= idx.month.map(_season)
    for h in [1, 2, 3, 6, 12, 24, 48]:
        fe[f"AQI_lag_{h}h"] = fe[TARGET].shift(h)
    for w in [3, 6, 24]:
        fe[f"AQI_roll_mean_{w}h"] = fe[TARGET].shift(1).rolling(w).mean()
        fe[f"AQI_roll_std_{w}h"]  = fe[TARGET].shift(1).rolling(w).std()
    fe["Hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    fe["Hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    fe["Month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    fe["Month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    fe["DOW_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    fe["DOW_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    return fe.dropna(subset=["AQI_next", "AQI_lag_48h"])


# ---------------------------------------------------------------------------
# Dispatch map  city_key → engineer function
# ---------------------------------------------------------------------------

XGB_ENGINEER = {
    "anandvihar"  : engineer_xgb_anandvihar,
    "colaba"      : engineer_xgb_colaba,
    "iitg"        : engineer_xgb_iitg,
    "jayanagar"   : engineer_xgb_jayanagar,
    "kodungaiyur" : engineer_xgb_kodungaiyur,
    "patia"       : engineer_xgb_patia,
}
