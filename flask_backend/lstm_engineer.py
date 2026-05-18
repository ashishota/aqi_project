"""
features/lstm_engineer.py
=========================
LSTM / GRU feature configuration and the shared feature-engineering function.
"""

import numpy as np
from config import TARGET, LOOK_BACK_DEFAULT


# ---------------------------------------------------------------------------
# Per-city feature lists
# ---------------------------------------------------------------------------

LSTM_FEATURES = {
    "anandvihar": [
        'PM25', 'PM10', 'NO2', 'CO', 'AQI_lag1', 'AQI_lag2',
        'AQI_roll3_mean', 'AQI_roll6_mean', 'O3', 'SO2', 'NH3',
        'NOx_total', 'hour_sin', 'hour_cos', 'AQI_roll12_mean',
        'PM25_lag1', 'RH', 'NO', 'WS', 'AT', 'SR', 'Toluene_Benzene_ratio',
    ],
    "iitg": [
        'PM25', 'PM10', 'CO', 'NO2', 'AQI_lag1', 'AQI_lag2',
        'AQI_roll3_mean', 'AQI_roll6_mean', 'O3', 'SO2', 'NH3',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        'RH', 'WS', 'SR', 'BP', 'WD_sin', 'WD_cos',
        'AQI_roll12_mean', 'AQI_roll24_mean', 'AQI_lag24',
        'PM25_lag1', 'RH_PM25_interact', 'NO', 'NOx', 'VOC_total',
        'RF_flag', 'RF_lag1',
    ],
    "patia": [
        'PM25', 'PM10', 'Benzene', 'Eth-Benzene', 'MP-Xylene',
        'lag_1', 'lag_6', 'lag_24',
        'roll_mean_6', 'roll_mean_24', 'ema_24',
        'NO2', 'NOx', 'SO2', 'O3', 'NH3', 'CO',
        'AT', 'RH', 'WS', 'RF', 'SR',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'ema_6', 'roll_std_24', 'diff_1',
        'AT_RH_stability', 'BTEX_total',
    ],
    "kodungaiyur": [
        'PM25', 'PM10',
        'lag_1', 'lag_6', 'lag_24',
        'roll_mean_6', 'roll_mean_24', 'ema_24',
        'SO2', 'NO2', 'NH3', 'CO', 'O3',
        'RH', 'WS', 'RF', 'SR',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'ema_6', 'roll_std_24', 'diff_1',
    ],
    "colaba": [
        'PM25', 'PM10', 'NO2', 'CO', 'AQI_lag1', 'AQI_lag2',
        'AQI_roll3_mean', 'AQI_roll6_mean', 'O3', 'SO2', 'NH3',
        'NOx_total', 'hour_sin', 'hour_cos', 'AQI_roll12_mean',
        'PM25_lag1', 'RH', 'NO', 'WS', 'AT', 'SR', 'EthBenzene_MPXylene_ratio',
    ],
    "jayanagar": [
        'PM25', 'PM10', 'NO2', 'CO', 'AQI_lag1', 'AQI_lag2', 'AQI_lag3',
        'AQI_roll3_mean', 'AQI_roll6_mean', 'O3', 'SO2', 'NH3',
        'NOx_total', 'RH', 'WD_sin', 'WD_cos', 'BP',
        'month_sin', 'month_cos', 'AQI_roll12_mean', 'AQI_roll24_mean',
        'PM25_lag1', 'NO', 'AT', 'hour_sin', 'hour_cos',
    ],
}

# Per-city look-back window (hours)
LSTM_LOOK_BACK = {
    "anandvihar" : 48,
    "iitg"       : 48,
    "patia"      : 48,
    "kodungaiyur": 48,
    "colaba"     : 48,
    "jayanagar"  : 48,
}


# ---------------------------------------------------------------------------
# Shared LSTM feature builder
# ---------------------------------------------------------------------------

def build_lstm_features(df_raw, city_key: str):
    """
    Build the full feature matrix for LSTM / GRU inference.
    Returns a DataFrame (NaN rows already dropped).
    """
    fe = df_raw.copy()

    # -- Lag features ---------------------------------------------------------
    fe["AQI_lag1"]  = fe[TARGET].shift(1)
    fe["AQI_lag2"]  = fe[TARGET].shift(2)
    fe["AQI_lag3"]  = fe[TARGET].shift(3)
    fe["AQI_lag24"] = fe[TARGET].shift(24)
    fe["PM25_lag1"] = fe["PM25"].shift(1) if "PM25" in fe.columns else 0.0

    # -- Rolling statistics ---------------------------------------------------
    for w in [3, 6, 12, 24]:
        fe[f"AQI_roll{w}_mean"] = fe[TARGET].shift(1).rolling(w).mean()

    # -- Cyclical time encodings ----------------------------------------------
    fe["hour_sin"]  = np.sin(2 * np.pi * fe.index.hour / 24)
    fe["hour_cos"]  = np.cos(2 * np.pi * fe.index.hour / 24)
    fe["month_sin"] = np.sin(2 * np.pi * fe.index.month / 12)
    fe["month_cos"] = np.cos(2 * np.pi * fe.index.month / 12)
    fe["dow_sin"]   = np.sin(2 * np.pi * fe.index.dayofweek / 7)
    fe["dow_cos"]   = np.cos(2 * np.pi * fe.index.dayofweek / 7)
    fe["is_weekend"] = (fe.index.dayofweek >= 5).astype(int)

    # -- Wind direction -------------------------------------------------------
    if "WD" in fe.columns:
        fe["WD_sin"] = np.sin(2 * np.pi * fe["WD"] / 360)
        fe["WD_cos"] = np.cos(2 * np.pi * fe["WD"] / 360)
    else:
        fe["WD_sin"] = 0.0
        fe["WD_cos"] = 0.0

    # -- Interaction / derived pollutant features -----------------------------
    fe["NOx_total"] = (
        fe["NO"] + fe["NO2"] if ("NO" in fe.columns and "NO2" in fe.columns) else 0.0
    )
    fe["Toluene_Benzene_ratio"] = (
        fe["Toluene"] / (fe["Benzene"] + 1e-6)
        if ("Toluene" in fe.columns and "Benzene" in fe.columns) else 0.0
    )

    # IITG-specific
    fe["RH_PM25_interact"] = (
        fe["RH"] * fe["PM25"] if ("RH" in fe.columns and "PM25" in fe.columns) else 0.0
    )
    voc_cols = [c for c in ["Toluene", "Benzene", "Xylene", "Ethylbenzene"] if c in fe.columns]
    fe["VOC_total"] = fe[voc_cols].sum(axis=1) if voc_cols else 0.0
    if "RF" in fe.columns:
        fe["RF_flag"] = (fe["RF"] > 0).astype(float)
        fe["RF_lag1"] = fe["RF_flag"].shift(1)
    else:
        fe["RF_flag"] = 0.0
        fe["RF_lag1"] = 0.0
    if "NOx" not in fe.columns:
        fe["NOx"] = fe.get("NOx_total", 0.0)
    if "BP" not in fe.columns:
        fe["BP"] = 0.0

    # Patia / Kodungaiyur-specific
    fe["lag_1"]  = fe[TARGET].shift(1)
    fe["lag_6"]  = fe[TARGET].shift(6)
    fe["lag_24"] = fe[TARGET].shift(24)
    fe["roll_mean_6"]  = fe[TARGET].shift(1).rolling(6).mean()
    fe["roll_mean_24"] = fe[TARGET].shift(1).rolling(24).mean()
    fe["roll_std_24"]  = fe[TARGET].shift(1).rolling(24).std()
    fe["ema_6"]  = fe[TARGET].shift(1).ewm(span=6,  adjust=False).mean()
    fe["ema_24"] = fe[TARGET].shift(1).ewm(span=24, adjust=False).mean()
    fe["diff_1"] = fe[TARGET].diff(1)
    fe["AT_RH_stability"] = (
        fe["AT"] / (fe["RH"] + 1) if ("AT" in fe.columns and "RH" in fe.columns) else 0.0
    )
    btex_cols = [c for c in ["Benzene", "Toluene", "Eth-Benzene", "MP-Xylene"] if c in fe.columns]
    fe["BTEX_total"] = fe[btex_cols].sum(axis=1) if btex_cols else 0.0

    # Colaba-specific
    fe["EthBenzene_MPXylene_ratio"] = (
        fe["Eth-Benzene"] / (fe["MP-Xylene"] + 1e-6)
        if ("Eth-Benzene" in fe.columns and "MP-Xylene" in fe.columns) else 0.0
    )

    fe.dropna(inplace=True)
    return fe
