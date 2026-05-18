"""
loader.py
=========
Startup loader: reads CSVs, fits XGBoost test sets, and loads LSTM/GRU models
for every city defined in CITY_CONFIG.  Call load_all_cities() once at startup;
it returns the populated CITY_DATA dict that routes read from.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from config import (
    CITY_CONFIG, DATA_DIR, XGB_MODELS_DIR, XGB_FEATS_DIR,
    LSTM_BASE_DIR, LSTM_CITY_FOLDER, TARGET, LOOK_BACK_DEFAULT, TEST_SPLIT,
)
from features.xgb_engineer  import XGB_ENGINEER
from features.lstm_engineer  import LSTM_FEATURES, LSTM_LOOK_BACK, build_lstm_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_entry(cfg: dict) -> dict:
    """Return a zeroed-out entry dict for one city."""
    return {
        "label"          : cfg["label"],
        "station"        : cfg["station"],
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
        # Static metrics (hardcoded from notebook runs)
        "arima_metrics"  : cfg["arima_metrics"],
        "sarimax_metrics": cfg["sarimax_metrics"],
        "date_range"     : None,
    }


def _load_xgboost(city_key: str, cfg: dict, df_raw: pd.DataFrame, entry: dict) -> None:
    """Mutate *entry* in-place with XGBoost model and test-set predictions."""
    xgb_path  = os.path.join(XGB_MODELS_DIR, cfg["xgb_model"])
    feat_path = os.path.join(XGB_FEATS_DIR,  cfg["xgb_feats"])

    if not (os.path.exists(xgb_path) and os.path.exists(feat_path)):
        print(f"  ⚠️  XGBoost model/features not found — skipping ({xgb_path})")
        return

    try:
        xgb_model     = joblib.load(xgb_path)
        xgb_feat_cols = json.load(open(feat_path))
        df_eng        = XGB_ENGINEER[city_key](df_raw)

        n          = len(df_eng)
        df_test    = df_eng.iloc[int(n * TEST_SPLIT):]
        available  = [c for c in xgb_feat_cols if c in df_test.columns]
        missing    = [c for c in xgb_feat_cols if c not in df_test.columns]
        if missing:
            print(f"  ⚠️  XGB missing cols (skipped): {missing}")

        X_xgb_test = df_test[available]
        y_xgb_test = df_test["AQI_next"].values
        y_pred_xgb = xgb_model.predict(X_xgb_test)

        entry.update({
            "xgb_model"      : xgb_model,
            "xgb_feat_cols"  : available,
            "X_xgb_test"     : X_xgb_test,
            "y_xgb_test"     : y_xgb_test,
            "xgb_timestamps" : df_test.index,
            "y_pred_xgb"     : y_pred_xgb,
            "y_true_xgb"     : y_xgb_test,
        })
        print(f"  ✅ XGBoost loaded  test={len(X_xgb_test):,} samples")
    except Exception as exc:
        print(f"  ❌ XGBoost failed: {exc}")


def _load_lstm_gru(city_key: str, cfg: dict, df_raw: pd.DataFrame, entry: dict) -> None:
    """Mutate *entry* in-place with LSTM + GRU models and test-set predictions."""
    city_folder = LSTM_CITY_FOLDER.get(city_key)
    saved_dir   = os.path.join(LSTM_BASE_DIR, city_folder, "saved_models")

    paths = {
        "lstm" : os.path.join(saved_dir, cfg["lstm_model"])  if cfg.get("lstm_model")  else None,
        "gru"  : os.path.join(saved_dir, cfg["gru_model"])   if cfg.get("gru_model")   else None,
        "fscl" : os.path.join(saved_dir, cfg["feat_scaler"]) if cfg.get("feat_scaler") else None,
        "tscl" : os.path.join(saved_dir, cfg["tgt_scaler"])  if cfg.get("tgt_scaler")  else None,
    }

    if not all(p and os.path.exists(p) for p in paths.values()):
        print("  ⏳ LSTM/GRU model files not present yet — will activate on next restart")
        return

    try:
        from tensorflow.keras.models import load_model  # lazy import to avoid TF overhead

        lstm_model    = load_model(paths["lstm"])
        gru_model     = load_model(paths["gru"])
        feat_scaler   = joblib.load(paths["fscl"])
        target_scaler = joblib.load(paths["tscl"])

        lstm_feats = LSTM_FEATURES.get(city_key, [])
        look_back  = LSTM_LOOK_BACK.get(city_key, LOOK_BACK_DEFAULT)

        if not lstm_feats:
            raise ValueError(f"LSTM_FEATURES not defined for '{city_key}'")

        fe = build_lstm_features(df_raw, city_key)

        # Determine which features are available
        if hasattr(feat_scaler, "feature_names_in_"):
            expected        = list(feat_scaler.feature_names_in_)
            missing_feats   = [f for f in expected if f not in fe.columns]
            available_feats = [f for f in expected if f in fe.columns]
        else:
            available_feats = [f for f in lstm_feats if f in fe.columns]
            missing_feats   = [f for f in lstm_feats if f not in fe.columns]
        if missing_feats:
            print(f"  ⚠️  LSTM missing cols: {missing_feats}")

        data            = fe[available_feats + [TARGET]].copy()
        scaled_features = feat_scaler.transform(data[available_feats])
        scaled_target   = target_scaler.transform(data[[TARGET]])
        scaled_data     = np.hstack([scaled_features, scaled_target])

        # Build sequences
        def _create_sequences(d, lb):
            X, y = [], []
            for i in range(lb, len(d)):
                X.append(d[i - lb:i, :-1])
                y.append(d[i, -1])
            return np.array(X), np.array(y)

        X_all, y_all = _create_sequences(scaled_data, look_back)
        n            = len(X_all)
        val_end      = int(n * TEST_SPLIT)
        X_test       = X_all[val_end:]
        y_test       = y_all[val_end:]
        test_ts      = data.index[look_back:][val_end:]

        y_true_lstm = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
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
    except Exception as exc:
        print(f"  ❌ LSTM/GRU failed: {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_all_cities() -> dict:
    """
    Load all cities defined in CITY_CONFIG.
    Returns the populated CITY_DATA dict.
    """
    city_data = {}

    print("=" * 60)
    print("  AQI Multi-City Forecasting System — Loading...")
    print("=" * 60)

    for city_key, cfg in CITY_CONFIG.items():
        print(f"\n[{cfg['label']}, {cfg['station']}]")

        csv_path = os.path.join(DATA_DIR, cfg["csv"])
        if not os.path.exists(csv_path):
            print(f"  ⚠️  CSV not found: {csv_path} — skipping city")
            continue

        df_raw = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        df_raw = df_raw.set_index("Timestamp").sort_index()

        entry = _empty_entry(cfg)
        entry["date_range"] = f"{df_raw.index.min().date()} → {df_raw.index.max().date()}"
        entry["df_raw"]     = df_raw
        print(f"  ✅ CSV loaded  {df_raw.shape}  {entry['date_range']}")

        _load_xgboost(city_key, cfg, df_raw, entry)
        _load_lstm_gru(city_key, cfg, df_raw, entry)

        city_data[city_key] = entry

    print(f"\n{'=' * 60}")
    print(f"  ✅ Ready. Cities loaded: {list(city_data.keys())}")
    print(f"{'=' * 60}\n")

    return city_data
