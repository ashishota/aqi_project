
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecasting System",
    page_icon="🌫️",
    layout="wide"
)

# ──────────────────────────────────────────────
# Model Registry
# Future models can be added here
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "Delhi": {
        "XGBoost": r"D:\AQI_Project\models\delhi_xgboost_aqi.pkl",
    },

    "Chennai": {
        "XGBoost": r"D:\AQI_Project\models\chennai_xgboost_aqi.pkl"
    },

    "Bangalore": {
        "XGBoost": r"D:\AQI_Project\models\bangalore_xgboost_aqi.pkl"
    },

    "Bhubaneswar": {
        "XGBoost": r"D:\AQI_Project\models\Bhubaneswar_xgboost_aqi.pkl"
    },

    "Guwahati": {
        "XGBoost": r"D:\AQI_Project\models\Guwahati_xgboost_aqi.pkl"
    },
    "Mumbai": {
        "XGBoost": r"D:\AQI_Project\models\mumbai_xgboost_aqi.pkl"
    }
}

# ──────────────────────────────────────────────
# EXACT FEATURES FROM NOTEBOOK
# ──────────────────────────────────────────────
BASE_FEATURES = [
    "PM25",
    "PM10",
    "NO2",
    "SO2",
    "NH3",
    "CO",
    "O3"
]

LAG_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30"
]

ROLLING_FEATURES = [
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_30"
]

CALENDAR_FEATURES = [
    "day_of_week",
    "month",
    "day_of_year",
    "quarter"
]

CYCLICAL_FEATURES = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos"
]

FEATURE_COLUMNS = (
    BASE_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
    + CALENDAR_FEATURES
    + CYCLICAL_FEATURES
)

# ──────────────────────────────────────────────
# AQI Category Function
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

# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):

    if not os.path.exists(model_path):
        return None

    return joblib.load(model_path)

# ──────────────────────────────────────────────
# Feature Engineering
# EXACTLY MATCHES NOTEBOOK
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# UI Header
# ──────────────────────────────────────────────
st.title("🌫️ AQI Forecasting System")

st.markdown(
    """
    Predict Air Quality Index using trained machine learning models.
    Supports future multi-model integration.
    """
)

st.divider()

# ──────────────────────────────────────────────
# City Selection
# ──────────────────────────────────────────────
city = st.selectbox(
    "📍 Select City",
    list(MODEL_REGISTRY.keys())
)

# Load XGBoost model for selected city
model_path = MODEL_REGISTRY[city]["XGBoost"]

model = load_model(model_path)

if model is None:

    st.warning(
        f"⚠️ Model file not found at: {model_path}"
    )

    st.stop()

st.success(f"✅ XGBoost model loaded for {city}")

# ──────────────────────────────────────────────
# Pollutant Inputs
# ──────────────────────────────────────────────
st.subheader("🧪 Pollutant Concentrations")

cols = st.columns(3)

inputs = {}

for idx, feature in enumerate(BASE_FEATURES):

    with cols[idx % 3]:

        inputs[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=10.0,
            step=0.1
        )

# Previous AQI input for lag features
previous_aqi = st.number_input(
    "Previous Day AQI",
    min_value=0.0,
    value=150.0,
    step=1.0
)

st.caption(
    "Used to generate lag and rolling statistical features."
)

st.divider()

# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────
if st.button(
    "🔍 Predict AQI",
    use_container_width=True,
    type="primary"
):

    try:

        feature_df = create_input_features(
            inputs,
            previous_aqi
        )

        prediction = model.predict(feature_df)[0]

        prediction = max(0.0, float(prediction))

        category, color = get_aqi_category(prediction)

        st.subheader("📊 Prediction Result")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                label="Predicted AQI",
                value=f"{prediction:.2f}"
            )

        with c2:
            st.markdown(
                f"""
                <div style='
                    padding:20px;
                    border-radius:10px;
                    background:{color};
                    color:white;
                    font-size:1.4rem;
                    font-weight:bold;
                    text-align:center;
                '>
                    {category}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Feature Preview
        with st.expander("View Model Input Features"):
            st.dataframe(feature_df)

        # AQI Scale Reference
        st.subheader("AQI Scale Reference")

        scale_df = pd.DataFrame({
            "AQI Range": [
                "0-50",
                "51-100",
                "101-200",
                "201-300",
                "301-400",
                "401+"
            ],
            "Category": [
                "Good",
                "Satisfactory",
                "Moderate",
                "Poor",
                "Very Poor",
                "Severe"
            ]
        })

        st.dataframe(
            scale_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")
