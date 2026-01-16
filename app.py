import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import joblib
import os

# =================================================
# FIXED LSTM
# =================================================
class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
DATA_PATH = "Smart Home Energy Consumption Optimization (1).csv"
MODEL_DIR = "models"

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
WINDOW = 6  # same window as training

def get_trained_devices():
    devices = []
    for file in os.listdir(MODEL_DIR):
        if file.endswith("_lstm_fixed.h5"):
            devices.append(file.replace("_lstm_fixed.h5", ""))
    return devices

def load_device_model(device_id):
    model_path = os.path.join(MODEL_DIR, f"{device_id}_lstm_fixed.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{device_id}_scaler.pkl")

    model = load_model(model_path, compile=False, custom_objects={"FixedLSTM": FixedLSTM})
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_last_n_hours(device_id, n=WINDOW):
    """Get last n data points for device."""
    if device_id == "washer9":
        values = df[df["device_id"] == device_id]["power_watt"].values
    else:
        values = df[df["device_type"] == device_id]["power_watt"].values
    if len(values) < n:
        return None
    return values[-n:]

def predict_future(device_id, user_input, hours):
    model, scaler = load_device_model(device_id)
    history_values = get_last_n_hours(device_id)
    if history_values is None:
        return None

    history = list(history_values[1:]) + [user_input]

    predictions = []

    for _ in range(hours):
        seq = np.array(history[-WINDOW:])
        seq_scaled = scaler.transform(seq.reshape(-1, 1))
        seq_scaled = seq_scaled.reshape(1, WINDOW, 1)

        pred_scaled = model.predict(seq_scaled, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]

        predictions.append(pred)
        history.append(pred)

    return np.mean(predictions)

# -------------------------------------------------
# PAGE TITLE
# -------------------------------------------------
st.markdown("<h1 style='text-align:center;'>SMART ENERGY CONSUMPTION FORECASTING</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>LSTM Based Device-Level Prediction</h4>", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation", ["Dashboard", "Forecast"])

# =================================================
# DASHBOARD
# =================================================
if menu == "Dashboard":
    st.subheader("📊 Energy Consumption Dashboard")
    hourly = df.groupby(df["timestamp"].dt.hour)["power_watt"].sum()
    daily = df.groupby(df["timestamp"].dt.date)["power_watt"].sum()
    weekly = df.groupby(df["timestamp"].dt.isocalendar().week)["power_watt"].sum()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar(hourly.index, hourly.values)
        ax.set_title("Hourly Energy Consumption")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (Watts)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(daily.values)
        ax.set_title("Daily Energy Consumption")
        ax.set_ylabel("Power (Watts)")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(weekly.values)
    ax.set_title("Weekly Energy Consumption")
    ax.set_xlabel("Week")
    ax.set_ylabel("Power (Watts)")
    st.pyplot(fig)

# =================================================
# FORECAST
# =================================================
if menu == "Forecast":
    st.subheader("🔮 Energy Consumption Forecast")
    trained_devices = get_trained_devices()

    if not trained_devices:
        st.error("❌ No devices have trained models yet. Please run training first.")
    else:
        device = st.selectbox("Select Device", trained_devices)
        current_energy = st.number_input("Enter Current Energy Consumption (Watts)", min_value=0.0, step=1.0)
        horizon = st.selectbox("Prediction Horizon", ["Next 1 Hour", "Next 6 Hours", "Next 24 Hours", "Next 48 Hours"])
        hours = {"Next 1 Hour":1, "Next 6 Hours":6, "Next 24 Hours":24, "Next 48 Hours":48}[horizon]

        if st.button("🔍 Predict"):
            prediction = predict_future(device, current_energy, hours)
            
            if prediction is None:
                st.warning(f"Not enough historical data for {device} to make predictions.")
            else:
                st.success(f"⚡ Predicted Energy Consumption: **{prediction:.2f} Watts**")

                # Device-specific thresholds
                thresholds = {
                    "washer9": {"moderate": 1000, "high": 2000},
                    "air_conditioner": {"moderate": 200, "high": 500},
                    "light": {"moderate": 50, "high": 100},
                    "tv": {"moderate": 100, "high": 300},
                    "fridge": {"moderate": 150, "high": 400}
                }

                device_thresh = thresholds.get(device, {"moderate": 1000, "high": 2000})

                if prediction > device_thresh["high"]:
                    st.warning("⚠️ High usage detected. Consider reducing usage.")
                elif prediction > device_thresh["moderate"]:
                    st.info("ℹ️ Moderate usage. Try optimizing power usage.")
                else:
                    st.success("✅ Energy usage is optimal.")
