import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# -------------------------------------------------
# PATHS
# -------------------------------------------------
DATA_PATH = "Smart Home Energy Consumption Optimization (1).csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# -------------------------------------------------
# DEVICES TO TRAIN
# -------------------------------------------------
DEVICES = ["washer9", "air_conditioner", "light", "tv", "fridge"]

# -------------------------------------------------
# SEQUENCE CREATION
# -------------------------------------------------
WINDOW = 6  # smaller window allows prediction with fewer data points

def create_sequences(data, window=WINDOW):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# -------------------------------------------------
# TRAIN FUNCTION
# -------------------------------------------------
def train_device(device):
    print(f"\nTraining model for: {device}")

    # washer9 is device_id, others are device_type
    if device == "washer9":
        values = df[df["device_id"] == device]["power_watt"].values
    else:
        values = df[df["device_type"] == device]["power_watt"].values

    if len(values) < WINDOW + 1:
        print(f"❌ Not enough data for {device}, skipping")
        return

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    X, y = create_sequences(values_scaled, window=WINDOW)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        Input(shape=(WINDOW, 1)),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    model.save(f"{MODEL_DIR}/{device}_lstm_fixed.h5")
    joblib.dump(scaler, f"{MODEL_DIR}/{device}_scaler.pkl")

    print(f"✅ Model trained & saved for {device}")

# -------------------------------------------------
# TRAIN ALL DEVICES
# -------------------------------------------------
for device in DEVICES:
    train_device(device)
