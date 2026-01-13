
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import joblib

app = Flask(__name__)

# -------------------------------
# LOAD MODEL WEIGHTS & SCALER
# -------------------------------

# Rebuild LSTM architecture exactly as in the trained model
lstm_model = Sequential([
    Input(shape=(24, 1)),          # input layer
    LSTM(32, return_sequences=True),  # first LSTM layer
    LSTM(16),                        # second LSTM layer
    Dense(1)                         # output layer
])

# Load trained weights
lstm_model.load_weights("tuned_lstm_energy_model_weights.h5")

# Load scaler (saved using joblib)
y_scaler = joblib.load("energy_scaler.pkl")

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.route("/")
def home():
    return "Energy Prediction API is running"

# -------------------------------
# PREDICTION ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting 24 normalized values
    last_24_values = data["last_24_values"]

    # Convert to model input shape
    seq = np.array(last_24_values).reshape(1, 24, 1)

    # Predict (normalized)
    pred_norm = lstm_model.predict(seq)

    # Convert back to real energy values
    pred_real = y_scaler.inverse_transform(pred_norm)

    return jsonify({
        "predicted_energy_watts": float(pred_real[0][0])
    })

# -------------------------------
# RUN FLASK APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

