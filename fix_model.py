import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import os

# ---- Custom LSTM without time_major ----
class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # remove unsupported argument
        super().__init__(*args, **kwargs)

# ---- Paths ----
OLD_MODEL_PATH = "models/washer9_lstm.h5"
NEW_MODEL_PATH = "models/washer9_lstm_fixed.h5"

# ---- Load old model safely ----
model = load_model(
    OLD_MODEL_PATH,
    compile=False,
    custom_objects={"LSTM": FixedLSTM}
)

# ---- Save fixed model ----
model.save(NEW_MODEL_PATH)

print("✅ Model fixed and saved as:", NEW_MODEL_PATH)
