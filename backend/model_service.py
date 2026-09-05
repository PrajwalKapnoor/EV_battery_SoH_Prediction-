"""
Stage 2 — Model service (no FastAPI here yet)

Purpose: load the trained model + scaler ONCE, and expose a single
predict() function. This file gets imported by FastAPI in Stage 3 —
today we just prove it works standalone.
"""

from pathlib import Path          # used instead of `os`
import numpy as np
import joblib
from tensorflow import keras

# Paths to the two artifacts you already trained
MODEL_PATH = Path(__file__).parent.parent / "model" / "soh_ann_model.keras"
SCALER_PATH = Path(__file__).parent.parent / "model" / "soh_feature_scaler.pkl"

# Loaded once, at import time — NOT inside predict(). Reloading a Keras
# model on every request would be slow and pointless.
_model = keras.models.load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)


def predict_soh(cycle: int, voltage: float, temperature: float) -> float:
    """
    Takes the three raw input values, scales them the same way the
    training data was scaled, runs the model, and returns a plain
    Python float (not a numpy type — matters later for JSON output).
    """
    raw_input = np.array([[cycle, voltage, temperature]])
    scaled_input = _scaler.transform(raw_input)

    prediction = _model.predict(scaled_input, verbose=0)

    return float(prediction[0][0])


# Quick manual sanity check — run this file directly to confirm loading works
if __name__ == "__main__":
    result = predict_soh(cycle=50, voltage=3.52, temperature=32.63)
    print(f"Predicted SoH: {result:.2f}%")
