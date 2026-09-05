"""
Stage 5 — Streamlit frontend

Collects input, calls the FastAPI backend, displays the result.
Run with:  streamlit run app.py
(Requires the FastAPI server from Stage 3 to already be running.)
"""

import streamlit as st
import requests
import os

# Reads from an environment variable if set (e.g. on Streamlit Cloud),
# otherwise falls back to localhost for local development.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_URL = f"{BACKEND_URL}/predict"

st.title("EV Battery SoH Predictor")
st.write("Enter cycle telemetry to estimate battery State-of-Health.")

# --- Input widgets ---------------------------------------------------
cycle = st.number_input("Cycle number", min_value=0, value=50, step=1)
voltage = st.number_input("Voltage (V)", min_value=3.46, max_value=3.56, value=3.50, format="%.2f")
temperature = st.number_input("Temperature (°C)", value=32.0, format="%.2f")

# --- Predict button ---------------------------------------------------
if st.button("Predict SoH"):
    payload = {
        "cycle": int(cycle),
        "voltage": float(voltage),
        "temperature": float(temperature),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()  # raises an error if status isn't 200
        result = response.json()

        st.metric("Predicted SoH", f"{result['predicted_soh']:.2f}%")

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Is FastAPI running on localhost:8000?")
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend returned an error: {e}")