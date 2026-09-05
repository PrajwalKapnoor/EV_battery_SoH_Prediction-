"""
Stage 3 — FastAPI backend
Run with:  uvicorn main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from model_service import predict_soh


# ------------------------------------------------------------------
# Pydantic schema: defines exactly what /predict expects as input.
# FastAPI uses this to auto-validate incoming JSON AND to generate
# the interactive /docs page — you get both for free from one class.
# ------------------------------------------------------------------
class SoHRequest(BaseModel):
    cycle: int = Field(..., description="Cycle number", ge=0,le=169,examples=[30])
    voltage: float = Field(..., description="Voltage reading",ge=3.46,le=3.56,examples=[3.50])
    temperature: float = Field(..., description="Temperature in Celsius",ge=31.27,le=34.24,examples=[31.50])


# Optional but nice: describes what /predict returns. Not required —
# FastAPI can infer JSON from a plain dict — but this makes the
# response shape show up correctly in the /docs page too.
class SoHResponse(BaseModel):
    predicted_soh: float


# ------------------------------------------------------------------
# The FastAPI app itself. This is the object uvicorn runs.
# ------------------------------------------------------------------
app = FastAPI(title="EV Battery SoH Prediction API")


# Optional but genuinely useful: a one-line endpoint to confirm the
# server is alive, separate from the real prediction logic.
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=SoHResponse)
def predict(request: SoHRequest):
    """
    request has ALREADY been validated by Pydantic by the time this
    function runs — no manual checking needed here.
    """
    result = predict_soh(
        cycle=request.cycle,
        voltage=request.voltage,
        temperature=request.temperature,
    )
    return SoHResponse(predicted_soh=result)
