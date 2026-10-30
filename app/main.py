from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
LOG_PATH = ROOT / "monitoring" / "predictions.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Smart Customer Churn Prediction API",
    description="Predicts churn risk from customer service and billing attributes.",
    version="1.0.0",
)

MODEL = None
METADATA = {}


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)


@app.on_event("startup")
def load_model():
    global MODEL, METADATA
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@app.get("/")
def root():
    return {"message": "Smart Customer Churn Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL is not None,
        "model_version": METADATA.get("model_version"),
    }


@app.get("/metrics")
def metrics():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Training metrics not found. Run src/pipeline.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(customer: CustomerInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python src/pipeline.py first.")
    payload = customer.model_dump()
    frame = pd.DataFrame([payload])
    probability = float(MODEL.predict_proba(frame)[0, 1])
    predicted = int(probability >= 0.5)
    result = {
        "churn_prediction": "Yes" if predicted else "No",
        "churn_probability": round(probability, 4),
        "risk_level": "high" if probability >= 0.65 else ("medium" if probability >= 0.35 else "low"),
        "model_version": METADATA.get("model_version", "unknown"),
    }
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                *payload.keys(),
                "churn_probability",
                "churn_prediction",
                "risk_level",
                "model_version",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
            **result,
        })
    return result
