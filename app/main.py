from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

LOG_PATH = ROOT / "monitoring" / "predictions.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
REFERENCE_PATH = ROOT / "monitoring" / "reference_data.csv"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Smart Customer Churn Prediction API",
    description=(
        "Customer churn prediction API with "
        "real-time model monitoring."
    ),
    version="2.0.0",
)

MODEL = None
METADATA = {}


# ============================================================
# CUSTOMER INPUT VALIDATION
# ============================================================

class CustomerInput(BaseModel):

    gender: Literal["Female", "Male"]

    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=100)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]

    OnlineSecurity: Literal[
        "Yes", "No", "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes", "No", "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes", "No", "No internet service"
    ]

    TechSupport: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingTV: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes", "No", "No internet service"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0, le=1000)

    TotalCharges: float = Field(ge=0, le=100000)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@app.on_event("startup")
def load_model():

    global MODEL, METADATA

    try:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        MODEL = joblib.load(MODEL_PATH)

        if METADATA_PATH.exists():

            METADATA = json.loads(
                METADATA_PATH.read_text(
                    encoding="utf-8"
                )
            )

        print("Model loaded successfully.")

    except Exception as error:

        MODEL = None

        print(f"Model loading failed: {error}")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Smart Customer Churn Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "monitoring": "/monitoring/summary"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if MODEL is not None
            else "model_not_loaded"
        ),
        "model_loaded": MODEL is not None,
        "model_version": METADATA.get(
            "model_version",
            "unknown"
        )
    }


# ============================================================
# TRAINING METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    if not METRICS_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail="Training metrics not found."
        )

    return json.loads(
        METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerInput):

    if MODEL is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    try:

        payload = customer.model_dump()

        frame = pd.DataFrame([payload])

        probability = float(
            MODEL.predict_proba(frame)[0, 1]
        )

        predicted = int(probability >= 0.5)

        risk_level = (
            "high"
            if probability >= 0.65
            else (
                "medium"
                if probability >= 0.35
                else "low"
            )
        )

        result = {

            "churn_prediction": (
                "Yes" if predicted else "No"
            ),

            "churn_probability": round(
                probability, 4
            ),

            "risk_level": risk_level,

            "model_version": METADATA.get(
                "model_version",
                "unknown"
            )
        }

        # ----------------------------------------
        # SAVE PREDICTION LOG
        # ----------------------------------------

        fieldnames = [
            "timestamp_utc",
            *payload.keys(),
            "churn_probability",
            "churn_prediction",
            "risk_level",
            "model_version"
        ]

        file_exists = LOG_PATH.exists()

        with LOG_PATH.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow({

                "timestamp_utc": (
                    datetime.now(timezone.utc).isoformat()
                ),

                **payload,
                **result
            })

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# MONITORING SUMMARY ENDPOINT
# ============================================================

@app.get("/monitoring/summary")
def monitoring_summary():

    # ----------------------------------------
    # LOAD TRAINING METRICS
    # ----------------------------------------

    metrics_data = {}

    if METRICS_PATH.exists():

        try:

            metrics_data = json.loads(
                METRICS_PATH.read_text(
                    encoding="utf-8"
                )
            )

        except (OSError, json.JSONDecodeError):

            metrics_data = {}

    # ----------------------------------------
    # CHECK PREDICTION LOGS
    # ----------------------------------------

    if not LOG_PATH.exists():

        return {

            "total_requests": 0,
            "churn_yes": 0,
            "churn_no": 0,
            "average_probability": 0,
            "predictions": [],
            "drift": [],
            "metrics": metrics_data
        }

    try:

        logs = pd.read_csv(LOG_PATH)

    except (OSError, pd.errors.ParserError):

        raise HTTPException(
            status_code=500,
            detail="Unable to read prediction logs."
        )

    if logs.empty:

        return {

            "total_requests": 0,
            "churn_yes": 0,
            "churn_no": 0,
            "average_probability": 0,
            "predictions": [],
            "drift": [],
            "metrics": metrics_data
        }

    required_columns = [
        "timestamp_utc",
        "churn_probability",
        "churn_prediction",
        "risk_level"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in logs.columns
    ]

    if missing_columns:

        raise HTTPException(
            status_code=500,
            detail=f"Missing log columns: {missing_columns}"
        )

    logs["timestamp_utc"] = pd.to_datetime(
        logs["timestamp_utc"],
        errors="coerce",
        utc=True
    )

    logs["churn_probability"] = pd.to_numeric(
        logs["churn_probability"],
        errors="coerce"
    )

    logs = logs.dropna(
        subset=[
            "timestamp_utc",
            "churn_probability"
        ]
    )

    if logs.empty:

        return {

            "total_requests": 0,
            "churn_yes": 0,
            "churn_no": 0,
            "average_probability": 0,
            "predictions": [],
            "drift": [],
            "metrics": metrics_data
        }

    # ----------------------------------------
    # PREDICTION DISTRIBUTION
    # ----------------------------------------

    churn_yes = int(
        (logs["churn_prediction"] == "Yes").sum()
    )

    churn_no = int(
        (logs["churn_prediction"] == "No").sum()
    )

    average_probability = float(
        logs["churn_probability"].mean()
    )

    # ----------------------------------------
    # RECENT PREDICTIONS
    # ----------------------------------------

    recent_logs = logs.tail(200)

    predictions = []

    for _, row in recent_logs.iterrows():

        predictions.append({

            "timestamp": row[
                "timestamp_utc"
            ].isoformat(),

            "probability": float(
                row["churn_probability"]
            ),

            "prediction": str(
                row["churn_prediction"]
            ),

            "risk": str(
                row["risk_level"]
            )
        })

    # ----------------------------------------
    # FEATURE DRIFT DETECTION
    # ----------------------------------------

    drift_results = []

    if REFERENCE_PATH.exists():

        try:

            reference = pd.read_csv(
                REFERENCE_PATH
            )

            numeric_features = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            for column in numeric_features:

                if (
                    column not in reference.columns
                    or column not in logs.columns
                ):
                    continue

                baseline = pd.to_numeric(
                    reference[column],
                    errors="coerce"
                ).dropna()

                current = pd.to_numeric(
                    logs[column],
                    errors="coerce"
                ).dropna()

                if (
                    len(baseline) >= 5
                    and len(current) >= 5
                ):

                    test_result = ks_2samp(
                        baseline,
                        current
                    )

                    p_value = float(
                        test_result.pvalue
                    )

                    drift_results.append({

                        "feature": column,

                        "baseline_n": len(
                            baseline
                        ),

                        "request_n": len(
                            current
                        ),

                        "ks_statistic": round(
                            float(
                                test_result.statistic
                            ),
                            4
                        ),

                        "p_value": round(
                            p_value,
                            4
                        ),

                        "signal": (
                            "Potential drift"
                            if p_value < 0.05
                            else "No clear signal"
                        )
                    })

        except Exception as error:

            print(
                f"Drift calculation failed: {error}"
            )

    # ----------------------------------------
    # FINAL MONITORING RESPONSE
    # ----------------------------------------

    return {

        "total_requests": len(logs),

        "churn_yes": churn_yes,

        "churn_no": churn_no,

        "average_probability": round(
            average_probability,
            4
        ),

        "predictions": predictions,

        "drift": drift_results,

        "metrics": metrics_data
    }
