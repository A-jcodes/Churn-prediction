"""FastAPI serving layer for the churn model.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

The pydantic schema mirrors the raw dataset columns so callers send data
in the same shape the business systems store it; all preprocessing happens
inside the persisted sklearn pipeline.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    churn_predicted: bool
    risk_band: Literal["low", "medium", "high"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])
    proba = float(get_model().predict_proba(row)[0, 1])
    band = "high" if proba >= 0.6 else "medium" if proba >= 0.3 else "low"
    return Prediction(
        churn_probability=round(proba, 4),
        churn_predicted=proba >= 0.5,
        risk_band=band,
    )
