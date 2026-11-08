"""FastAPI service that scores customers for churn risk.

Run locally:   uvicorn app.main:app --reload
Docker:        docker build -t churn-api . && docker run -p 8000:8000 churn-api
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import CATEGORICAL, NUMERIC, add_features, clean

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"))
artifact = joblib.load(MODEL_PATH)
app = FastAPI(title="Churn Risk API", version="1.0.0")

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
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
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    high_risk: bool
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict", response_model=list[Prediction])
def predict(customers: list[Customer]):
    df = add_features(clean(pd.DataFrame([c.model_dump() for c in customers])))
    p = artifact["model"].predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]
    thr = float(artifact["threshold"])
    return [Prediction(churn_probability=round(float(x), 4), high_risk=bool(x >= thr), threshold=thr) for x in p]
