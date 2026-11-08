"""Data cleaning and feature engineering for the Telco churn dataset (pandas path).

The same transformations are implemented in PySpark in ``etl_spark.py``; ``tests/test_features.py``
checks that both paths produce identical feature tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]
NUMERIC = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "num_services",
    "avg_monthly_spend", "charge_ratio", "is_month_to_month", "auto_pay", "has_fiber",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, impute the 11 blank TotalCharges (new customers, tenure 0) and encode the label."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan))
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    df["charge_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, np.nan)
    df["charge_ratio"] = df["charge_ratio"].fillna(1.0)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                 labels=["0-6", "7-12", "13-24", "25-48", "49-72"]).astype(str)
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(clean(df))
