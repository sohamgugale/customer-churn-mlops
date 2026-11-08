from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "churn_model.joblib").exists(), reason="train the model first")

CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_high_risk_profile(client):
    loyal = CUSTOMER | {"tenure": 70, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
                        "OnlineSecurity": "Yes", "TechSupport": "Yes", "TotalCharges": 6500.0}
    r = client.post("/predict", json=[CUSTOMER, loyal]).json()
    assert r[0]["churn_probability"] > r[1]["churn_probability"]
    assert r[0]["high_risk"] is True


def test_rejects_invalid_input(client):
    assert client.post("/predict", json=[CUSTOMER | {"Contract": "Weekly"}]).status_code == 422
