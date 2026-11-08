"""Post-training analysis of the champion model:
  * SHAP explainability (global feature importance + summary plot)
  * Fairness audit across gender and senior-citizen groups (responsible AI)
  * Data-drift monitoring with the Population Stability Index (PSI) on a simulated production batch
  * Retention-campaign simulation: expected savings from targeting the top-scored customers

Run after src.train:  python -m src.analyze
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import recall_score

from src.features import CATEGORICAL, NUMERIC

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(7)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    if len(np.unique(expected)) <= bins:  # categorical / binary feature: compare category shares
        cats = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
        e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
        return float(np.sum((a - e) * np.log(a / e)))
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def shap_report(art, X: pd.DataFrame) -> dict:
    # explain the underlying (uncalibrated) XGBoost pipeline from the first calibration fold
    pipe = art["model"].calibrated_classifiers_[0].estimator
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xt = prep.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = list(prep.get_feature_names_out())
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    shap.summary_plot(sv, Xt, feature_names=names, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "shap_summary.png", dpi=150); plt.close()
    return {k.split("__", 1)[1]: round(float(v), 4) for k, v in imp.head(10).items()}


def fairness(df: pd.DataFrame, thr: float) -> dict:
    out = {}
    for col, mapping in {"gender": None, "SeniorCitizen": {0: "non-senior", 1: "senior"}}.items():
        for g, sub in df.groupby(col):
            yhat = (sub["score"] >= thr).astype(int)
            name = mapping[g] if mapping else g
            out[f"{col}={name}"] = {"n": int(len(sub)), "recall": round(recall_score(sub["Churn"], yhat), 3),
                                   "flag_rate": round(float(yhat.mean()), 3),
                                   "false_positive_rate": round(float(yhat[sub["Churn"] == 0].mean()), 3)}
    return out


def drift(ref: pd.DataFrame) -> dict:
    """Simulate a production batch where prices rose 15% and more customers moved to month-to-month."""
    prod = ref.sample(2000, replace=True, random_state=1).copy()
    prod["MonthlyCharges"] *= RNG.normal(1.15, 0.03, len(prod))
    flip = RNG.random(len(prod)) < 0.2
    prod.loc[flip, "is_month_to_month"] = 1
    return {c: round(psi(ref[c].values, prod[c].values), 3) for c in ["MonthlyCharges", "tenure", "TotalCharges",
                                                                     "num_services", "is_month_to_month"]}


def campaign(df: pd.DataFrame, budget_share=0.2, offer_cost=50.0, save_rate=0.3, clv=500.0) -> dict:
    """Target the top `budget_share` of customers by churn score vs. random targeting."""
    k = int(len(df) * budget_share)
    top = df.nlargest(k, "score")
    rnd = df.sample(k, random_state=0)
    val = lambda s: float(s["Churn"].sum() * save_rate * clv - k * offer_cost)
    return {"customers_targeted": k, "churners_reached_model": int(top["Churn"].sum()),
            "churners_reached_random": int(rnd["Churn"].sum()),
            "net_value_model": round(val(top)), "net_value_random": round(val(rnd))}


def main():
    art = joblib.load(ROOT / "models" / "churn_model.joblib")
    test = pd.read_parquet(ROOT / "models" / "test_scored.parquet")
    ref = pd.read_parquet(ROOT / "models" / "train_reference.parquet")
    report = {
        "shap_top_features": shap_report(art, test[NUMERIC + CATEGORICAL]),
        "fairness": fairness(test, art["threshold"]),
        "psi_drift": drift(ref),
        "retention_campaign": campaign(test),
    }
    (ROOT / "reports" / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
