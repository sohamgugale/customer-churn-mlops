"""Train, tune and compare churn models; log everything to MLflow.

Run:  python -m src.train            (writes models/, reports/metrics.json, mlruns/)
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, NUMERIC, TARGET, build

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
# Business costs used to pick the decision threshold: a missed churner loses ~1 year of margin,
# a retention offer to a customer who would have stayed costs one discount.
COST_FN, COST_FP = 500.0, 50.0


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def evaluate(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "precision": precision_score(y, yhat),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "threshold": thr,
    }


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    cost = [COST_FN * ((p < t) & (y == 1)).sum() + COST_FP * ((p >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(cost))])


def tune_xgb(X, y, n_trials=40) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        pipe = Pipeline([("prep", preprocessor()),
                         ("clf", xgb.XGBClassifier(**params, eval_metric="logloss", random_state=SEED, n_jobs=4))])
        return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params | {"cv_auc": study.best_value}


def main():
    raw = pd.read_csv(ROOT / "data" / "telco_churn.csv")
    df = build(raw)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("churn")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    best = tune_xgb(X_tr, y_tr, n_trials=int(os.environ.get('N_TRIALS', 40)))
    cv_auc_xgb = best.pop("cv_auc")
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=4),
        "lightgbm": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                       colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "xgboost_optuna": xgb.XGBClassifier(**best, eval_metric="logloss", random_state=SEED, n_jobs=4),
    }
    try:  # PyTorch neural network (optional dependency)
        from src.mlp import TorchMLPClassifier
        candidates["pytorch_mlp"] = TorchMLPClassifier()
    except ImportError:
        pass

    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", clf)])
        cv_auc = cv_auc_xgb if name == "xgboost_optuna" else cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        model = CalibratedClassifierCV(pipe, method="isotonic", cv=5).fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, p) | {"cv_roc_auc": cv_auc}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name} | (best if name == "xgboost_optuna" else {}))
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
        results[name], fitted[name] = m, model
        print(f"{name:22s} cv_auc={cv_auc:.3f} test_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}")

    champ = max(results, key=lambda k: results[k]["cv_roc_auc"])
    model = fitted[champ]
    p_te = model.predict_proba(X_te)[:, 1]
    # pick threshold on out-of-fold training predictions to avoid tuning on the test set
    from sklearn.model_selection import cross_val_predict
    p_oof = cross_val_predict(CalibratedClassifierCV(Pipeline([("prep", preprocessor()), ("clf", candidates[champ])]),
                                                     method="isotonic", cv=5), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = best_threshold(y_tr, p_oof)
    final = evaluate(y_te, p_te, thr)
    cost = lambda t: COST_FN * ((p_te < t) & (y_te == 1)).sum() + COST_FP * ((p_te >= t) & (y_te == 0)).sum()
    final["cost_at_0.5"], final["cost_at_threshold"] = float(cost(0.5)), float(cost(thr))
    # decile lift: churn rate in the top 10% scored customers vs base rate
    top = np.argsort(-p_te)[: len(p_te) // 10]
    final["top_decile_lift"] = float(y_te[top].mean() / y_te.mean())

    with mlflow.start_run(run_name=f"champion_{champ}"):
        mlflow.log_metrics({k: float(v) for k, v in final.items()})
        mlflow.sklearn.log_model(model, name="model", registered_model_name="churn_classifier",
                                 serialization_format="cloudpickle", input_example=X_te.head(3))

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "features": NUMERIC + CATEGORICAL}, ROOT / "models" / "churn_model.joblib")
    X_tr.assign(Churn=y_tr).to_parquet(ROOT / "models" / "train_reference.parquet")
    X_te.assign(Churn=y_te, score=p_te).to_parquet(ROOT / "models" / "test_scored.parquet")
    out = {"champion": champ, "models": results, "champion_test": final, "xgb_best_params": best,
           "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "churn_rate": float(y.mean())}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
