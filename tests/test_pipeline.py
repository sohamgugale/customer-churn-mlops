from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analyze import psi
from src.features import CATEGORICAL, NUMERIC, build

ROOT = Path(__file__).resolve().parents[1]
RAW = pd.read_csv(ROOT / "data" / "telco_churn.csv")


def test_build_has_no_missing_values():
    df = build(RAW)
    assert df[NUMERIC + CATEGORICAL].isna().sum().sum() == 0
    assert set(df["Churn"].unique()) == {0, 1}


def test_blank_total_charges_are_new_customers():
    blank = RAW["TotalCharges"].astype(str).str.strip() == ""
    assert (RAW.loc[blank, "tenure"] == 0).all()
    assert (build(RAW).loc[blank, "TotalCharges"] == 0).all()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(1, 1, 5000)) > 0.25


@pytest.mark.skipif(not (ROOT / "data" / "features.parquet").exists(), reason="run src.etl_spark first")
def test_spark_and_pandas_features_match():
    spark = pd.read_parquet(ROOT / "data" / "features.parquet").sort_values("customerID").reset_index(drop=True)
    pdf = build(RAW).sort_values("customerID").reset_index(drop=True)
    for c in NUMERIC:
        np.testing.assert_allclose(spark[c].astype(float), pdf[c].astype(float), rtol=1e-9)
    assert (spark["tenure_bucket"] == pdf["tenure_bucket"]).all()
