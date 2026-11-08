"""PySpark ETL: the same cleaning + feature engineering as src/features.py, written for Spark so
it scales past a single machine, plus a Spark MLlib baseline (logistic regression + GBT).

Run:  python -m src.etl_spark      (writes data/features.parquet, reports/spark_mllib.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.features import CATEGORICAL, NUMERIC, SERVICE_COLS

ROOT = Path(__file__).resolve().parents[1]


def spark_session(app="churn-etl") -> SparkSession:
    return (SparkSession.builder.master("local[*]").appName(app)
            .config("spark.sql.shuffle.partitions", "8").getOrCreate())


def build_features(df: DataFrame) -> DataFrame:
    tc = F.trim(F.col("TotalCharges"))
    df = (df.withColumn("TotalCharges", F.when(tc == "", 0.0).otherwise(tc.cast("double")))
            .withColumn("Churn", (F.col("Churn") == "Yes").cast("int")))
    df = df.withColumn("num_services", sum((F.col(c) == "Yes").cast("int") for c in SERVICE_COLS))
    avg = F.when(F.col("tenure") > 0, F.col("TotalCharges") / F.greatest(F.col("tenure"), F.lit(1))) \
           .otherwise(F.col("MonthlyCharges"))
    df = df.withColumn("avg_monthly_spend", avg)
    df = df.withColumn("charge_ratio", F.coalesce(
        F.col("MonthlyCharges") / F.when(F.col("avg_monthly_spend") != 0, F.col("avg_monthly_spend")), F.lit(1.0)))
    df = (df.withColumn("is_month_to_month", (F.col("Contract") == "Month-to-month").cast("int"))
            .withColumn("auto_pay", F.col("PaymentMethod").contains("automatic").cast("int"))
            .withColumn("has_fiber", (F.col("InternetService") == "Fiber optic").cast("int"))
            .withColumn("tenure_bucket",
                        F.when(F.col("tenure") <= 6, "0-6").when(F.col("tenure") <= 12, "7-12")
                         .when(F.col("tenure") <= 24, "13-24").when(F.col("tenure") <= 48, "25-48")
                         .otherwise("49-72")))
    return df


def mllib_baseline(df: DataFrame) -> dict:
    idx = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[c + "_idx" for c in CATEGORICAL], outputCols=[c + "_oh" for c in CATEGORICAL])
    vec = VectorAssembler(inputCols=NUMERIC + [c + "_oh" for c in CATEGORICAL], outputCol="features")
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    ev = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderROC")
    out = {}
    for name, clf in {"spark_logistic_regression": LogisticRegression(labelCol="Churn", regParam=0.01),
                      "spark_gbt": GBTClassifier(labelCol="Churn", maxIter=60, maxDepth=3, seed=42)}.items():
        model = Pipeline(stages=idx + [ohe, vec, clf]).fit(train)
        out[name] = round(ev.evaluate(model.transform(test)), 4)
    return out


def main():
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    raw = spark.read.csv(str(ROOT / "data" / "telco_churn.csv"), header=True, inferSchema=True)
    feats = build_features(raw).cache()
    feats.write.mode("overwrite").parquet(str(ROOT / "data" / "features.parquet"))
    print("rows:", feats.count(), "| churn rate:", round(feats.agg(F.avg("Churn")).first()[0], 4))
    res = mllib_baseline(feats)
    print(res)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "spark_mllib.json").write_text(json.dumps(res, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
