"""Train and evaluate churn models.

Usage:
    python -m src.train                # trains, evaluates, saves models/churn_model.joblib
    python -m src.train --no-mlflow    # skip experiment tracking

Design notes:
- All preprocessing lives inside a sklearn Pipeline so the exact same
  transformations run at serving time. No leakage, no train/serve skew.
- Stratified split because churn is imbalanced (~26.5% positive).
- ROC-AUC and PR-AUC are the headline metrics; accuracy is reported but
  not trusted (a model predicting "no churn" for everyone gets 73%).
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import clean, load_raw, split_features_target

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def build_pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
        ],
        remainder=OneHotEncoder(handle_unknown="ignore"),
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
    }


def main(use_mlflow: bool = True) -> None:
    df = clean(load_raw())
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_experiment("telco-churn")
        except ImportError:
            print("mlflow not installed; continuing without tracking")

    results = {}
    fitted = {}
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipe
        print(f"{name}: {metrics}")

        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "random_state": RANDOM_STATE})
                mlflow.log_metrics(metrics)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], out_dir / "churn_model.joblib")
    (out_dir / "metrics.json").write_text(
        json.dumps({"best_model": best_name, "results": results}, indent=2)
    )
    print(f"\nBest model: {best_name} -> saved to models/churn_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    main(use_mlflow=not args.no_mlflow)
