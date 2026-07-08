"""Data loading and cleaning for the Telco churn dataset.

Handles the known data quality issues in the IBM Telco sample data:
- TotalCharges stored as text with 11 blank strings (all tenure = 0 customers)
- SeniorCitizen encoded as 0/1 while every other binary column is Yes/No
"""

from pathlib import Path

import pandas as pd

RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COLUMN = "customerID"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV. Defaults to data/ inside the project root."""
    if path is None:
        path = project_root() / "data" / RAW_FILENAME
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning decisions.

    1. TotalCharges: coerce to numeric. The 11 blanks belong to brand new
       customers (tenure = 0) who have not been billed yet, so the true
       total charged is 0, not missing. Impute 0 rather than drop.
    2. SeniorCitizen: map 0/1 to No/Yes for consistency with other
       categorical columns (simplifies encoding and plotting).
    3. Strip whitespace from column values defensively.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].str.strip()

    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with the ID column dropped and target binarized."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, ID_COLUMN])
    return X, y
