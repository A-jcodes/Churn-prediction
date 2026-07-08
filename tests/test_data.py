"""Tests for data loading and cleaning.

These encode the data quality contract: if the upstream extract changes
shape or the known quirks regress, these fail loudly before training does.
"""

import pandas as pd
import pytest

from src.data import clean, load_raw, split_features_target


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="module")
def cleaned(raw) -> pd.DataFrame:
    return clean(raw)


def test_expected_shape(raw):
    assert raw.shape == (7043, 21)


def test_no_duplicate_customers(raw):
    assert raw["customerID"].duplicated().sum() == 0


def test_totalcharges_numeric_after_clean(cleaned):
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_blank_totalcharges_are_new_customers(raw):
    blanks = raw[raw["TotalCharges"].str.strip() == ""]
    assert len(blanks) == 11
    assert (blanks["tenure"] == 0).all()


def test_seniorcitizen_standardized(cleaned):
    assert set(cleaned["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_target_binarized(cleaned):
    X, y = split_features_target(cleaned)
    assert set(y.unique()) <= {0, 1}
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert 0.25 < y.mean() < 0.28  # known class balance
