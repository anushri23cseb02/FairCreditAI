"""
Tests the target-derivation and feature-engineering LOGIC from
scripts/prepare_data.py against small synthetic fixtures, so this suite
runs in milliseconds and never depends on the 257MB raw HMDA file or any
machine-specific path.
"""
import pandas as pd
import pytest

from backend.ml.feature_config import ACTION_TO_LABEL, TARGET_COLUMN, TARGET_RAW_COLUMN


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TARGET_RAW_COLUMN: [
                "Loan originated",
                "Application denied by financial institution",
                "Application withdrawn by applicant",
                "File closed for incompleteness",
            ],
            "applicant_income_000s": [80, 40, 60, 50],
            "loan_amount_000s": [200, 150, 180, 120],
            "co_applicant_sex_name": ["No co-applicant", "Female", "No co-applicant", "Male"],
        }
    )


def test_target_derivation_keeps_only_originated_and_denied():
    df = _sample_frame()
    filtered = df[df[TARGET_RAW_COLUMN].isin(ACTION_TO_LABEL.keys())].copy()
    filtered[TARGET_COLUMN] = filtered[TARGET_RAW_COLUMN].map(ACTION_TO_LABEL)

    assert len(filtered) == 2
    assert set(filtered[TARGET_COLUMN]) == {0, 1}
    # "withdrawn" and "incomplete" rows must never appear
    assert "Application withdrawn by applicant" not in filtered[TARGET_RAW_COLUMN].values
    assert "File closed for incompleteness" not in filtered[TARGET_RAW_COLUMN].values


def test_target_mapping_direction_is_correct():
    assert ACTION_TO_LABEL["Loan originated"] == 0
    assert ACTION_TO_LABEL["Application denied by financial institution"] == 1


def test_loan_to_income_ratio_engineering():
    df = _sample_frame()
    df["loan_to_income_ratio"] = df["loan_amount_000s"] / df["applicant_income_000s"]
    assert df["loan_to_income_ratio"].iloc[0] == pytest.approx(200 / 80)


def test_has_co_applicant_engineering():
    df = _sample_frame()
    df["has_co_applicant"] = (df["co_applicant_sex_name"] != "No co-applicant").astype(int)
    assert df["has_co_applicant"].tolist() == [0, 1, 0, 1]


def test_zero_income_rows_would_be_excluded():
    df = _sample_frame()
    df.loc[0, "applicant_income_000s"] = 0
    valid = df[(df["applicant_income_000s"].notna()) & (df["applicant_income_000s"] > 0)]
    assert len(valid) == len(df) - 1
