"""
FairCredit AI - Phase 2 data pipeline.

Raw HMDA CSV -> validation -> cleaning -> target derivation ->
feature engineering -> train/val/test split -> processed CSVs +
a JSON summary report documenting every decision with real numbers.

Run from the project root:

    python scripts/prepare_data.py

Reads:
    data/raw/Washington_State_HDMA-2016.csv

Writes:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/reports/data_summary.json

This script performs NO model training. It only produces a clean,
leakage-free modelling table. Run scripts/train_model.py afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.feature_config import (  # noqa: E402
    ACTION_TO_LABEL,
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_BINARY_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    PROTECTED_ATTRIBUTES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_RAW_COLUMN,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Washington_State_HDMA-2016.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

PROTECTED_COLUMNS = [cfg["column"] for cfg in PROTECTED_ATTRIBUTES.values()]

# Columns we actually read from the ~50-column raw file (keeps memory down
# and makes every retained column traceable to feature_config.py).
USE_COLUMNS = (
    [TARGET_RAW_COLUMN]
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + PROTECTED_COLUMNS
    + ["co_applicant_sex_name"]
)


def validate_raw_file() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Place the HMDA Washington "
            "State 2016 CSV there before running this script."
        )
    df = pd.read_csv(RAW_PATH, usecols=lambda c: c in USE_COLUMNS)
    missing_cols = set(USE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Raw file is missing expected columns: {sorted(missing_cols)}. "
            "The schema does not match what backend/ml/feature_config.py expects."
        )
    return df


def derive_target(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    report["raw_row_count"] = int(len(df))
    report["action_taken_value_counts"] = (
        df[TARGET_RAW_COLUMN].value_counts(dropna=False).to_dict()
    )
    df = df[df[TARGET_RAW_COLUMN].isin(ACTION_TO_LABEL.keys())].copy()
    df[TARGET_COLUMN] = df[TARGET_RAW_COLUMN].map(ACTION_TO_LABEL).astype(int)
    report["rows_after_target_filter"] = int(len(df))
    report["rows_excluded_ambiguous_action"] = report["raw_row_count"] - report["rows_after_target_filter"]
    report["target_class_balance"] = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    return df


def clean_and_engineer(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    before = len(df)

    # Duplicate rows (exact duplicates only — HMDA rows are not naturally
    # unique-keyed in this extract, so we only drop true full-row dupes).
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)

    # Drop rows with non-positive or missing income/loan amount — these are
    # data-quality problems, not real applications, and would corrupt the
    # engineered ratio feature.
    before = len(df)
    df = df[(df["applicant_income_000s"].notna()) & (df["applicant_income_000s"] > 0)]
    df = df[(df["loan_amount_000s"].notna()) & (df["loan_amount_000s"] > 0)]
    report["rows_dropped_invalid_income_or_loan_amount"] = before - len(df)

    # Cap extreme outliers at the 99.5th percentile instead of dropping them,
    # so genuine (if unusual) applications are not discarded outright.
    for col in ["loan_amount_000s", "applicant_income_000s"]:
        cap = df[col].quantile(0.995)
        n_capped = int((df[col] > cap).sum())
        df[col] = df[col].clip(upper=cap)
        report.setdefault("outliers_capped_at_p99_5", {})[col] = n_capped

    # --- Feature engineering ---
    df["loan_to_income_ratio"] = df["loan_amount_000s"] / df["applicant_income_000s"]
    df["has_co_applicant"] = (df["co_applicant_sex_name"] != "No co-applicant").astype(int)

    # Missing-value report for the numeric/categorical model features
    # (imputation itself happens inside the sklearn pipeline at train time,
    # so inference uses the exact same fitted imputer — this is just an
    # audit of how much is missing going in).
    missing = df[ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES].isna().mean().round(4)
    report["missing_value_fraction_by_feature"] = missing.to_dict()

    return df


def fairness_slice_report(df: pd.DataFrame, report: dict) -> None:
    slice_report = {}
    for name, cfg in PROTECTED_ATTRIBUTES.items():
        col = cfg["column"]
        counts = df[col].value_counts(dropna=False).to_dict()
        known = df[~df[col].isin(cfg["unknown_values"])]
        slice_report[name] = {
            "column": col,
            "privileged_group": cfg["privileged"],
            "comparison_group": cfg["comparison"],
            "value_counts_all_rows": counts,
            "privileged_group_n": int((known[col] == cfg["privileged"]).sum()),
            "comparison_group_n": int((known[col] == cfg["comparison"]).sum()),
        }
    report["protected_attribute_slices"] = slice_report


def split_and_save(df: pd.DataFrame, report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
    keep_cols = feature_cols + [TARGET_COLUMN] + PROTECTED_COLUMNS
    df = df[keep_cols]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[TARGET_COLUMN], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET_COLUMN], random_state=RANDOM_STATE
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    report["split_sizes"] = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    report["split_strategy"] = (
        "70/15/15 train/val/test, stratified on the target, random_state="
        f"{RANDOM_STATE}."
    )


def main() -> None:
    report: dict = {}
    df = validate_raw_file()
    df = derive_target(df, report)
    df = clean_and_engineer(df, report)
    fairness_slice_report(df, report)
    split_and_save(df, report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("Data pipeline complete.")
    print(f"  Rows after target filter: {report['rows_after_target_filter']}")
    print(f"  Target balance: {report['target_class_balance']}")
    print(f"  Split sizes: {report['split_sizes']}")
    print(f"  Summary written to {REPORTS_DIR / 'data_summary.json'}")


if __name__ == "__main__":
    main()
