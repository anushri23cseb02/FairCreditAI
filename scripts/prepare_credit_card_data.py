"""
Track A (UCI Default of Credit Card Clients) data pipeline.

Run from the project root:
    python scripts/prepare_credit_card_data.py

Reads:  data/raw/default_of_credit_card_clients.csv
Writes: data/processed/credit_card_train.csv / _val.csv / _test.csv
        data/reports/data_summary_credit_card.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.credit_card.feature_config import (  # noqa: E402
    AGE_BAND_COLUMN, AGE_BAND_CUTOFF, ALL_FEATURES, EDUCATION_LABELS, EDUCATION_OTHER_LABEL,
    MARRIAGE_LABELS, MARRIAGE_OTHER_LABEL, PROTECTED_ATTRIBUTES, RANDOM_STATE,
    TARGET_COLUMN, TARGET_RAW_COLUMN,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default_of_credit_card_clients.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

SEX_LABELS = {1: "Male", 2: "Female"}


def main() -> None:
    report: dict = {"dataset_name": "UCI Default of Credit Card Clients"}

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}.")
    df = pd.read_csv(RAW_PATH)
    report["raw_row_count"] = int(len(df))
    report["raw_column_count"] = int(df.shape[1])

    # --- target: already binary, just rename for clarity -------------
    df = df.rename(columns={TARGET_RAW_COLUMN: TARGET_COLUMN})
    report["target_class_balance"] = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()

    # --- clean: exact duplicates (excluding ID), documented undocumented codes ---
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    report["duplicate_rows_dropped"] = before - len(df)

    report["education_raw_value_counts"] = df["EDUCATION"].value_counts().to_dict()
    report["marriage_raw_value_counts"] = df["MARRIAGE"].value_counts().to_dict()
    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_LABELS).fillna(EDUCATION_OTHER_LABEL)
    df["MARRIAGE"] = df["MARRIAGE"].map(MARRIAGE_LABELS).fillna(MARRIAGE_OTHER_LABEL)

    # --- protected attribute: map SEX codes to labels -----------------
    df["SEX"] = df["SEX"].map(SEX_LABELS)

    # --- age band for the secondary fairness proxy check ---------------
    df[AGE_BAND_COLUMN] = df["AGE"].apply(lambda a: "35 and over" if a >= AGE_BAND_CUTOFF else "Under 35")
    report["age_band_counts"] = df[AGE_BAND_COLUMN].value_counts().to_dict()

    report["missing_value_fraction_by_feature"] = df[ALL_FEATURES].isna().mean().round(4).to_dict()

    # --- split: 70/15/15 stratified, same seed as the HMDA track -------
    keep_cols = ALL_FEATURES + [TARGET_COLUMN, "SEX", AGE_BAND_COLUMN]
    df = df[keep_cols]
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[TARGET_COLUMN], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[TARGET_COLUMN], random_state=RANDOM_STATE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "credit_card_train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "credit_card_val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "credit_card_test.csv", index=False)
    report["split_sizes"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    report["split_strategy"] = f"70/15/15 train/val/test, stratified on target, random_state={RANDOM_STATE}"

    report["protected_attributes"] = {
        name: {"column": cfg["column"], "privileged": cfg["privileged"], "comparison": cfg["comparison"]}
        for name, cfg in PROTECTED_ATTRIBUTES.items()
    }
    report["leakage_review"] = (
        "BILL_AMT1-6 and PAY_AMT1-6 are the 6 statement months preceding the "
        "target month (next month's default) -- historical behaviour, not "
        "post-decision information. No denial-reason-style or outcome-adjacent "
        "columns exist in this dataset."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_summary_credit_card.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("Credit-card data pipeline complete.")
    print(f"  Rows after cleaning: {len(df)}")
    print(f"  Target balance: {report['target_class_balance']}")
    print(f"  Split sizes: {report['split_sizes']}")


if __name__ == "__main__":
    main()
