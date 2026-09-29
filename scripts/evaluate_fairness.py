"""
FairCredit AI - Phase 5 runner.

Loads the active model + held-out test set, generates predictions, and
computes group fairness metrics for every protected attribute. Writes
data/reports/fairness_report.json.

Run after scripts/train_model.py:

    python scripts/evaluate_fairness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.fairness import evaluate_all  # noqa: E402
from backend.ml.feature_config import (  # noqa: E402
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    PROTECTED_ATTRIBUTES,
    TARGET_COLUMN,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry" / "registry.json"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

FEATURE_COLUMNS = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES


def main() -> None:
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    active_version = registry["active_version"]
    artifact_path = PROJECT_ROOT / registry["versions"][active_version]["artifact_path"]
    model = joblib.load(artifact_path)

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test_df[FEATURE_COLUMNS]
    y_true = test_df[TARGET_COLUMN]
    y_pred = pd.Series(model.predict(X_test), index=test_df.index)

    protected_df = test_df[[cfg["column"] for cfg in PROTECTED_ATTRIBUTES.values()]]
    report = evaluate_all(y_true, y_pred, protected_df)
    report["_meta"] = {
        "model_version": active_version,
        "test_rows": int(len(test_df)),
        "overall_denial_rate_actual": round(float(y_true.mean()), 4),
        "overall_denial_rate_predicted": round(float(y_pred.mean()), 4),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "fairness_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("Fairness evaluation complete.")
    for name in PROTECTED_ATTRIBUTES:
        r = report[name]
        print(f"  [{name}] {r['privileged_group']} vs {r['comparison_group']}: "
              f"denial rates {r['denial_rate_by_group']}, "
              f"disparate impact ratio={r['disparate_impact_ratio']}")
    print(f"  Report written to {REPORTS_DIR / 'fairness_report.json'}")


if __name__ == "__main__":
    main()
