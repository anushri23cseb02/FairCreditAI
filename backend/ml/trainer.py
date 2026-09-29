"""
Training logic used by BOTH scripts/train_model.py (CLI) and
backend/api/routes_train.py (POST /train). One implementation, two entry
points, so the API can never drift from what the CLI script does.

Requires data/processed/{train,val,test}.csv to already exist (produced
by scripts/prepare_data.py). Retraining does NOT re-run the data pipeline
-- that is a separate, deliberate step, since it reads the raw 260MB CSV.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from backend.ml.feature_config import ALL_CATEGORICAL_FEATURES, ALL_NUMERIC_FEATURES, RANDOM_STATE, TARGET_COLUMN
from backend.ml.pipeline import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
METADATA_DIR = PROJECT_ROOT / "models" / "metadata"
REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"

FEATURE_COLUMNS = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES


class TrainingDataMissingError(RuntimeError):
    pass


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise TrainingDataMissingError(
            f"{path} not found. Run scripts/prepare_data.py before training."
        )
    return pd.read_csv(path)


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "confusion_matrix": {
            "labels": ["not_denied(0)", "denied(1)"],
            "matrix": confusion_matrix(y, pred).tolist(),
        },
        "classification_report": classification_report(y, pred, output_dict=True),
        "n_samples": int(len(y)),
        "positive_rate_actual": round(float(y.mean()), 4),
        "positive_rate_predicted": round(float(pred.mean()), 4),
    }


def train_and_persist(version: str = "v1") -> dict:
    """
    Fits Logistic Regression on train, evaluates on val and test, then
    refits on train+val for the final persisted artifact (test stays
    fully held out throughout). Writes the artifact, metadata, and
    updates the registry's active_version. Returns the metadata dict.
    """
    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)
    val_metrics = _evaluate(model, X_val, y_val)
    test_metrics = _evaluate(model, X_test, y_test)

    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    final_model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    final_model.fit(X_train_full, y_train_full)
    final_test_metrics = _evaluate(final_model, X_test, y_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    artifact_path = ARTIFACTS_DIR / f"model_{version}.joblib"
    joblib.dump(final_model, artifact_path)

    feature_names_out = final_model.named_steps["preprocessor"].get_feature_names_out().tolist()

    metadata = {
        "version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "LogisticRegression",
        "sklearn_pipeline": True,
        "hyperparameters": {"max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE},
        "feature_columns_raw": FEATURE_COLUMNS,
        "feature_names_transformed": feature_names_out,
        "target_column": TARGET_COLUMN,
        "training_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "validation_metrics_before_refit": val_metrics,
        "held_out_test_metrics_before_refit": test_metrics,
        "held_out_test_metrics_final_model": final_test_metrics,
        "notes": (
            "final_model is refit on train+val and is the artifact actually "
            "persisted; held_out_test_metrics_final_model is the metric set "
            "the API/dashboard display, since it reflects the exact "
            "persisted model evaluated on data it never saw."
        ),
    }
    metadata_path = METADATA_DIR / f"model_{version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    registry_path = REGISTRY_DIR / "registry.json"
    registry = json.load(open(registry_path)) if registry_path.exists() else {}
    registry["active_version"] = version
    registry.setdefault("versions", {})
    registry["versions"][version] = {
        "artifact_path": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
        "metadata_path": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
        "trained_at_utc": metadata["trained_at_utc"],
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    return metadata
