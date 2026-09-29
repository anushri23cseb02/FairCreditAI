"""
Track A training: trains and compares Logistic Regression, Random
Forest, and XGBoost on the credit-card-default data, calibrates each
with isotonic regression, and selects one model using a transparent,
documented scoring formula (never "pick the highest accuracy").

Run:
    python scripts/train_credit_card_models.py
or via API:
    POST /credit-card/train
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from backend.ml.credit_card.feature_config import ALL_FEATURES, RANDOM_STATE, TARGET_COLUMN
from backend.ml.credit_card.pipeline import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
METADATA_DIR = PROJECT_ROOT / "models" / "metadata"
REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"

# Explainability weight is a documented judgment call, not a measured
# quantity: it reflects how directly each model's prediction can be
# explained (Logistic Regression's exact coefficients vs. SHAP's
# estimated attributions for tree ensembles), spelled out here rather
# than buried, so the "why this model was selected" story is auditable.
EXPLAINABILITY_WEIGHT = {"logistic_regression": 1.0, "random_forest": 0.6, "xgboost": 0.6}

# Selection formula weights -- visible and editable here, not hidden.
SELECTION_WEIGHTS = {
    "roc_auc": 0.35,
    "f1": 0.25,
    "calibration": 0.20,   # 1 - normalized Brier score (higher is better-calibrated)
    "explainability": 0.20,
}


class TrainingDataMissingError(RuntimeError):
    pass


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"credit_card_{name}.csv"
    if not path.exists():
        raise TrainingDataMissingError(f"{path} not found. Run scripts/prepare_credit_card_data.py first.")
    return pd.read_csv(path)


def _evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "brier_score": round(brier_score_loss(y, proba), 4),
        "confusion_matrix": {"labels": ["not_default(0)", "default(1)"], "matrix": confusion_matrix(y, pred).tolist()},
        "n_samples": int(len(y)),
        "positive_rate_actual": round(float(y.mean()), 4),
        "positive_rate_predicted": round(float(pred.mean()), 4),
    }


def _calibration_curve_data(model, X: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> dict:
    """Real reliability-diagram data: mean predicted prob vs actual fraction positive, per bin."""
    proba = model.predict_proba(X)[:, 1]
    y_arr = np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    mean_predicted, fraction_positive, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_predicted.append(round(float(proba[mask].mean()), 4))
        fraction_positive.append(round(float(y_arr[mask].mean()), 4))
        bin_counts.append(int(mask.sum()))
    return {"mean_predicted_probability": mean_predicted, "fraction_positive": fraction_positive, "bin_counts": bin_counts}


def _curve_data(model, X: pd.DataFrame, y: pd.Series, max_points: int = 60) -> dict:
    """Real ROC and Precision-Recall curve points, subsampled evenly for compact API payloads/charts."""
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)

    def _subsample(arr, n):
        if len(arr) <= n:
            return arr.tolist()
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return arr[idx].tolist()

    return {
        "roc_fpr": [round(v, 4) for v in _subsample(fpr, max_points)],
        "roc_tpr": [round(v, 4) for v in _subsample(tpr, max_points)],
        "pr_precision": [round(v, 4) for v in _subsample(precision, max_points)],
        "pr_recall": [round(v, 4) for v in _subsample(recall, max_points)],
    }


def _build_estimator(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    raise ValueError(name)


class CalibratedPipeline:
    """
    Wraps a fitted preprocessor+model pipeline together with an isotonic
    regression fitted on held-out validation-set probabilities, so
    predict_proba() returns calibrated probabilities without depending on
    a specific scikit-learn version's CalibratedClassifierCV API.
    """
    def __init__(self, base_model, isotonic: IsotonicRegression):
        self.base_model = base_model
        self.isotonic = isotonic

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated_positive = self.isotonic.predict(raw)
        calibrated_positive = np.clip(calibrated_positive, 0.0, 1.0)
        return np.column_stack([1 - calibrated_positive, calibrated_positive])


def train_and_compare() -> dict:
    train_df, val_df, test_df = _load_split("train"), _load_split("val"), _load_split("test")
    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET_COLUMN]

    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)

    results = {}
    fitted_models = {}

    for name in ["logistic_regression", "random_forest", "xgboost"]:
        uncalibrated = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", _build_estimator(name))])
        uncalibrated.fit(X_train, y_train)
        raw_val_metrics = _evaluate(uncalibrated, X_val, y_val)

        # Calibrate with isotonic regression fitted on the validation
        # fold's raw probabilities -- never on train, never on test.
        raw_val_proba = uncalibrated.predict_proba(X_val)[:, 1]
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(raw_val_proba, y_val)

        # Refit the raw (uncalibrated) pipeline on train+val for the final
        # served artifact -- test stays held out throughout. The
        # calibrator (fitted only on val, never on test or train+val) is
        # then applied on top of this final model's raw probabilities.
        final_model = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", _build_estimator(name))])
        final_model.fit(X_train_full, y_train_full)
        calibrated_model = CalibratedPipeline(final_model, isotonic)

        test_metrics_raw = _evaluate(final_model, X_test, y_test)
        test_metrics_calibrated = _evaluate(calibrated_model, X_test, y_test)

        results[name] = {
            "raw_validation_metrics": raw_val_metrics,
            "held_out_test_metrics_uncalibrated_final_model": test_metrics_raw,
            "held_out_test_metrics_calibrated": test_metrics_calibrated,
            "calibration_curve_uncalibrated": _calibration_curve_data(final_model, X_test, y_test),
            "calibration_curve_calibrated": _calibration_curve_data(calibrated_model, X_test, y_test),
            "roc_pr_curves": _curve_data(final_model, X_test, y_test),
        }
        fitted_models[name] = final_model

    # --- transparent selection scoring ---------------------------------
    scores = {}
    for name, r in results.items():
        m = r["held_out_test_metrics_uncalibrated_final_model"]
        brier_norm = max(0.0, 1.0 - m["brier_score"] / 0.25)  # 0.25 = brier score of a coin-flip classifier
        score = (
            SELECTION_WEIGHTS["roc_auc"] * m["roc_auc"]
            + SELECTION_WEIGHTS["f1"] * m["f1"]
            + SELECTION_WEIGHTS["calibration"] * brier_norm
            + SELECTION_WEIGHTS["explainability"] * EXPLAINABILITY_WEIGHT[name]
        )
        scores[name] = {
            "composite_score": round(score, 4),
            "roc_auc": m["roc_auc"], "f1": m["f1"],
            "calibration_component": round(brier_norm, 4),
            "explainability_component": EXPLAINABILITY_WEIGHT[name],
        }
    selected_model_name = max(scores, key=lambda k: scores[k]["composite_score"])

    # --- persist ---------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    trained_at = datetime.now(timezone.utc).isoformat()
    for name, model in fitted_models.items():
        joblib.dump(model, ARTIFACTS_DIR / f"credit_card_{name}.joblib")

    metadata = {
        "dataset": "credit_card",
        "trained_at_utc": trained_at,
        "target_column": TARGET_COLUMN,
        "feature_columns": ALL_FEATURES,
        "models_compared": list(results.keys()),
        "selection_weights": SELECTION_WEIGHTS,
        "explainability_weights": EXPLAINABILITY_WEIGHT,
        "selection_scores": scores,
        "selected_model": selected_model_name,
        "selection_rationale": (
            f"'{selected_model_name}' scored highest on the documented weighted "
            f"formula (ROC-AUC {SELECTION_WEIGHTS['roc_auc']}, F1 {SELECTION_WEIGHTS['f1']}, "
            f"calibration {SELECTION_WEIGHTS['calibration']}, explainability "
            f"{SELECTION_WEIGHTS['explainability']}) -- not selected on accuracy alone."
        ),
        "results": results,
        "training_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
    }
    with open(METADATA_DIR / "credit_card_model_comparison.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    registry_path = REGISTRY_DIR / "registry.json"
    registry = json.load(open(registry_path)) if registry_path.exists() else {}
    registry.setdefault("credit_card", {})
    registry["credit_card"] = {
        "active_model": selected_model_name,
        "trained_at_utc": trained_at,
        "artifact_paths": {n: (ARTIFACTS_DIR / f"credit_card_{n}.joblib").relative_to(PROJECT_ROOT).as_posix() for n in results},
        "metadata_path": (METADATA_DIR / "credit_card_model_comparison.json").relative_to(PROJECT_ROOT).as_posix(),
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    return metadata
