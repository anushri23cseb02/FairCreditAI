"""
Track A fairness audit and mitigation.

Reuses the same Fairlearn metric core as the HMDA track
(backend/ml/fairness.py::evaluate_group_fairness), applied to SEX (the
only true protected attribute in this dataset) and, as a clearly labelled
secondary proxy check, an AGE band -- see feature_config.py for why AGE
is handled differently from SEX.

Mitigation: Fairlearn's ThresholdOptimizer, applied post-hoc to the
selected model's probabilities, constrained on "equalized_odds". Before
and after metrics are both computed from real model output on the same
held-out test set -- nothing here is asserted without a matching
computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer

from backend.ml.credit_card.feature_config import (
    AGE_BAND_COLUMN, AGE_BAND_COMPARISON, AGE_BAND_PRIVILEGED, PROTECTED_ATTRIBUTES,
)
from backend.ml.fairness import evaluate_group_fairness


def audit_sex(y_true: pd.Series, y_pred: pd.Series, sex_series: pd.Series) -> dict:
    cfg = PROTECTED_ATTRIBUTES["sex"]
    result = evaluate_group_fairness(
        cfg["privileged"], cfg["comparison"], y_true, y_pred, sex_series,
        attribute_label="sex", column_name=cfg["column"], outcome_name="predicted default",
    )
    if "error" not in result:
        result["limitations"].append(
            "SEX is used only for this audit; it is not given to the model as a prediction feature."
        )
    return result


def audit_age_band(y_true: pd.Series, y_pred: pd.Series, age_band_series: pd.Series) -> dict:
    result = evaluate_group_fairness(
        AGE_BAND_PRIVILEGED, AGE_BAND_COMPARISON, y_true, y_pred, age_band_series,
        attribute_label="age_band", column_name=AGE_BAND_COLUMN, outcome_name="predicted default",
    )
    if "error" not in result:
        result["limitations"].append(
            "AGE (unlike SEX) IS used as a model feature, so this is a disparate-impact/proxy "
            "check on a feature the model sees, not a same-treatment guarantee. The "
            "35-and-over / under-35 cutoff is a methodological choice made for this audit, "
            "not a scientifically privileged threshold."
        )
    return result


def mitigate_with_threshold_optimizer(
    base_model, X_train: pd.DataFrame, y_train: pd.Series, sensitive_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, sensitive_test: pd.Series,
) -> dict:
    """
    Fits a Fairlearn ThresholdOptimizer (constraint="equalized_odds") on
    top of the already-trained base_model's predict_proba, using SEX as
    the sensitive feature, fit on train and evaluated on the held-out
    test set. Returns real before/after predictive AND fairness metrics
    -- never asserted, always recomputed from the actual post-processed
    predictions.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    class _ProbaWrapper:
        """ThresholdOptimizer needs .predict_proba(X) -> array; wraps the pipeline directly (already provides this)."""
        def __init__(self, model):
            self.model = model
            # A dummy fitted-looking attribute so sklearn's check_is_fitted
            # recognizes this wrapper as fitted (it wraps an already-fitted
            # pipeline; ThresholdOptimizer's own fit() call below is only
            # calibrating thresholds, not fitting model.predict_proba itself).
            self.classes_ = np.array([0, 1])
        def fit(self, X, y):
            return self
        def predict(self, X):
            return (self.model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        def predict_proba(self, X):
            return self.model.predict_proba(X)

    wrapped = _ProbaWrapper(base_model)
    optimizer = ThresholdOptimizer(
        estimator=wrapped, constraints="equalized_odds", predict_method="predict_proba", prefit=True,
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)

    y_pred_before = (base_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    y_pred_after = optimizer.predict(X_test, sensitive_features=sensitive_test, random_state=42)

    def _predictive_metrics(y_true, y_pred, proba=None):
        m = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred), 4),
        }
        if proba is not None:
            m["roc_auc"] = round(roc_auc_score(y_true, proba), 4)
        return m

    proba_before = base_model.predict_proba(X_test)[:, 1]
    cfg = PROTECTED_ATTRIBUTES["sex"]

    return {
        "method": "Fairlearn ThresholdOptimizer (constraint=equalized_odds, fit on train, evaluated on held-out test)",
        "predictive_metrics_before": _predictive_metrics(y_test, y_pred_before, proba_before),
        "predictive_metrics_after": _predictive_metrics(y_test, y_pred_after),
        "fairness_before": evaluate_group_fairness(
            cfg["privileged"], cfg["comparison"], y_test, pd.Series(y_pred_before, index=y_test.index),
            sensitive_test, attribute_label="sex", column_name=cfg["column"], outcome_name="predicted default",
        ),
        "fairness_after": evaluate_group_fairness(
            cfg["privileged"], cfg["comparison"], y_test, pd.Series(y_pred_after, index=y_test.index),
            sensitive_test, attribute_label="sex", column_name=cfg["column"], outcome_name="predicted default",
        ),
        "note": (
            "Fairness metrics can trade off against predictive metrics -- compare both "
            "the predictive and fairness blocks before and after, rather than reading "
            "either in isolation. Mitigation does not make the model 'fair'; it changes "
            "where the equalized-odds trade-off point sits."
        ),
    }
