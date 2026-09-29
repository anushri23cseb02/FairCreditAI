"""
Track A explainability.

Logistic Regression: exact coefficient decomposition (same technique as
the HMDA track -- see backend/ml/explain.py).
Random Forest / XGBoost: real SHAP TreeExplainer values -- not an
approximation of an approximation; this is the actual SHAP library
computing actual Shapley-value estimates against the actual fitted tree
ensemble. If SHAP fails for any reason, this module returns an explicit
"unavailable" result rather than silently falling back to a fake
explanation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _humanize(raw_name: str) -> str:
    if "__" in raw_name:
        raw_name = raw_name.split("__", 1)[1]
    return raw_name.replace("_", " ")


def explain_single(model_name: str, model: Pipeline, row_df: pd.DataFrame, top_n: int = 5) -> dict:
    if model_name == "logistic_regression":
        return _explain_logistic(model, row_df, top_n)
    if model_name in ("random_forest", "xgboost"):
        return _explain_tree_shap(model_name, model, row_df, top_n)
    raise ValueError(f"Unknown model_name: {model_name}")


def _explain_logistic(model: Pipeline, row_df: pd.DataFrame, top_n: int) -> dict:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(row_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)[0]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    contributions = coefficients * transformed
    logit = intercept + float(contributions.sum())
    probability = float(1 / (1 + np.exp(-logit)))

    order = np.argsort(-np.abs(contributions))[:top_n]
    top_features = []
    for idx in order:
        value = float(transformed[idx])
        if abs(value) < 1e-9:
            continue
        contribution = float(contributions[idx])
        top_features.append({
            "feature": _humanize(feature_names[idx]),
            "direction": "increases risk" if contribution > 0 else "decreases risk",
            "contribution": round(contribution, 4),
        })

    return {
        "explanation_method": "Exact coefficient decomposition (Logistic Regression)",
        "default_probability": round(probability, 4),
        "top_contributing_features": top_features,
        "available": True,
    }


def _explain_tree_shap(model_name: str, model: Pipeline, row_df: pd.DataFrame, top_n: int) -> dict:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    try:
        transformed = preprocessor.transform(row_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        # RandomForestClassifier returns a list [class0_values, class1_values];
        # XGBClassifier returns a single array for the positive class.
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]
            if values.ndim > 1:
                values = values[:, 1] if values.shape[-1] == 2 else values.ravel()

        proba = float(model.predict_proba(row_df)[0, 1])

        order = np.argsort(-np.abs(values))[:top_n]
        top_features = []
        for idx in order:
            contribution = float(values[idx])
            top_features.append({
                "feature": _humanize(feature_names[idx]),
                "direction": "increases risk" if contribution > 0 else "decreases risk",
                "contribution": round(contribution, 4),
            })

        return {
            "explanation_method": f"SHAP TreeExplainer ({model_name})",
            "default_probability": round(proba, 4),
            "top_contributing_features": top_features,
            "available": True,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("SHAP explanation failed for %s", model_name)
        return {
            "explanation_method": f"SHAP TreeExplainer ({model_name})",
            "available": False,
            "error": "Explanation unavailable for this prediction.",
            "technical_detail": str(exc),
        }


def global_feature_importance(model_name: str, model: Pipeline, background_df: pd.DataFrame, top_n: int = 15) -> dict:
    """
    Global importance: for Logistic Regression, summed |coefficient| per
    raw feature (categoricals collapsed across one-hot levels). For tree
    models, mean |SHAP value| across a background sample -- real SHAP
    output, not the model's built-in (less faithful) impurity importance.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name == "logistic_regression":
        coefficients = classifier.coef_[0]
        grouped: dict[str, float] = {}
        for name, coef in zip(feature_names, coefficients):
            base = _humanize(name).split(" ")[0] if "__" not in name else _humanize(name)
            grouped[_humanize(name)] = grouped.get(_humanize(name), 0.0) + abs(float(coef))
        ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        total = sum(v for _, v in ranked) or 1.0
        return {
            "explanation_method": "Sum of |coefficient| (Logistic Regression)",
            "available": True,
            "features": [{"feature": n, "importance": round(v, 4), "relative_importance": round(v / total, 4)} for n, v in ranked],
        }

    try:
        transformed = preprocessor.transform(background_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            values = shap_values[1]
        else:
            values = shap_values
            if values.ndim == 3:
                values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        ranked_idx = np.argsort(-mean_abs)[:top_n]
        total = mean_abs[ranked_idx].sum() or 1.0
        return {
            "explanation_method": f"Mean |SHAP value| over a {len(background_df)}-row sample ({model_name})",
            "available": True,
            "features": [
                {"feature": _humanize(feature_names[i]), "importance": round(float(mean_abs[i]), 4),
                 "relative_importance": round(float(mean_abs[i] / total), 4)}
                for i in ranked_idx
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Global SHAP importance failed for %s", model_name)
        return {"explanation_method": f"SHAP ({model_name})", "available": False, "error": "Explanation unavailable.", "technical_detail": str(exc)}
