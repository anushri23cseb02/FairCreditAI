"""
Phase 6 — explainability.

The persisted model is a scikit-learn Pipeline(preprocessor, LogisticRegression).
For a linear model, the logit score decomposes EXACTLY as:

    logit = intercept + sum_i( coefficient_i * transformed_feature_value_i )

so "which features contributed, and by how much" is not an approximation
(as SHAP/LIME would need to estimate for a black-box model) — it is the
literal arithmetic the model performs. This module exposes that
decomposition for a single prediction.

Language rule (per project spec): explanations describe association
with the model's output, never a causal claim about the applicant. We
say "contributed to the prediction", never "caused the denial".
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def explain_single(model: Pipeline, row_df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    row_df: a single-row DataFrame with the raw (untransformed) feature
        columns the model expects.
    Returns predicted label/probability plus the top_n features (by
    absolute contribution) driving THIS prediction's logit score.
    """
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
    predicted_label = int(probability >= 0.5)

    order = np.argsort(-np.abs(contributions))[:top_n]
    top_features = []
    for idx in order:
        raw_name = feature_names[idx]
        value = float(transformed[idx])
        contribution = float(contributions[idx])
        if abs(value) < 1e-9:
            continue  # a one-hot column that is 0 for this row contributed nothing
        top_features.append(
            {
                "feature": _humanize_feature_name(raw_name),
                "direction": "increased" if contribution > 0 else "decreased",
                "contribution_to_denial_logit": round(contribution, 4),
            }
        )

    return {
        "predicted_label": "denied" if predicted_label == 1 else "not_denied",
        "denial_probability": round(probability, 4),
        "top_contributing_features": top_features,
        "explanation_note": (
            "These features contributed to the model's prediction, in the "
            "direction and magnitude shown. This is not a causal statement "
            "about the applicant and should not be interpreted as the "
            "reason the applicant would be denied in reality."
        ),
    }


def _humanize_feature_name(raw_name: str) -> str:
    # sklearn ColumnTransformer names look like "numeric__loan_amount_000s"
    # or "categorical__loan_type_name_FHA-insured". Strip the transformer
    # prefix for a cleaner label in the UI/API response.
    if "__" in raw_name:
        raw_name = raw_name.split("__", 1)[1]
    return raw_name.replace("_", " ")


def global_feature_importance(model, top_n: int = 15) -> list[dict]:
    """
    Model-wide (not per-prediction) feature importance for the linear
    model: |coefficient| for a numeric feature, or the sum of
    |coefficient| across all one-hot levels for a categorical feature
    that was expanded into several columns. This tells you which raw
    inputs the model leans on most overall, as opposed to explain_single
    which explains one specific prediction.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    grouped: dict[str, float] = {}
    for name, coef in zip(feature_names, coefficients):
        clean = _humanize_feature_name(name)
        # Collapse one-hot levels of the same categorical feature back to
        # one entry, e.g. "loan type name Conventional" and
        # "loan type name FHA-insured" both roll up under "loan type name".
        base = clean
        for cat_col in ["loan type name", "loan purpose name", "property type name",
                        "owner occupancy name", "lien status name", "preapproval name"]:
            if clean.startswith(cat_col):
                base = cat_col
                break
        grouped[base] = grouped.get(base, 0.0) + abs(float(coef))

    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    total = sum(v for _, v in ranked) or 1.0
    return [
        {"feature": name, "importance": round(value, 4), "relative_importance": round(value / total, 4)}
        for name, value in ranked
    ]
