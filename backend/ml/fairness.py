"""
Phase 5 — fairness evaluation.

Computes group-level fairness metrics for each protected attribute
defined in backend/ml/feature_config.py, comparing the documented
privileged group against the documented comparison group. Only rows
where the attribute is a known (non-"unknown") value are included in a
given comparison, since "unknown" cannot be fairly assigned to either
group.

This module makes NO claim that the model "is fair" or "is unbiased" —
it reports numeric group differences and lets the reader interpret them
against the accompanying documentation. See docstrings on each metric
for exactly what it measures and does not measure.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import recall_score

from backend.ml.feature_config import PROTECTED_ATTRIBUTES


def _known_mask(series: pd.Series, unknown_values: list[str]) -> pd.Series:
    return ~series.isin(unknown_values)


def evaluate_group_fairness(
    privileged: str,
    comparison: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_series: pd.Series,
    attribute_label: str = "attribute",
    column_name: str = "",
    outcome_name: str = "positive outcome",
) -> dict:
    """
    Dataset-agnostic core: compares a privileged vs comparison group on the
    same set of Fairlearn metrics. Used by both the HMDA fairness track
    (evaluate_attribute below) and the credit-card fairness track
    (backend/ml/credit_card/fairness.py), so the metric math is identical
    across both datasets rather than duplicated.

    outcome_name: what "selection"/prediction=1 means in plain language
    for this dataset (e.g. "denied" for HMDA, "predicted default" for the
    credit-card dataset) -- used only in the returned metric_definitions
    text, not in the calculation itself.
    """
    mask = sensitive_series.isin([privileged, comparison])
    if mask.sum() == 0:
        return {"error": f"No rows found for {privileged}/{comparison} in this dataset slice."}

    yt, yp, sf = y_true[mask], y_pred[mask], sensitive_series[mask]

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=yt,
        y_pred=yp,
        sensitive_features=sf,
    )

    by_group = mf.by_group.to_dict()
    priv_selection = by_group["selection_rate"].get(privileged)
    comp_selection = by_group["selection_rate"].get(comparison)

    dp_diff = demographic_parity_difference(yt, yp, sensitive_features=sf)
    dp_ratio = demographic_parity_ratio(yt, yp, sensitive_features=sf)
    eo_diff = equalized_odds_difference(yt, yp, sensitive_features=sf)

    tpr_priv = by_group["true_positive_rate"].get(privileged)
    tpr_comp = by_group["true_positive_rate"].get(comparison)
    equal_opportunity_difference = (
        None if tpr_priv is None or tpr_comp is None else round(tpr_comp - tpr_priv, 4)
    )

    disparate_impact_ratio = (
        None if priv_selection in (None, 0) else round(comp_selection / priv_selection, 4)
    )

    return {
        "attribute": attribute_label,
        "column": column_name,
        "privileged_group": privileged,
        "comparison_group": comparison,
        "group_sample_sizes": {
            privileged: int((sf == privileged).sum()),
            comparison: int((sf == comparison).sum()),
        },
        "selection_rate_by_group": {
            privileged: round(float(priv_selection), 4) if priv_selection is not None else None,
            comparison: round(float(comp_selection), 4) if comp_selection is not None else None,
        },
        "true_positive_rate_by_group": {
            privileged: round(float(tpr_priv), 4) if tpr_priv is not None else None,
            comparison: round(float(tpr_comp), 4) if tpr_comp is not None else None,
        },
        "false_positive_rate_by_group": {
            k: round(float(v), 4) for k, v in by_group["false_positive_rate"].items()
        },
        "demographic_parity_difference": round(float(dp_diff), 4),
        "demographic_parity_ratio": round(float(dp_ratio), 4),
        "disparate_impact_ratio": disparate_impact_ratio,
        "equal_opportunity_difference": equal_opportunity_difference,
        "equalized_odds_difference": round(float(eo_diff), 4),
        "metric_definitions": {
            "selection_rate_by_group": f"P(model predicts {outcome_name}) within each group.",
            "demographic_parity_difference": "max group selection rate minus min group selection rate. 0 = identical rates across groups.",
            "demographic_parity_ratio": "min group selection rate divided by max group selection rate. 1.0 = identical rates.",
            "disparate_impact_ratio": "comparison-group selection rate divided by privileged-group selection rate. Values below the commonly cited 0.80 'four-fifths rule' threshold are a widely used (not legally dispositive) flag for further review.",
            "true_positive_rate_by_group": f"of applicants who ACTUALLY have the {outcome_name} outcome, the fraction the model also predicts that way, per group.",
            "false_positive_rate_by_group": f"of applicants who do NOT actually have the {outcome_name} outcome, the fraction the model incorrectly predicts that way, per group.",
            "equal_opportunity_difference": "comparison-group TPR minus privileged-group TPR. 0 = the model is equally likely to correctly identify the outcome in both groups.",
            "equalized_odds_difference": "the larger of the TPR difference and FPR difference across groups (fairlearn's combined measure).",
        },
        "limitations": [
            "These are statistical outcome comparisons on one historical dataset, not a legal determination of discrimination.",
            "Rows with an unknown/not-provided value for this attribute are excluded from this specific comparison.",
            "Group sample sizes differ; metrics for smaller groups carry more statistical uncertainty.",
        ],
    }


def evaluate_attribute(
    attribute_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_series: pd.Series,
) -> dict:
    """
    HMDA-specific wrapper: attribute_name is one of "sex", "race",
    "ethnicity" (keys of PROTECTED_ATTRIBUTES). y_pred=1 means "denied".
    """
    cfg = PROTECTED_ATTRIBUTES[attribute_name]
    result = evaluate_group_fairness(
        cfg["privileged"], cfg["comparison"], y_true, y_pred, sensitive_series,
        attribute_label=attribute_name, column_name=cfg["column"], outcome_name="denied",
    )
    if "error" in result:
        return result
    # Keep the original key name ("denial_rate_by_group") for HMDA so the
    # existing API/frontend contract for this dataset is unchanged.
    result["denial_rate_by_group"] = result.pop("selection_rate_by_group")
    result["metric_definitions"]["denial_rate_by_group"] = result["metric_definitions"].pop("selection_rate_by_group")
    result["limitations"].append(
        "The model does not use this attribute as an input feature, but it can still show disparities if correlated features (e.g. geography-linked income fields) act as proxies."
    )
    return result


def evaluate_all(y_true: pd.Series, y_pred: pd.Series, protected_df: pd.DataFrame) -> dict:
    """protected_df must contain one column per PROTECTED_ATTRIBUTES entry."""
    return {
        name: evaluate_attribute(name, y_true, y_pred, protected_df[cfg["column"]])
        for name, cfg in PROTECTED_ATTRIBUTES.items()
    }
