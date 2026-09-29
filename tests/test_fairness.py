"""
Tests backend/ml/fairness.py against a small synthetic dataset where the
group disparity is known by construction, so the correctness of the
metric math can be checked independently of the real trained model.
"""
import pandas as pd

from backend.ml.fairness import evaluate_attribute


def test_evaluate_attribute_detects_known_disparity():
    # Group A ("Male"): 10 rows, model denies 2 -> selection_rate 0.2
    # Group B ("Female"): 10 rows, model denies 6 -> selection_rate 0.6
    n = 10
    sensitive = pd.Series(["Male"] * n + ["Female"] * n)
    y_true = pd.Series([0] * n + [0] * n)  # ground truth not denied for all, isolates selection rate
    y_pred = pd.Series([1, 1] + [0] * (n - 2) + [1] * 6 + [0] * (n - 6))

    result = evaluate_attribute("sex", y_true, y_pred, sensitive)

    assert result["denial_rate_by_group"]["Male"] == 0.2
    assert result["denial_rate_by_group"]["Female"] == 0.6
    assert result["demographic_parity_difference"] == 0.4
    # disparate_impact_ratio = comparison(Female)/privileged(Male) = 0.6/0.2 = 3.0
    assert result["disparate_impact_ratio"] == 3.0


def test_evaluate_attribute_zero_difference_when_groups_identical():
    n = 10
    sensitive = pd.Series(["Male"] * n + ["Female"] * n)
    y_true = pd.Series([0] * (2 * n))
    y_pred = pd.Series(([1, 1] + [0] * (n - 2)) * 2)

    result = evaluate_attribute("sex", y_true, y_pred, sensitive)

    assert result["demographic_parity_difference"] == 0.0
    assert result["disparate_impact_ratio"] == 1.0


def test_evaluate_attribute_excludes_unknown_values():
    sensitive = pd.Series(["Male", "Female", "Not applicable", "Information not provided by applicant in mail, Internet, or telephone application"])
    y_true = pd.Series([0, 0, 0, 0])
    y_pred = pd.Series([0, 1, 1, 1])

    result = evaluate_attribute("sex", y_true, y_pred, sensitive)

    # only the Male and Female rows should count towards group sizes
    assert result["group_sample_sizes"]["Male"] == 1
    assert result["group_sample_sizes"]["Female"] == 1
