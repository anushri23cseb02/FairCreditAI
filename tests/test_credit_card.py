import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.credit_card_service import get_credit_card_model_service

client = TestClient(app)

VALID_APPLICATION = {
    "LIMIT_BAL": 200000, "AGE": 35, "EDUCATION": "University", "MARRIAGE": "Married",
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 45000, "BILL_AMT4": 42000, "BILL_AMT5": 40000, "BILL_AMT6": 38000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
}


@pytest.fixture(scope="module", autouse=True)
def skip_if_no_model():
    if not get_credit_card_model_service().is_ready:
        pytest.skip("Credit-card models not trained in this environment.")


def test_predict_default_model():
    r = client.post("/credit-card/predict", json=VALID_APPLICATION)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["explanation_available"] is True
    assert len(body["top_contributing_features"]) > 0


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
def test_predict_each_model_explicitly(model_name):
    r = client.post("/credit-card/predict", json=dict(VALID_APPLICATION, model_name=model_name))
    assert r.status_code == 200
    assert r.json()["model_used"] == model_name


def test_predict_rejects_invalid_pay_status():
    r = client.post("/credit-card/predict", json=dict(VALID_APPLICATION, PAY_0=99))
    assert r.status_code == 422


def test_predict_rejects_invalid_education_category():
    r = client.post("/credit-card/predict", json=dict(VALID_APPLICATION, EDUCATION="Not A Real Category"))
    assert r.status_code == 422


def test_predict_rejects_unknown_model_name():
    r = client.post("/credit-card/predict", json=dict(VALID_APPLICATION, model_name="not_a_real_model"))
    assert r.status_code == 422  # caught by the Pydantic validator


def test_batch_prediction():
    r = client.post("/credit-card/predict/batch", json={"applications": [VALID_APPLICATION] * 3})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_prediction_history_endpoint_never_raises():
    r = client.get("/credit-card/predictions")
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)


def test_model_info_shows_transparent_selection():
    r = client.get("/credit-card/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert set(body["models_compared"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert body["selected_model"] in body["models_compared"]
    assert "selection_rationale" in body
    # the selection weights must sum to 1.0 -- a sanity check on the formula itself
    assert abs(sum(body["selection_weights"].values()) - 1.0) < 1e-6


def test_model_comparison_has_all_three_models():
    r = client.get("/credit-card/model/comparison")
    assert r.status_code == 200
    results = r.json()["results"]
    assert set(results.keys()) == {"logistic_regression", "random_forest", "xgboost"}
    for name, r_ in results.items():
        assert "held_out_test_metrics_uncalibrated_final_model" in r_
        assert "calibration_curve_calibrated" in r_


def test_feature_importance_available():
    r = client.get("/credit-card/model/feature-importance")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["available"] is True
    assert len(body["features"]) > 0


def test_dataset_info():
    r = client.get("/credit-card/dataset/info")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["dataset_name"] == "UCI Default of Credit Card Clients"


def test_fairness_audit_has_sex_and_age_band():
    r = client.get("/credit-card/fairness")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert "sex" in body and "age_band" in body
    assert body["sex"]["privileged_group"] == "Male"
    assert "disclaimer" in body


def test_fairness_mitigation_shows_before_and_after():
    r = client.get("/credit-card/fairness/mitigation")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert "predictive_metrics_before" in body and "predictive_metrics_after" in body
    assert "fairness_before" in body and "fairness_after" in body
