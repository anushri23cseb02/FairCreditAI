import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.model_service import get_model_service

client = TestClient(app)

VALID_APPLICATION = {
    "loan_amount_000s": 250,
    "applicant_income_000s": 80,
    "tract_to_msamd_income": 95,
    "population": 5000,
    "minority_population": 20,
    "number_of_owner_occupied_units": 1500,
    "number_of_1_to_4_family_units": 1800,
    "hud_median_family_income": 70000,
    "has_co_applicant": False,
    "loan_type_name": "Conventional",
    "loan_purpose_name": "Home purchase",
    "property_type_name": "One-to-four family dwelling (other than manufactured housing)",
    "owner_occupancy_name": "Owner-occupied as a principal dwelling",
    "lien_status_name": "Secured by a first lien",
    "preapproval_name": "Not applicable",
}


@pytest.fixture(scope="module", autouse=True)
def skip_if_no_model():
    if not get_model_service().is_ready:
        pytest.skip("No trained model artifact available in this environment.")


def test_predict_endpoint_returns_200_and_valid_shape():
    response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] in {"denied", "not_denied"}
    assert 0.0 <= body["denial_probability"] <= 1.0
    assert body["model_version"]


def test_predict_endpoint_rejects_negative_loan_amount():
    bad = dict(VALID_APPLICATION, loan_amount_000s=-10)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_category():
    bad = dict(VALID_APPLICATION, loan_type_name="Not a real loan type")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_required_field():
    bad = dict(VALID_APPLICATION)
    del bad["loan_amount_000s"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_batch_endpoint():
    response = client.post(
        "/predict/batch", json={"applications": [VALID_APPLICATION, VALID_APPLICATION]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "roc_auc" in response.json()["metrics"]


def test_fairness_endpoint():
    response = client.get("/fairness")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert "sex" in body and "race" in body and "ethnicity" in body
