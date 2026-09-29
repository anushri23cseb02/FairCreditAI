from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.model_service import get_model_service

client = TestClient(app)
HMDA_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module", autouse=True)
def skip_if_no_model():
    if not get_model_service().is_ready:
        pytest.skip("No trained model artifact available in this environment.")


def _hmda_processed_data_available() -> bool:
    return all(
        (HMDA_PROCESSED_DIR / name).exists()
        for name in ("train.csv", "val.csv", "test.csv")
    )


def test_feature_importance_endpoint():
    response = client.get("/model/feature-importance")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert len(body["features"]) > 0
    assert "feature" in body["features"][0]
    assert "importance" in body["features"][0]


def test_train_endpoint_retrains_and_reloads_model():
    if not _hmda_processed_data_available():
        pytest.skip("HMDA processed data (data/processed/{train,val,test}.csv) not available in this environment.")
    response = client.post("/train")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "trained"
    assert "roc_auc" in body["test_metrics"]
    assert 0.5 <= body["test_metrics"]["roc_auc"] <= 1.0

    # the reloaded model must still serve predictions afterwards
    payload = {
        "loan_amount_000s": 250, "applicant_income_000s": 80,
        "tract_to_msamd_income": 95, "population": 5000, "minority_population": 20,
        "number_of_owner_occupied_units": 1500, "number_of_1_to_4_family_units": 1800,
        "hud_median_family_income": 70000, "has_co_applicant": False,
        "loan_type_name": "Conventional", "loan_purpose_name": "Home purchase",
        "property_type_name": "One-to-four family dwelling (other than manufactured housing)",
        "owner_occupancy_name": "Owner-occupied as a principal dwelling",
        "lien_status_name": "Secured by a first lien", "preapproval_name": "Not applicable",
    }
    pred_response = client.post("/predict", json=payload)
    assert pred_response.status_code == 200
