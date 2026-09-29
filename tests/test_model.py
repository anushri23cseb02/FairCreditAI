"""
Tests the trained model artifact loads and produces sane single/batch
predictions. Skips (rather than fails) if no artifact has been trained
yet in this environment, since scripts/train_model.py requires the raw
dataset which is not committed to git.
"""
import pytest

from backend.services.model_service import get_model_service

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


@pytest.fixture(scope="module")
def service():
    svc = get_model_service()
    if not svc.is_ready:
        pytest.skip(f"No trained model artifact available: {svc.load_error}")
    return svc


def test_model_loads_and_reports_ready(service):
    assert service.is_ready
    info = service.get_model_info()
    assert info["ready"] is True
    assert info["model_type"] == "LogisticRegression"


def test_single_prediction_has_valid_probability(service):
    result = service.predict(VALID_APPLICATION)
    assert result["predicted_label"] in {"denied", "not_denied"}
    assert 0.0 <= result["denial_probability"] <= 1.0
    assert isinstance(result["top_contributing_features"], list)
    assert len(result["top_contributing_features"]) > 0


def test_batch_prediction_matches_single(service):
    batch = service.predict_batch([VALID_APPLICATION, VALID_APPLICATION])
    single = service.predict(VALID_APPLICATION)
    assert len(batch) == 2
    assert batch[0]["denial_probability"] == pytest.approx(single["denial_probability"])
    assert batch[0]["denial_probability"] == pytest.approx(batch[1]["denial_probability"])


def test_metrics_available(service):
    metrics = service.get_metrics()
    assert metrics["ready"] is True
    assert "roc_auc" in metrics["metrics"]
    assert 0.5 <= metrics["metrics"]["roc_auc"] <= 1.0
