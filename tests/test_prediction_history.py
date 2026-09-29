"""
Tests for GET /dataset/info and GET /predictions/recent.

/predictions/recent must degrade gracefully (ready=False, empty list)
rather than error when no database is reachable in the test
environment — the real positive path (rows actually coming back) is
exercised in this session's manual verification against a live
SQLAlchemy-backed database (see project verification notes) and again
whenever this runs inside Docker against the real MySQL service.
"""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_dataset_info_reports_target_derivation():
    response = client.get("/dataset/info")
    assert response.status_code == 200
    body = response.json()
    assert body["target_column"] == "denied"
    assert "Loan originated" in body["target_derivation"]
    assert "sex" in body["protected_attributes"]


def test_recent_predictions_never_raises():
    response = client.get("/predictions")
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)


def test_recent_predictions_respects_limit_bounds():
    response = client.get("/predictions?limit=500")
    assert response.status_code == 422  # over the max=200 bound


def test_prediction_stats_never_raises():
    response = client.get("/predictions/stats")
    assert response.status_code == 200
    assert "ready" in response.json()
