"""
Phase 1 smoke test.

Verifies the FastAPI app boots and the /health endpoint responds with the
exact contract required by the spec, without requiring a live MySQL
connection (TestClient exercises the app in-process).
"""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint_reports_service_info():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["docs"] == "/docs"


def test_detailed_health_endpoint_returns_structure_even_without_db():
    """
    This must never crash even if MySQL is unreachable in the test
    environment -- it should report 'degraded' / 'unreachable' instead.
    """
    response = client.get("/health/detailed")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert body["database"] in {"connected", "unreachable"}
