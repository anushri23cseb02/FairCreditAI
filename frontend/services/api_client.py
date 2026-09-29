"""
Thin HTTP client the Streamlit frontend uses to talk to the FastAPI
backend. Streamlit never touches MySQL or business logic directly --
everything goes through this client and the REST API.
"""
from __future__ import annotations

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

DEFAULT_TIMEOUT = 10  # seconds


def get_health() -> dict:
    """Calls GET /health on the backend. Never raises -- returns a dict
    describing the failure instead, so the UI can render a clear status
    message rather than crashing."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": f"Cannot reach backend at {BACKEND_URL}. Is the backend service running?",
        }
    except requests.exceptions.Timeout:
        return {"ok": False, "error": f"Backend at {BACKEND_URL} timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": f"Backend returned an error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Unexpected error contacting backend: {exc}"}


def get_health_detailed() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/health/detailed", timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _get(path: str) -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": f"Cannot reach backend at {BACKEND_URL}."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": f"Backend returned an error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Unexpected error contacting backend: {exc}"}


def _post(path: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 422:
            return {"ok": False, "error": "Validation error", "detail": resp.json()}
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": f"Cannot reach backend at {BACKEND_URL}."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": f"Backend returned an error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Unexpected error contacting backend: {exc}"}


def _post_no_body(path: str, timeout: int = 60) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": f"Cannot reach backend at {BACKEND_URL}."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": f"Backend returned an error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Unexpected error contacting backend: {exc}"}


def predict(application: dict) -> dict:
    return _post("/predict", application)


def predict_batch(applications: list) -> dict:
    return _post("/predict/batch", {"applications": applications})


def get_model_info() -> dict:
    return _get("/model/info")


def get_metrics() -> dict:
    return _get("/metrics")


def get_fairness_report() -> dict:
    return _get("/fairness")


def get_dataset_info() -> dict:
    return _get("/dataset/info")


def get_recent_predictions(limit: int = 10) -> dict:
    return _get(f"/predictions?limit={limit}")


def get_prediction_stats() -> dict:
    return _get("/predictions/stats")


def get_feature_importance() -> dict:
    return _get("/model/feature-importance")


def retrain_model() -> dict:
    return _post_no_body("/train")


# --- Track A: Credit Card (UCI Default of Credit Card Clients) ---------

def cc_predict(application: dict) -> dict:
    return _post("/credit-card/predict", application)


def cc_predict_batch(applications: list) -> dict:
    return _post("/credit-card/predict/batch", {"applications": applications})


def cc_model_info() -> dict:
    return _get("/credit-card/model/info")


def cc_model_comparison() -> dict:
    return _get("/credit-card/model/comparison")


def cc_feature_importance(model_name: str | None = None) -> dict:
    path = "/credit-card/model/feature-importance"
    if model_name:
        path += f"?model_name={model_name}"
    return _get(path)


def cc_dataset_info() -> dict:
    return _get("/credit-card/dataset/info")


def cc_fairness() -> dict:
    return _get("/credit-card/fairness")


def cc_fairness_mitigation() -> dict:
    return _get("/credit-card/fairness/mitigation")


def cc_retrain() -> dict:
    return _post_no_body("/credit-card/train", timeout=120)
