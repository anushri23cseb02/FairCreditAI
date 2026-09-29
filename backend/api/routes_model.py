from __future__ import annotations

from fastapi import APIRouter

from backend.services.model_service import get_model_service

router = APIRouter(tags=["model"])


@router.get("/model/info")
def model_info() -> dict:
    return get_model_service().get_model_info()


@router.get("/model/feature-importance")
def model_feature_importance() -> dict:
    return get_model_service().get_global_feature_importance()


@router.get("/metrics")
def metrics() -> dict:
    return get_model_service().get_metrics()


@router.get("/fairness")
def fairness() -> dict:
    return get_model_service().get_fairness_report()
