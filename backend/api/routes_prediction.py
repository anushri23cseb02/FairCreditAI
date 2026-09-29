from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    LoanApplication,
    PredictionResponse,
)
from backend.services.model_service import ModelNotLoadedError, get_model_service
from backend.services.prediction_logger import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    service = get_model_service()
    try:
        result = service.predict(application.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not score this application: {exc}") from exc
    log_prediction(application.model_dump(), result)
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    service = get_model_service()
    try:
        applications = [app.model_dump() for app in request.applications]
        results = service.predict_batch(applications)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not score batch: {exc}") from exc
    for app, result in zip(applications, results):
        log_prediction(app, result)
    return BatchPredictionResponse(results=[PredictionResponse(**r) for r in results], count=len(results))
