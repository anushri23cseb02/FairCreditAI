from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.ml.credit_card.fairness import audit_age_band, audit_sex, mitigate_with_threshold_optimizer
from backend.ml.credit_card.feature_config import ALL_FEATURES, AGE_BAND_COLUMN, TARGET_COLUMN
from backend.ml.credit_card.trainer import TrainingDataMissingError, train_and_compare
from backend.schemas.credit_card import (
    CreditCardApplication, CreditCardBatchRequest, CreditCardBatchResponse, CreditRiskAssessment,
)
from backend.services.credit_card_service import CreditCardModelNotLoadedError, get_credit_card_model_service
from backend.services.prediction_logger import log_credit_card_prediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/credit-card", tags=["credit-card"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SUMMARY_PATH = PROJECT_ROOT / "data" / "reports" / "data_summary_credit_card.json"


@router.post("/predict", response_model=CreditRiskAssessment)
def predict(application: CreditCardApplication) -> CreditRiskAssessment:
    service = get_credit_card_model_service()
    try:
        result = service.predict(application.model_dump())
    except CreditCardModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Credit-card prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not score this application: {exc}") from exc
    log_credit_card_prediction(application.model_dump(), result)
    return CreditRiskAssessment(**result)


@router.post("/predict/batch", response_model=CreditCardBatchResponse)
def predict_batch(request: CreditCardBatchRequest) -> CreditCardBatchResponse:
    service = get_credit_card_model_service()
    try:
        applications = [app.model_dump() for app in request.applications]
        results = service.predict_batch(applications)
    except CreditCardModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Credit-card batch prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not score batch: {exc}") from exc
    for app, result in zip(applications, results):
        log_credit_card_prediction(app, result)
    return CreditCardBatchResponse(results=[CreditRiskAssessment(**r) for r in results], count=len(results))


@router.get("/model/info")
def model_info() -> dict:
    return get_credit_card_model_service().get_model_info()


@router.get("/model/comparison")
def model_comparison() -> dict:
    return get_credit_card_model_service().get_model_comparison()


@router.get("/model/feature-importance")
def feature_importance(model_name: str | None = None) -> dict:
    return get_credit_card_model_service().get_feature_importance(model_name)


@router.get("/dataset/info")
def dataset_info() -> dict:
    if not DATA_SUMMARY_PATH.exists():
        return {"ready": False, "error": "No data summary found. Run scripts/prepare_credit_card_data.py."}
    with open(DATA_SUMMARY_PATH) as f:
        summary = json.load(f)
    summary["ready"] = True
    return summary


@router.get("/fairness")
def fairness() -> dict:
    """Fairness audit (before any mitigation) on the selected model's test-set predictions."""
    service = get_credit_card_model_service()
    if not service.is_ready:
        return {"ready": False, "error": service.load_error}
    test_path = PROCESSED_DIR / "credit_card_test.csv"
    if not test_path.exists():
        return {"ready": False, "error": "No processed test set found."}

    test_df = pd.read_csv(test_path)
    model = service.models[service.selected_model_name]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COLUMN]
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = pd.Series((proba >= 0.5).astype(int), index=test_df.index)

    return {
        "ready": True,
        "model_used": service.selected_model_name,
        "test_rows": len(test_df),
        "sex": audit_sex(y_test, y_pred, test_df["SEX"]),
        "age_band": audit_age_band(y_test, y_pred, test_df[AGE_BAND_COLUMN]),
        "disclaimer": (
            "Fairness metrics measure differences in model outcomes between groups. "
            "They do not by themselves establish whether discrimination occurred. "
            "Results depend on the dataset, population, sample size, model, threshold "
            "and fairness definition used."
        ),
    }


@router.get("/fairness/mitigation")
def fairness_mitigation() -> dict:
    """Real before/after ThresholdOptimizer mitigation on SEX, fit on train, evaluated on held-out test."""
    service = get_credit_card_model_service()
    if not service.is_ready:
        return {"ready": False, "error": service.load_error}
    train_path = PROCESSED_DIR / "credit_card_train.csv"
    test_path = PROCESSED_DIR / "credit_card_test.csv"
    if not train_path.exists() or not test_path.exists():
        return {"ready": False, "error": "Processed train/test data not found."}

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model = service.models[service.selected_model_name]

    try:
        result = mitigate_with_threshold_optimizer(
            model,
            train_df[ALL_FEATURES], train_df[TARGET_COLUMN], train_df["SEX"],
            test_df[ALL_FEATURES], test_df[TARGET_COLUMN], test_df["SEX"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mitigation failed")
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {exc}") from exc

    result["ready"] = True
    result["model_used"] = service.selected_model_name
    return result


@router.get("/predictions")
def credit_card_predictions(limit: int = 20) -> dict:
    """Prediction history for Track A, stored in the credit_card_prediction_log MySQL table."""
    from sqlalchemy import select
    from backend.database.connection import session_scope
    from backend.database.models import CreditCardPredictionLog

    limit = max(1, min(limit, 200))
    try:
        with session_scope() as db:
            rows = db.execute(
                select(CreditCardPredictionLog).order_by(CreditCardPredictionLog.id.desc()).limit(limit)
            ).scalars().all()
            return {
                "ready": True,
                "count": len(rows),
                "predictions": [
                    {
                        "id": r.id,
                        "created_at": r.created_at.isoformat(),
                        "model_used": r.model_used,
                        "limit_bal": r.limit_bal,
                        "age": r.age,
                        "pay_0": r.pay_0,
                        "risk_level": r.risk_level,
                        "default_probability": r.default_probability,
                    }
                    for r in rows
                ],
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read credit-card prediction history: %s", exc)
        return {"ready": False, "count": 0, "predictions": [], "error": str(exc)}


@router.post("/train")
def train() -> dict:
    try:
        metadata = train_and_compare()
    except TrainingDataMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Credit-card training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    get_credit_card_model_service.cache_clear()
    get_credit_card_model_service()

    return {
        "status": "trained",
        "selected_model": metadata["selected_model"],
        "selection_rationale": metadata["selection_rationale"],
        "selection_scores": metadata["selection_scores"],
        "trained_at_utc": metadata["trained_at_utc"],
    }
