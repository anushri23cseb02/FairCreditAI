from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.ml.trainer import TrainingDataMissingError, train_and_persist
from backend.services.model_service import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["training"])


@router.post("/train")
def train() -> dict:
    """
    Retrains the model from the already-prepared processed data splits
    (data/processed/train.csv, val.csv, test.csv) and replaces the active
    model version. Does NOT re-read the raw dataset or re-run
    scripts/prepare_data.py -- that is a separate, explicit step, since it
    means scanning the full raw CSV. Safe to call repeatedly: it always
    reproduces the same deterministic split/training given the same
    processed data (fixed random_state).
    """
    try:
        metadata = train_and_persist(version="v1")
    except TrainingDataMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    # Force the in-process model service to pick up the freshly trained
    # artifact immediately, so the very next /predict call uses it
    # without needing a container restart.
    get_model_service.cache_clear()
    get_model_service()

    return {
        "status": "trained",
        "version": metadata["version"],
        "trained_at_utc": metadata["trained_at_utc"],
        "test_metrics": metadata["held_out_test_metrics_final_model"],
    }
