from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from backend.core.config import get_settings
from backend.ml.explain import explain_single, global_feature_importance
from backend.ml.feature_config import ALL_CATEGORICAL_FEATURES, ALL_NUMERIC_FEATURES

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no trained artifact exists."""


class ModelService:
    """
    Loads the active model artifact + its metadata + the fairness report
    ONCE, at process startup (via the lru_cache'd get_model_service()
    below). GET/POST endpoints never retrain or reload from disk per
    request.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.registry_path = self.project_root / "models" / "registry" / "registry.json"
        self.fairness_report_path = self.project_root / "data" / "reports" / "fairness_report.json"

        self.model = None
        self.metadata: dict = {}
        self.active_version: Optional[str] = None
        self.load_error: Optional[str] = None

        self._load()

    def _load(self) -> None:
        try:
            with open(self.registry_path) as f:
                registry = json.load(f)
            self.active_version = registry["active_version"]
            artifact_rel_path = registry["versions"][self.active_version]["artifact_path"]
            metadata_rel_path = registry["versions"][self.active_version]["metadata_path"]

            self.model = joblib.load(self.project_root / artifact_rel_path)
            with open(self.project_root / metadata_rel_path) as f:
                self.metadata = json.load(f)

            logger.info("Loaded model version %s", self.active_version)
        except FileNotFoundError as exc:
            self.load_error = (
                "No trained model artifact found. Run scripts/prepare_data.py "
                f"and scripts/train_model.py first. ({exc})"
            )
            logger.warning(self.load_error)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"Failed to load model: {exc}"
            logger.exception(self.load_error)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _to_frame(self, application: dict) -> pd.DataFrame:
        row = {col: application.get(col) for col in FEATURE_COLUMNS}
        # has_co_applicant / loan_to_income_ratio are engineered features;
        # the API accepts a boolean has_co_applicant and raw loan/income,
        # and derives the ratio the same way scripts/prepare_data.py does.
        row["has_co_applicant"] = int(bool(application.get("has_co_applicant", False)))
        row["loan_to_income_ratio"] = (
            application["loan_amount_000s"] / application["applicant_income_000s"]
        )
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    def predict(self, application: dict) -> dict:
        if not self.is_ready:
            raise ModelNotLoadedError(self.load_error or "Model not loaded.")
        row_df = self._to_frame(application)
        result = explain_single(self.model, row_df)
        result["model_version"] = self.active_version
        return result

    def predict_batch(self, applications: list[dict]) -> list[dict]:
        return [self.predict(app) for app in applications]

    def get_model_info(self) -> dict:
        if not self.is_ready:
            return {"ready": False, "error": self.load_error}
        return {
            "ready": True,
            "version": self.active_version,
            "model_type": self.metadata.get("model_type"),
            "trained_at_utc": self.metadata.get("trained_at_utc"),
            "training_rows": self.metadata.get("training_rows"),
            "test_rows": self.metadata.get("test_rows"),
            "feature_columns": self.metadata.get("feature_columns_raw"),
            "hyperparameters": self.metadata.get("hyperparameters"),
        }

    def get_metrics(self) -> dict:
        if not self.is_ready:
            return {"ready": False, "error": self.load_error}
        return {
            "ready": True,
            "version": self.active_version,
            "metrics": self.metadata.get("held_out_test_metrics_final_model"),
            "note": self.metadata.get("notes"),
        }

    def get_global_feature_importance(self) -> dict:
        if not self.is_ready:
            return {"ready": False, "error": self.load_error}
        return {"ready": True, "version": self.active_version, "features": global_feature_importance(self.model)}

    def get_fairness_report(self) -> dict:
        if not self.fairness_report_path.exists():
            return {
                "ready": False,
                "error": (
                    "No fairness report found. Run scripts/evaluate_fairness.py "
                    "after training the model."
                ),
            }
        with open(self.fairness_report_path) as f:
            report = json.load(f)
        report["ready"] = True
        return report


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()
