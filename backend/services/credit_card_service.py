from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from backend.ml.credit_card.explain import explain_single, global_feature_importance
from backend.ml.credit_card.feature_config import ALL_FEATURES

logger = logging.getLogger(__name__)


class CreditCardModelNotLoadedError(RuntimeError):
    pass


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "LOW"
    if probability < 0.60:
        return "MODERATE"
    return "HIGH"


class CreditCardModelService:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.registry_path = self.project_root / "models" / "registry" / "registry.json"
        self.metadata_path = self.project_root / "models" / "metadata" / "credit_card_model_comparison.json"

        self.models: dict = {}
        self.selected_model_name: Optional[str] = None
        self.metadata: dict = {}
        self.load_error: Optional[str] = None
        self._background_sample: Optional[pd.DataFrame] = None
        self._load()

    def _load(self) -> None:
        try:
            with open(self.registry_path) as f:
                registry = json.load(f)
            cc = registry["credit_card"]
            self.selected_model_name = cc["active_model"]
            for name, rel_path in cc["artifact_paths"].items():
                self.models[name] = joblib.load(self.project_root / rel_path)
            with open(cc["metadata_path"] if Path(cc["metadata_path"]).is_absolute() else self.project_root / cc["metadata_path"]) as f:
                self.metadata = json.load(f)

            test_path = self.project_root / "data" / "processed" / "credit_card_test.csv"
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                self._background_sample = test_df[ALL_FEATURES].sample(min(50, len(test_df)), random_state=42)

            logger.info("Loaded credit-card models: %s (selected: %s)", list(self.models.keys()), self.selected_model_name)
        except (FileNotFoundError, KeyError) as exc:
            self.load_error = f"Credit-card models not trained yet. Run scripts/prepare_credit_card_data.py and scripts/train_credit_card_models.py. ({exc})"
            logger.warning(self.load_error)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"Failed to load credit-card models: {exc}"
            logger.exception(self.load_error)

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0

    def _resolve_model_name(self, requested: Optional[str]) -> str:
        return requested or self.selected_model_name

    def predict(self, application: dict) -> dict:
        if not self.is_ready:
            raise CreditCardModelNotLoadedError(self.load_error or "Credit-card models not loaded.")
        model_name = self._resolve_model_name(application.get("model_name"))
        if model_name not in self.models:
            raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(self.models.keys())}")
        model = self.models[model_name]
        row = {col: application.get(col, 0) for col in ALL_FEATURES}
        row_df = pd.DataFrame([row])[ALL_FEATURES]

        explanation = explain_single(model_name, model, row_df)
        probability = explanation["default_probability"]
        return {
            "risk_level": _risk_level(probability),
            "default_probability": probability,
            "model_used": model_name,
            "dataset": "credit_card",
            "explanation_method": explanation["explanation_method"],
            "explanation_available": explanation["available"],
            "top_contributing_features": explanation.get("top_contributing_features", []),
        }

    def predict_batch(self, applications: list[dict]) -> list[dict]:
        return [self.predict(app) for app in applications]

    def get_model_info(self) -> dict:
        if not self.is_ready:
            return {"ready": False, "error": self.load_error}
        return {
            "ready": True,
            "dataset": "credit_card",
            "models_compared": self.metadata.get("models_compared"),
            "selected_model": self.selected_model_name,
            "selection_rationale": self.metadata.get("selection_rationale"),
            "selection_weights": self.metadata.get("selection_weights"),
            "selection_scores": self.metadata.get("selection_scores"),
            "trained_at_utc": self.metadata.get("trained_at_utc"),
            "training_rows": self.metadata.get("training_rows"),
            "test_rows": self.metadata.get("test_rows"),
            "feature_columns": self.metadata.get("feature_columns"),
        }

    def get_model_comparison(self) -> dict:
        if not self.is_ready:
            return {"ready": False, "error": self.load_error}
        return {"ready": True, "results": self.metadata.get("results")}

    def get_feature_importance(self, model_name: Optional[str] = None) -> dict:
        if not self.is_ready:
            return {"ready": False, "error": self.load_error}
        name = self._resolve_model_name(model_name)
        if name not in self.models:
            return {"ready": False, "error": f"Unknown model_name '{name}'."}
        result = global_feature_importance(name, self.models[name], self._background_sample)
        result["ready"] = True
        result["model_used"] = name
        return result


@lru_cache
def get_credit_card_model_service() -> CreditCardModelService:
    return CreditCardModelService()
