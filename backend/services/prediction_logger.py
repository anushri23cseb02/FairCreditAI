from __future__ import annotations

import logging

from backend.database.connection import session_scope
from backend.database.models import CreditCardPredictionLog, PredictionLog

logger = logging.getLogger(__name__)


def log_prediction(application: dict, result: dict) -> None:
    """
    Best-effort write of one prediction to MySQL. Deliberately swallows
    any database error (down, still starting, schema missing) — logging
    a prediction is a nice-to-have, not something that should turn a
    successful prediction into a 500 for the applicant.
    """
    try:
        with session_scope() as db:
            db.add(
                PredictionLog(
                    model_version=result["model_version"],
                    loan_amount_000s=application["loan_amount_000s"],
                    applicant_income_000s=application["applicant_income_000s"],
                    loan_type_name=application["loan_type_name"],
                    loan_purpose_name=application["loan_purpose_name"],
                    predicted_label=result["predicted_label"],
                    denial_probability=result["denial_probability"],
                    request_payload=application,
                    response_payload=result,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write prediction to database (continuing without it): %s", exc)


def log_credit_card_prediction(application: dict, result: dict) -> None:
    """Same best-effort pattern as log_prediction, for the Track A audit trail."""
    try:
        with session_scope() as db:
            db.add(
                CreditCardPredictionLog(
                    model_used=result["model_used"],
                    limit_bal=application["LIMIT_BAL"],
                    age=application["AGE"],
                    pay_0=application["PAY_0"],
                    risk_level=result["risk_level"],
                    default_probability=result["default_probability"],
                    request_payload=application,
                    response_payload=result,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write credit-card prediction to database (continuing without it): %s", exc)
