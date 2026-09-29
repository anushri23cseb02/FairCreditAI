"""
ORM models for FairCredit AI.

One table: prediction_log. Every successful /predict or /predict/batch
call writes one row here — this is the "useful application information"
the project keeps in MySQL. Writing to it is best-effort: if the
database is unreachable, the prediction still succeeds and a warning is
logged, rather than failing the user's request because of a storage
side-effect (see backend/services/prediction_logger.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Key application inputs, kept as their own columns so they're easy to
    # query/aggregate later without parsing JSON. Full input/output kept in
    # the JSON columns below for completeness.
    loan_amount_000s: Mapped[float] = mapped_column(Float, nullable=False)
    applicant_income_000s: Mapped[float] = mapped_column(Float, nullable=False)
    loan_type_name: Mapped[str] = mapped_column(String(100), nullable=False)
    loan_purpose_name: Mapped[str] = mapped_column(String(100), nullable=False)

    predicted_label: Mapped[str] = mapped_column(String(20), nullable=False)
    denial_probability: Mapped[float] = mapped_column(Float, nullable=False)

    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class CreditCardPredictionLog(Base):
    """
    Audit trail for Track A (credit-card default) predictions -- kept as
    a separate table from PredictionLog (Track B / HMDA) since the two
    tracks have different schemas and populations; see
    backend/ml/credit_card/feature_config.py for why they are not merged.
    """
    __tablename__ = "credit_card_prediction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)

    limit_bal: Mapped[float] = mapped_column(Float, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    pay_0: Mapped[int] = mapped_column(Integer, nullable=False)

    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    default_probability: Mapped[float] = mapped_column(Float, nullable=False)

    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
