from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

VALID_MODELS = ["logistic_regression", "random_forest", "xgboost"]


class CreditCardApplication(BaseModel):
    """
    Raw fields the credit-card-default models were trained on. SEX is
    intentionally NOT accepted here -- it is never used as a model
    feature (see backend/ml/credit_card/feature_config.py), only in the
    separate fairness audit.
    """
    LIMIT_BAL: float = Field(..., gt=0, description="Credit limit (NT dollar).")
    AGE: int = Field(..., ge=18, le=100)
    EDUCATION: str = Field(..., description="Graduate School / University / High School / Other/Unspecified")
    MARRIAGE: str = Field(..., description="Married / Single / Other/Unspecified")
    PAY_0: int = Field(..., ge=-2, le=8, description="Most recent repayment status (-2..8).")
    PAY_2: int = Field(..., ge=-2, le=8)
    PAY_3: int = Field(..., ge=-2, le=8)
    PAY_4: int = Field(..., ge=-2, le=8)
    PAY_5: int = Field(..., ge=-2, le=8)
    PAY_6: int = Field(..., ge=-2, le=8)
    BILL_AMT1: float = 0
    BILL_AMT2: float = 0
    BILL_AMT3: float = 0
    BILL_AMT4: float = 0
    BILL_AMT5: float = 0
    BILL_AMT6: float = 0
    PAY_AMT1: float = Field(0, ge=0)
    PAY_AMT2: float = Field(0, ge=0)
    PAY_AMT3: float = Field(0, ge=0)
    PAY_AMT4: float = Field(0, ge=0)
    PAY_AMT5: float = Field(0, ge=0)
    PAY_AMT6: float = Field(0, ge=0)

    model_name: Optional[str] = Field(
        default=None,
        description="Which of the 3 compared models to use: logistic_regression / random_forest / xgboost. Defaults to the selected model.",
    )

    @field_validator("EDUCATION")
    @classmethod
    def _check_education(cls, v: str) -> str:
        allowed = {"Graduate School", "University", "High School", "Other/Unspecified"}
        if v not in allowed:
            raise ValueError(f"EDUCATION must be one of {sorted(allowed)}")
        return v

    @field_validator("MARRIAGE")
    @classmethod
    def _check_marriage(cls, v: str) -> str:
        allowed = {"Married", "Single", "Other/Unspecified"}
        if v not in allowed:
            raise ValueError(f"MARRIAGE must be one of {sorted(allowed)}")
        return v

    @field_validator("model_name")
    @classmethod
    def _check_model_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_MODELS:
            raise ValueError(f"model_name must be one of {VALID_MODELS}")
        return v


class ContributingFactor(BaseModel):
    feature: str
    direction: Literal["increases risk", "decreases risk"]
    contribution: float


class CreditRiskAssessment(BaseModel):
    risk_level: Literal["LOW", "MODERATE", "HIGH"]
    default_probability: float
    model_used: str
    dataset: Literal["credit_card"] = "credit_card"
    explanation_method: str
    explanation_available: bool
    top_contributing_features: List[ContributingFactor]
    assessment_note: str = (
        "This is an AI-assisted risk assessment, not a lending decision. "
        "Final lending decisions require appropriate human/institutional review."
    )


class CreditCardBatchRequest(BaseModel):
    applications: List[CreditCardApplication] = Field(..., min_length=1, max_length=500)


class CreditCardBatchResponse(BaseModel):
    results: List[CreditRiskAssessment]
    count: int
