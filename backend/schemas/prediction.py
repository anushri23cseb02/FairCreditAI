from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

LOAN_TYPES = ["Conventional", "FHA-insured", "VA-guaranteed", "FSA/RHS-guaranteed"]
LOAN_PURPOSES = ["Home purchase", "Refinancing", "Home improvement"]
PROPERTY_TYPES = [
    "One-to-four family dwelling (other than manufactured housing)",
    "Manufactured housing",
    "Multifamily dwelling",
]
OWNER_OCCUPANCY = [
    "Owner-occupied as a principal dwelling",
    "Not owner-occupied as a principal dwelling",
    "Not applicable",
]
LIEN_STATUS = [
    "Secured by a first lien",
    "Secured by a subordinate lien",
    "Not secured by a lien",
    "Not applicable",
]
PREAPPROVAL = ["Not applicable", "Preapproval was not requested", "Preapproval was requested"]


class LoanApplication(BaseModel):
    """
    Raw application fields the model was trained on. Applicant race, sex,
    and ethnicity are intentionally NOT accepted here — the model never
    uses them as features (see backend/ml/feature_config.py); they only
    ever appear in the separate, aggregate fairness report.
    """

    loan_amount_000s: float = Field(..., gt=0, description="Loan amount, in thousands of USD.")
    applicant_income_000s: float = Field(..., gt=0, description="Applicant annual income, in thousands of USD.")
    tract_to_msamd_income: Optional[float] = Field(None, description="Census tract income as % of MSA/MD median.")
    population: Optional[float] = Field(None, ge=0, description="Census tract population.")
    minority_population: Optional[float] = Field(None, ge=0, le=100, description="% minority population in tract.")
    number_of_owner_occupied_units: Optional[float] = Field(None, ge=0)
    number_of_1_to_4_family_units: Optional[float] = Field(None, ge=0)
    hud_median_family_income: Optional[float] = Field(None, ge=0)
    has_co_applicant: bool = Field(False, description="Whether the application includes a co-applicant.")

    loan_type_name: str = Field(..., description=f"One of {LOAN_TYPES}")
    loan_purpose_name: str = Field(..., description=f"One of {LOAN_PURPOSES}")
    property_type_name: str = Field(..., description=f"One of {PROPERTY_TYPES}")
    owner_occupancy_name: str = Field(..., description=f"One of {OWNER_OCCUPANCY}")
    lien_status_name: str = Field(..., description=f"One of {LIEN_STATUS}")
    preapproval_name: str = Field(..., description=f"One of {PREAPPROVAL}")

    @field_validator("loan_type_name")
    @classmethod
    def _check_loan_type(cls, v: str) -> str:
        if v not in LOAN_TYPES:
            raise ValueError(f"loan_type_name must be one of {LOAN_TYPES}")
        return v

    @field_validator("loan_purpose_name")
    @classmethod
    def _check_loan_purpose(cls, v: str) -> str:
        if v not in LOAN_PURPOSES:
            raise ValueError(f"loan_purpose_name must be one of {LOAN_PURPOSES}")
        return v

    @field_validator("property_type_name")
    @classmethod
    def _check_property_type(cls, v: str) -> str:
        if v not in PROPERTY_TYPES:
            raise ValueError(f"property_type_name must be one of {PROPERTY_TYPES}")
        return v

    @field_validator("owner_occupancy_name")
    @classmethod
    def _check_owner_occupancy(cls, v: str) -> str:
        if v not in OWNER_OCCUPANCY:
            raise ValueError(f"owner_occupancy_name must be one of {OWNER_OCCUPANCY}")
        return v

    @field_validator("lien_status_name")
    @classmethod
    def _check_lien_status(cls, v: str) -> str:
        if v not in LIEN_STATUS:
            raise ValueError(f"lien_status_name must be one of {LIEN_STATUS}")
        return v

    @field_validator("preapproval_name")
    @classmethod
    def _check_preapproval(cls, v: str) -> str:
        if v not in PREAPPROVAL:
            raise ValueError(f"preapproval_name must be one of {PREAPPROVAL}")
        return v


class ContributingFeature(BaseModel):
    feature: str
    direction: Literal["increased", "decreased"]
    contribution_to_denial_logit: float


class PredictionResponse(BaseModel):
    predicted_label: Literal["denied", "not_denied"]
    denial_probability: float
    model_version: str
    top_contributing_features: List[ContributingFeature]
    explanation_note: str


class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplication] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    count: int
