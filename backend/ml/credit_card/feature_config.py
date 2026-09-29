"""
Single source of truth for the Track A (UCI "Default of Credit Card
Clients") modelling table — the credit-card-default counterpart to
backend/ml/feature_config.py (HMDA).

============================================================
TARGET
============================================================
Raw column: "default payment next month" (already binary: 1 = defaulted
the following month, 0 = did not). No derivation needed — unlike HMDA,
this dataset ships with its target already defined by the source
institution, so nothing is inferred here.

============================================================
PROTECTED ATTRIBUTE
============================================================
SEX is the only demographic attribute this dataset actually contains
(coded 1=Male, 2=Female in the source documentation). There is no race
or ethnicity column here — the project does not invent one for this
dataset. SEX is used ONLY for the fairness audit; it is excluded from
the model's input features.

AGE is present and is a plausible fairness dimension too (age is a
protected class under the U.S. Equal Credit Opportunity Act), but it is
continuous, so a fairness comparison needs an explicit banding decision
rather than a silent one. This project bands it as:
    "Under 35"  vs  "35 and over"
purely to have two comparably-sized groups for the audit — this is a
methodological choice, stated here rather than hidden, not a claim that
35 is a scientifically special cutoff. Unlike SEX, AGE IS still used as
a model feature (age is a broadly-used, legitimate underwriting
variable), so the age fairness check is a proxy/disparate-impact check
on a feature the model does see, not a same-treatment guarantee.

============================================================
FEATURES
============================================================
"""

TARGET_RAW_COLUMN = "default payment next month"
TARGET_COLUMN = "default"

PROTECTED_ATTRIBUTES = {
    "sex": {
        "column": "SEX",
        # Mapped from the raw {1, 2} codes to labels in
        # scripts/prepare_credit_card_data.py so the fairness report reads
        # in plain English rather than raw integers.
        "privileged": "Male",
        "comparison": "Female",
        "unknown_values": [],
    },
}

# AGE fairness is handled as a secondary, clearly-labelled proxy check
# (see docstring above) rather than a first-class PROTECTED_ATTRIBUTES
# entry, since AGE remains a model feature.
AGE_BAND_COLUMN = "age_band"
AGE_BAND_PRIVILEGED = "35 and over"
AGE_BAND_COMPARISON = "Under 35"
AGE_BAND_CUTOFF = 35

NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# EDUCATION and MARRIAGE are numerically coded categories in the source
# file, not continuous quantities, so they're treated as categorical
# (one-hot encoded) rather than numeric.
CATEGORICAL_FEATURES = ["EDUCATION", "MARRIAGE"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The source documentation only defines EDUCATION in {1,2,3,4} and
# MARRIAGE in {1,2,3}. Codes outside that range exist in the real data
# (see data/reports/data_summary_credit_card.json for exact counts) and
# are bucketed into "Other/Unspecified" rather than guessed at.
EDUCATION_LABELS = {1: "Graduate School", 2: "University", 3: "High School"}
EDUCATION_OTHER_LABEL = "Other/Unspecified"
MARRIAGE_LABELS = {1: "Married", 2: "Single"}
MARRIAGE_OTHER_LABEL = "Other/Unspecified"

RANDOM_STATE = 42
