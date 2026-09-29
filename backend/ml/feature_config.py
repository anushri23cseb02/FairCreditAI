"""
Single source of truth for how FairCredit AI turns the raw HMDA
Washington State 2016 CSV into a modelling table.

Every decision here is deliberate and documented so the pipeline is
reproducible and auditable (Phase 2 / Phase 5 requirements). Nothing in
this file is fabricated: every choice is justified against what is
actually present in the raw dataset (see data/reports/data_summary.json
after running scripts/prepare_data.py for the real value counts that
back these decisions).

============================================================
TARGET DEFINITION — "denied"
============================================================
The raw column `action_taken_name` has 8 possible values. Only two of
them represent a completed, comparable lending decision by the
institution:

    "Loan originated"                              -> denied = 0
    "Application denied by financial institution"   -> denied = 1

The other six values are EXCLUDED from the modelling table, not
relabelled, because they do not represent a comparable adjudicated
outcome:

    "Application withdrawn by applicant"            -> applicant action, not a lender decision
    "File closed for incompleteness"                -> no decision was ever made
    "Loan purchased by the institution"              -> this institution was not the originating decision-maker
    "Application approved but not accepted"          -> approved, so it is not a denial, but including it as a
                                                          "0" would blur "approved" with "originated"; excluded for
                                                          a clean binary contrast
    "Preapproval request denied by financial institution" -> a different product/process (preapproval, not final loan)
    "Preapproval request approved but not accepted"  -> same reason as above

This is the standard "denial model" framing used in HMDA fair-lending
analysis: compare completed originations against completed denials,
and leave every ambiguous or non-adjudicated status out rather than
guess at a label for it.

============================================================
PROTECTED ATTRIBUTES (never used as model features)
============================================================
Held out and used ONLY for the fairness evaluation in
backend/ml/fairness.py — never fed into the classifier, to avoid
disparate treatment (deciding using race/sex/ethnicity directly).

    applicant_sex_name        privileged="Male"   comparison="Female"
    applicant_race_name_1     privileged="White"  comparison="Black or African American"
    applicant_ethnicity_name  privileged="Not Hispanic or Latino"  comparison="Hispanic or Latino"

Rows where the attribute is "Not applicable" or
"Information not provided by applicant in mail, Internet, or telephone
application" are kept in the modelling table (they still have a valid
target and valid features) but are EXCLUDED from that specific
fairness slice, since "unknown" is not a group you can compare fairly.

These are the only protected attributes used because they are the only
demographic fields actually collected in HMDA for the applicant. No
attribute is invented.

============================================================
MODEL FEATURES
============================================================
"""

# --- Raw target column and the two comparable outcomes ------------------
TARGET_RAW_COLUMN = "action_taken_name"
TARGET_COLUMN = "denied"

ACTION_TO_LABEL = {
    "Loan originated": 0,
    "Application denied by financial institution": 1,
}

# --- Protected attributes (analysis-only, never modelling features) -----
PROTECTED_ATTRIBUTES = {
    "sex": {
        "column": "applicant_sex_name",
        "privileged": "Male",
        "comparison": "Female",
        "unknown_values": [
            "Not applicable",
            "Information not provided by applicant in mail, Internet, or telephone application",
        ],
    },
    "race": {
        "column": "applicant_race_name_1",
        "privileged": "White",
        "comparison": "Black or African American",
        "unknown_values": [
            "Not applicable",
            "Information not provided by applicant in mail, Internet, or telephone application",
        ],
    },
    "ethnicity": {
        "column": "applicant_ethnicity_name",
        "privileged": "Not Hispanic or Latino",
        "comparison": "Hispanic or Latino",
        "unknown_values": [
            "Not applicable",
            "Information not provided by applicant in mail, Internet, or telephone application",
        ],
    },
}

# --- Numeric model features ---------------------------------------------
NUMERIC_FEATURES = [
    "loan_amount_000s",
    "applicant_income_000s",
    "tract_to_msamd_income",
    "population",
    "minority_population",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
    "hud_median_family_income",
]

# --- Categorical model features -----------------------------------------
CATEGORICAL_FEATURES = [
    "loan_type_name",
    "loan_purpose_name",
    "property_type_name",
    "owner_occupancy_name",
    "lien_status_name",
    "preapproval_name",
]

# --- Engineered features (created in scripts/prepare_data.py) -----------
# has_co_applicant: derived from co_applicant_sex_name != "No co-applicant".
#   Whether an application has a co-applicant is itself informative for
#   credit risk and is NOT a protected attribute (it doesn't identify the
#   co-applicant's race/sex, just their presence).
# loan_to_income_ratio: loan_amount_000s / applicant_income_000s, a
#   standard, interpretable credit-risk ratio.
ENGINEERED_NUMERIC_FEATURES = ["loan_to_income_ratio"]
ENGINEERED_BINARY_FEATURES = ["has_co_applicant"]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES + ENGINEERED_BINARY_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES

# --- Columns explicitly excluded, and why (leakage / identifiers) -------
EXCLUDED_COLUMNS = {
    "denial_reason_name_1": "Only ever populated when the application was denied — direct target leakage.",
    "denial_reason_name_2": "Same as denial_reason_name_1 — direct target leakage.",
    "denial_reason_name_3": "Same as denial_reason_name_1 — direct target leakage.",
    "rate_spread": "Only populated for a subset of higher-priced loans tied to the lending decision context — leakage risk.",
    "purchaser_type_name": "Post-decision secondary-market information; near-empty for denied applications — leakage.",
    "edit_status_name": "Internal HMDA data-quality edit flag, not a cause of the lending decision.",
    "sequence_number": "Row identifier, not predictive.",
    "respondent_id": "Institution identifier, not an applicant/loan feature; too high-cardinality to safely encode.",
    "census_tract_number": "High-cardinality geographic identifier; tract-level info is already captured via population/income features.",
    "msamd_name": "High-cardinality geography; risk of the model learning place-based proxies instead of loan/applicant risk.",
    "county_name": "Same reasoning as msamd_name.",
    "state_name": "Constant (Washington only) — no signal.",
    "state_abbr": "Constant — no signal.",
    "agency_name": "Regulatory reporting agency, not a property of the applicant or loan.",
    "agency_abbr": "Same as agency_name.",
    "as_of_year": "Constant (2016) — no signal.",
    "application_date_indicator": "Near-constant, no meaningful signal in this extract.",
    "hoepa_status_name": "Extremely rare positive class in this extract; kept out of v1 baseline to avoid a near-constant feature.",
}

RANDOM_STATE = 42
