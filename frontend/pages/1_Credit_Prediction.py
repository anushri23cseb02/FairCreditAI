"""
Credit Prediction page.

Applicant input -> POST /predict -> prediction, probability, and the
top contributing features, clearly labelled as non-causal.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Credit Prediction - FairCredit AI", page_icon="🏦", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("Credit Prediction", "Enter applicant and loan details to get a model prediction.")

st.info(
    "This tool predicts what an AI model estimates, based on historical WA-2016 "
    "HMDA data. It is a demonstration model for a final-year project, not a "
    "real lending decision, and must not be used to make one.",
    icon="ℹ️",
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan details")
        loan_amount_000s = st.number_input("Loan amount ($000s)", min_value=1.0, value=250.0, step=1.0)
        applicant_income_000s = st.number_input("Applicant annual income ($000s)", min_value=1.0, value=80.0, step=1.0)
        loan_type_name = st.selectbox(
            "Loan type", ["Conventional", "FHA-insured", "VA-guaranteed", "FSA/RHS-guaranteed"]
        )
        loan_purpose_name = st.selectbox(
            "Loan purpose", ["Home purchase", "Refinancing", "Home improvement"]
        )
        lien_status_name = st.selectbox(
            "Lien status",
            ["Secured by a first lien", "Secured by a subordinate lien", "Not secured by a lien", "Not applicable"],
        )
        preapproval_name = st.selectbox(
            "Preapproval", ["Not applicable", "Preapproval was not requested", "Preapproval was requested"]
        )
        has_co_applicant = st.checkbox("Application has a co-applicant")

    with col2:
        st.markdown("#### Property & tract details")
        property_type_name = st.selectbox(
            "Property type",
            [
                "One-to-four family dwelling (other than manufactured housing)",
                "Manufactured housing",
                "Multifamily dwelling",
            ],
        )
        owner_occupancy_name = st.selectbox(
            "Owner occupancy",
            ["Owner-occupied as a principal dwelling", "Not owner-occupied as a principal dwelling", "Not applicable"],
        )
        tract_to_msamd_income = st.number_input("Tract income as % of area median", min_value=0.0, value=95.0)
        population = st.number_input("Census tract population", min_value=0.0, value=5000.0)
        minority_population = st.number_input("Tract minority population (%)", min_value=0.0, max_value=100.0, value=20.0)
        number_of_owner_occupied_units = st.number_input("Owner-occupied units in tract", min_value=0.0, value=1500.0)
        number_of_1_to_4_family_units = st.number_input("1-4 family units in tract", min_value=0.0, value=1800.0)
        hud_median_family_income = st.number_input("HUD median family income ($)", min_value=0.0, value=70000.0)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    payload = {
        "loan_amount_000s": loan_amount_000s,
        "applicant_income_000s": applicant_income_000s,
        "tract_to_msamd_income": tract_to_msamd_income,
        "population": population,
        "minority_population": minority_population,
        "number_of_owner_occupied_units": number_of_owner_occupied_units,
        "number_of_1_to_4_family_units": number_of_1_to_4_family_units,
        "hud_median_family_income": hud_median_family_income,
        "has_co_applicant": has_co_applicant,
        "loan_type_name": loan_type_name,
        "loan_purpose_name": loan_purpose_name,
        "property_type_name": property_type_name,
        "owner_occupancy_name": owner_occupancy_name,
        "lien_status_name": lien_status_name,
        "preapproval_name": preapproval_name,
    }

    with st.spinner("Scoring application..."):
        result = api_client.predict(payload)

    st.divider()

    if not result["ok"]:
        status_badge(False, "", f"Prediction failed — {result['error']}")
        if "detail" in result:
            st.json(result["detail"])
    else:
        data = result["data"]

        headline = (
            "Credit Application Likely to Be Denied"
            if data["predicted_label"] == "denied"
            else "Credit Application Likely to Be Approved"
        )
        st.subheader(f"Prediction Result: {headline}")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            tone = "peach" if data["predicted_label"] == "denied" else "green"
            metric_card("Result", "DENIED" if data["predicted_label"] == "denied" else "APPROVED", tone=tone)
        with col_b:
            metric_card("Denial probability", f"{data['denial_probability'] * 100:.1f}%", tone="blue")
        with col_c:
            metric_card("Approval probability", f"{(1 - data['denial_probability']) * 100:.1f}%", tone="green")
        with col_d:
            metric_card("Model", data["model_version"], tone="gray")

        from datetime import datetime, timezone
        st.caption(f"Predicted at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} · saved to prediction history")

        st.markdown("#### Why this prediction?")
        st.caption(data["explanation_note"])
        for feat in data["top_contributing_features"]:
            arrow = "🔺" if feat["direction"] == "increased" else "🔻"
            st.write(
                f"{arrow} **{feat['feature']}** {feat['direction']} the denial score "
                f"(contribution: {feat['contribution_to_denial_logit']:+.3f})"
            )
