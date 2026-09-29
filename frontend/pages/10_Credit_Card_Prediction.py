"""
Credit Card Risk Prediction page (Track A: UCI Default of Credit Card
Clients — a different dataset and target from the HMDA mortgage-denial
pages elsewhere in this app; see the banner below).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Credit Card Risk - FairCreditAI", page_icon="💳", layout="wide")

section_header("Credit Card Risk Prediction", "Track A — UCI Default of Credit Card Clients dataset.")

st.info(
    "This page uses a **different dataset and target** from the mortgage-denial "
    "pages elsewhere in this app: it predicts **next-month credit-card default**, "
    "not mortgage denial. The two are not merged or compared directly.",
    icon="ℹ️",
)

info_result = api_client.cc_model_info()
if not info_result["ok"] or not info_result["data"].get("ready"):
    status_badge(False, "", f"Models not available — {info_result.get('error') or info_result['data'].get('error')}")
    st.stop()

model_choice = st.selectbox(
    "Model to use for this prediction",
    ["Use selected model (recommended)", "logistic_regression", "random_forest", "xgboost"],
    help=f"The selected model is currently: {info_result['data']['selected_model']}. See Model Comparison for why.",
)

with st.form("cc_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Applicant & Credit Information")
        LIMIT_BAL = st.number_input("Credit limit (NT$)", min_value=1000, value=200000, step=1000)
        AGE = st.number_input("Age", min_value=18, max_value=100, value=35)
        EDUCATION = st.selectbox("Education", ["Graduate School", "University", "High School", "Other/Unspecified"])
        MARRIAGE = st.selectbox("Marital status", ["Married", "Single", "Other/Unspecified"])

        st.markdown("#### Repayment Status (last 6 months, most recent first)")
        st.caption("-2/-1 = paid in full/on time, 0 = revolving credit used, 1+ = months delayed")
        PAY_0 = st.slider("Most recent month", -2, 8, 0)
        PAY_2 = st.slider("2 months ago", -2, 8, 0)
        PAY_3 = st.slider("3 months ago", -2, 8, 0)
        PAY_4 = st.slider("4 months ago", -2, 8, 0)
        PAY_5 = st.slider("5 months ago", -2, 8, 0)
        PAY_6 = st.slider("6 months ago", -2, 8, 0)

    with col2:
        st.markdown("#### Bill Statement Amounts (last 6 months, NT$)")
        BILL_AMT1 = st.number_input("Most recent bill", value=20000, step=1000)
        BILL_AMT2 = st.number_input("2 months ago bill", value=19000, step=1000)
        BILL_AMT3 = st.number_input("3 months ago bill", value=18000, step=1000)
        BILL_AMT4 = st.number_input("4 months ago bill", value=17000, step=1000)
        BILL_AMT5 = st.number_input("5 months ago bill", value=16000, step=1000)
        BILL_AMT6 = st.number_input("6 months ago bill", value=15000, step=1000)

        st.markdown("#### Payment Amounts (last 6 months, NT$)")
        PAY_AMT1 = st.number_input("Most recent payment", min_value=0, value=2000, step=500)
        PAY_AMT2 = st.number_input("2 months ago payment", min_value=0, value=2000, step=500)
        PAY_AMT3 = st.number_input("3 months ago payment", min_value=0, value=2000, step=500)
        PAY_AMT4 = st.number_input("4 months ago payment", min_value=0, value=2000, step=500)
        PAY_AMT5 = st.number_input("5 months ago payment", min_value=0, value=2000, step=500)
        PAY_AMT6 = st.number_input("6 months ago payment", min_value=0, value=2000, step=500)

    submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)

if submitted:
    payload = {
        "LIMIT_BAL": LIMIT_BAL, "AGE": AGE, "EDUCATION": EDUCATION, "MARRIAGE": MARRIAGE,
        "PAY_0": PAY_0, "PAY_2": PAY_2, "PAY_3": PAY_3, "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2, "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4, "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2, "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4, "PAY_AMT5": PAY_AMT5, "PAY_AMT6": PAY_AMT6,
    }
    if model_choice != "Use selected model (recommended)":
        payload["model_name"] = model_choice

    with st.spinner("Assessing..."):
        result = api_client.cc_predict(payload)

    st.divider()
    if not result["ok"]:
        status_badge(False, "", f"Assessment failed — {result['error']}")
        if "detail" in result:
            st.json(result["detail"])
    else:
        data = result["data"]
        st.markdown("## Credit Risk Assessment")
        c1, c2, c3, c4 = st.columns(4)
        tone = {"LOW": "green", "MODERATE": "blue", "HIGH": "peach"}[data["risk_level"]]
        with c1:
            metric_card("Risk Level", data["risk_level"], tone=tone)
        with c2:
            metric_card("Predicted Default Probability", f"{data['default_probability'] * 100:.1f}%", tone="blue")
        with c3:
            metric_card("Model", data["model_used"].replace("_", " ").title(), tone="gray")
        with c4:
            metric_card("Dataset", "UCI Credit Card", tone="gray")

        st.warning(
            "**This is an AI-assisted risk assessment, not a lending decision.** "
            "Final lending decisions require appropriate human/institutional review.",
            icon="⚠️",
        )

        st.markdown("### Why this assessment?")
        if not data["explanation_available"]:
            st.caption("Explanation unavailable for this prediction.")
        else:
            st.caption(f"Method: {data['explanation_method']}")
            increasing = [f for f in data["top_contributing_features"] if f["direction"] == "increases risk"]
            decreasing = [f for f in data["top_contributing_features"] if f["direction"] == "decreases risk"]
            col_inc, col_dec = st.columns(2)
            with col_inc:
                st.markdown("**Factors increasing predicted risk**")
                for f in increasing:
                    st.write(f"🔺 {f['feature']} (contribution: {f['contribution']:+.3f})")
                if not increasing:
                    st.caption("None among the top contributors.")
            with col_dec:
                st.markdown("**Factors decreasing predicted risk**")
                for f in decreasing:
                    st.write(f"🔻 {f['feature']} (contribution: {f['contribution']:+.3f})")
                if not decreasing:
                    st.caption("None among the top contributors.")
            st.caption(
                "These factors are model-calculated associations with this prediction, not a "
                "causal or documented institutional reason for a lending decision."
            )
