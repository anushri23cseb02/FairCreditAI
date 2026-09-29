"""
Batch Prediction page.

Upload a CSV of applications, validate its columns, score every row
through the same /predict/batch endpoint the single-prediction page
uses (so it's the same trained pipeline, no retraining), and download
the results.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Batch Prediction - FairCreditAI", page_icon="📁", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("Batch Prediction", "Score many applications at once from a CSV file.")

REQUIRED_COLUMNS = [
    "loan_amount_000s", "applicant_income_000s", "tract_to_msamd_income", "population",
    "minority_population", "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "hud_median_family_income", "has_co_applicant", "loan_type_name", "loan_purpose_name",
    "property_type_name", "owner_occupancy_name", "lien_status_name", "preapproval_name",
]

with st.expander("Required CSV columns"):
    st.code(", ".join(REQUIRED_COLUMNS))
    st.caption("Up to 500 rows per upload. Download a template below if you don't have a file ready.")

template_df = pd.DataFrame([{
    "loan_amount_000s": 250, "applicant_income_000s": 80, "tract_to_msamd_income": 95,
    "population": 5000, "minority_population": 20, "number_of_owner_occupied_units": 1500,
    "number_of_1_to_4_family_units": 1800, "hud_median_family_income": 70000,
    "has_co_applicant": False, "loan_type_name": "Conventional", "loan_purpose_name": "Home purchase",
    "property_type_name": "One-to-four family dwelling (other than manufactured housing)",
    "owner_occupancy_name": "Owner-occupied as a principal dwelling",
    "lien_status_name": "Secured by a first lien", "preapproval_name": "Not applicable",
}])
st.download_button(
    "Download a template CSV", template_df.to_csv(index=False), file_name="batch_template.csv", mime="text/csv"
)

uploaded_file = st.file_uploader("Upload applications CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        status_badge(False, "", f"Could not read this file as a CSV — {exc}")
        st.stop()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        status_badge(False, "", f"This CSV is missing required columns: {', '.join(missing)}")
        st.stop()

    if len(df) == 0:
        status_badge(False, "", "This CSV has no rows.")
        st.stop()

    if len(df) > 500:
        st.warning(f"File has {len(df)} rows; only the first 500 will be scored.")
        df = df.head(500)

    st.markdown(f"#### Preview ({len(df)} rows)")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Run batch prediction", type="primary"):
        applications = df[REQUIRED_COLUMNS].to_dict(orient="records")
        with st.spinner(f"Scoring {len(applications)} applications..."):
            result = api_client.predict_batch(applications)

        if not result["ok"]:
            status_badge(False, "", f"Batch prediction failed — {result['error']}")
            if "detail" in result:
                st.json(result["detail"])
        else:
            results = result["data"]["results"]
            out_df = df.copy()
            out_df["prediction"] = ["Denied" if r["predicted_label"] == "denied" else "Approved" for r in results]
            out_df["denial_probability"] = [r["denial_probability"] for r in results]
            out_df["model_version"] = [r["model_version"] for r in results]

            denied_count = sum(1 for r in results if r["predicted_label"] == "denied")
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Applications scored", str(len(results)), tone="blue")
            with c2:
                metric_card("Predicted denied", str(denied_count), tone="peach")
            with c3:
                metric_card("Predicted approved", str(len(results) - denied_count), tone="green")

            st.markdown("#### Results")
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                "Download results as CSV",
                out_df.to_csv(index=False),
                file_name="batch_prediction_results.csv",
                mime="text/csv",
            )
