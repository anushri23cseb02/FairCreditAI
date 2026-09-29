"""
Prediction History page.

Reads real rows from MySQL (via GET /predictions) -- every prediction
made through the Credit Prediction or Batch Prediction pages ends up
here, since the backend logs each one when it's made.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Prediction History - FairCreditAI", page_icon="📜", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("Prediction History", "Every prediction made through this application, stored in MySQL.")

col1, col2 = st.columns([1, 3])
with col1:
    limit = st.selectbox("Rows to show", [10, 25, 50, 100], index=1)
with col2:
    label_filter = st.selectbox("Filter by result", ["All", "denied", "not_denied"])

history = api_client.get_recent_predictions(limit=limit)

if not history["ok"] or not history["data"].get("ready"):
    error = history.get("error") or history["data"].get("error")
    status_badge(False, "", f"Prediction history unavailable — {error}")
    st.caption("This is expected if no prediction has been made yet, or the database is still starting up.")
    st.stop()

data = history["data"]
rows = data["predictions"]

c1, c2 = st.columns(2)
with c1:
    metric_card("Total predictions stored", str(data.get("total", len(rows))), tone="blue")
with c2:
    metric_card("Rows shown here", str(len(rows)), tone="gray")

if label_filter != "All":
    rows = [r for r in rows if r["predicted_label"] == label_filter]

if not rows:
    st.info("No predictions match this filter yet. Try making one on the Credit Prediction page.")
else:
    st.table(
        [
            {
                "Time (UTC)": r["created_at"],
                "Loan amount ($000s)": r["loan_amount_000s"],
                "Income ($000s)": r["applicant_income_000s"],
                "Loan type": r["loan_type_name"],
                "Loan purpose": r["loan_purpose_name"],
                "Prediction": "Denied" if r["predicted_label"] == "denied" else "Approved",
                "Denial probability": f"{r['denial_probability'] * 100:.1f}%",
                "Model version": r["model_version"],
            }
            for r in rows
        ]
    )
