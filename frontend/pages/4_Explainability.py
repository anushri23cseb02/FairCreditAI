"""
Explainability page.

Global feature importance for the trained model — which inputs the
model leans on most overall, across every prediction it makes. This
is distinct from the per-prediction explanation shown on the Credit
Prediction page (which explains one specific applicant's result).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Explainability - FairCreditAI", page_icon="🔍", layout="wide")

section_header("Explainability", "Which factors the model relies on most, overall.")

st.info(
    "This model is a Logistic Regression, so its logit score is literally "
    "intercept + sum(coefficient × feature value) — the contribution of "
    "each feature is exact, not an approximation. These are statistical "
    "associations the model learned from historical data, not causal "
    "claims about any applicant.",
    icon="ℹ️",
)

result = api_client.get_feature_importance()

if not result["ok"] or not result["data"].get("ready"):
    error = result.get("error") or result["data"].get("error")
    status_badge(False, "", f"Feature importance unavailable — {error}")
    st.stop()

features = result["data"]["features"]
df = pd.DataFrame(features)
df = df.rename(columns={"feature": "Feature", "importance": "Importance (|coefficient| sum)", "relative_importance": "Share of total"})

st.markdown("#### Global feature importance")
st.bar_chart(df.set_index("Feature")["Importance (|coefficient| sum)"])

st.markdown("#### Details")
st.dataframe(
    df.assign(**{"Share of total": (df["Share of total"] * 100).round(1).astype(str) + "%"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Categorical features (e.g. loan type) are shown as one combined bar — the sum "
    "across every category of that feature — rather than one bar per category, so "
    "the chart reflects how much the feature as a whole matters to the model."
)
