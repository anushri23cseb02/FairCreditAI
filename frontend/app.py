"""
FairCredit AI - Dashboard (Streamlit entrypoint).

Run with:
    streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501

Streamlit contains NO business logic. Every number here comes from the
FastAPI backend (frontend/services/api_client.py) — nothing is
hardcoded.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.components.sidebar import render_sidebar_branding
from frontend.services import api_client

st.set_page_config(
    page_title="FairCredit AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()
render_sidebar_branding()
st.sidebar.markdown(
    "**Use the pages in the sidebar** to make a prediction, run a batch of "
    "predictions, review prediction history, check fairness results, or "
    "see how the model performs."
)

section_header(
    "FairCredit AI — Dashboard",
    "Explainable & Fair Credit Risk Prediction System.",
)

st.write(
    "This project predicts whether a mortgage application is likely to be denied, "
    "using real historical HMDA data for Washington State (2016). Every prediction "
    "comes with an explanation of which factors drove it, and the system reports "
    "whether the model's outcomes differ across applicant groups, rather than "
    "reporting accuracy alone."
)

st.info(
    "**This app now covers two independent tracks.** The pages above (Credit Prediction, "
    "Fairness Audit, Model Performance, Dataset Information) are the **HMDA mortgage-denial** "
    "track. **Credit Card Prediction, Model Comparison, and Credit Card Fairness** (further down "
    "the sidebar) are a separate **credit-card default** track, using a different dataset, target, "
    "and three compared models. The two are deliberately kept separate — they model different "
    "populations and outcomes, and are never merged.",
    icon="🔀",
)

# --- Live prediction stats, from MySQL --------------------------------
stats_result = api_client.get_prediction_stats()
metrics_result = api_client.get_metrics()

st.markdown("#### Prediction activity")
if not stats_result["ok"] or not stats_result["data"].get("ready"):
    st.caption("Prediction stats unavailable right now — this appears once the database is reachable.")
else:
    s = stats_result["data"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total predictions made", str(s["total"]), tone="blue")
    with c2:
        metric_card("Predicted denied", str(s.get("denied", 0)), tone="peach")
    with c3:
        metric_card("Predicted approved", str(s.get("not_denied", 0)), tone="green")
    with c4:
        rate = s.get("denial_rate")
        metric_card("Denial rate (this app's usage)", f"{rate * 100:.1f}%" if rate is not None else "—", tone="gray")

st.markdown("#### Current model")
if not metrics_result["ok"] or not metrics_result["data"].get("ready"):
    st.caption("Model metrics unavailable right now.")
else:
    m = metrics_result["data"]["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Model version", metrics_result["data"]["version"], tone="gray")
    with c2:
        metric_card("Test accuracy", f"{m['accuracy'] * 100:.1f}%", tone="blue")
    with c3:
        metric_card("Test F1 score", f"{m['f1'] * 100:.1f}%", tone="blue")
    with c4:
        metric_card("Test ROC-AUC", f"{m['roc_auc']:.3f}", tone="green")

# --- Quick system status -----------------------------------------------
st.markdown("#### System status")
col1, col2, col3 = st.columns(3)

with col1:
    health = api_client.get_health()
    status_badge(health["ok"], "Backend: Connected", f"Backend: Unreachable — {health.get('error', '')}")

with col2:
    detailed = api_client.get_health_detailed()
    db_ok = detailed["ok"] and detailed["data"].get("database") == "connected"
    status_badge(db_ok, "Database: Connected", "Database: Unreachable")

with col3:
    model_ok = metrics_result["ok"] and metrics_result["data"].get("ready", False)
    status_badge(model_ok, "Model: Loaded", "Model: Not loaded")

st.caption("See the **System Status** page in the sidebar for the full, detailed check.")

st.divider()
st.caption(
    "FairCredit AI — final-year project, built on real HMDA Washington State "
    "2016 mortgage data. No synthetic data, no fabricated metrics."
)
