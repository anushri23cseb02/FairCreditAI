"""
Model Performance page.

Shows classification metrics (GET /metrics) and model/dataset metadata
(GET /model/info) transparently. Never claims fairness here — that
lives only on the Fairness Audit page.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Model Performance - FairCreditAI", page_icon="📊", layout="wide")

section_header("Model Performance & Information", "Held-out test set metrics for the active model.")

metrics_result = api_client.get_metrics()
info_result = api_client.get_model_info()

if not metrics_result["ok"] or not metrics_result["data"].get("ready"):
    error = metrics_result.get("error") or metrics_result["data"].get("error")
    status_badge(False, "", f"Metrics unavailable — {error}")
else:
    m = metrics_result["data"]["metrics"]
    st.markdown("#### Classification metrics (held-out test set)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Accuracy", f"{m['accuracy'] * 100:.1f}%", tone="blue")
    with c2:
        metric_card("Precision", f"{m['precision'] * 100:.1f}%", tone="blue")
    with c3:
        metric_card("Recall", f"{m['recall'] * 100:.1f}%", tone="blue")
    with c4:
        metric_card("F1", f"{m['f1'] * 100:.1f}%", tone="blue")
    with c5:
        metric_card("ROC-AUC", f"{m['roc_auc']:.3f}", tone="green")

    st.markdown("#### Confusion matrix")
    cm = m["confusion_matrix"]
    st.table(
        {
            "": cm["labels"],
            cm["labels"][0]: [cm["matrix"][0][0], cm["matrix"][1][0]],
            cm["labels"][1]: [cm["matrix"][0][1], cm["matrix"][1][1]],
        }
    )
    st.caption(
        f"Actual positive rate: {m['positive_rate_actual'] * 100:.1f}% · "
        f"Predicted positive rate: {m['positive_rate_predicted'] * 100:.1f}%"
    )
    if metrics_result["data"].get("note"):
        st.caption(metrics_result["data"]["note"])

st.divider()

if not info_result["ok"] or not info_result["data"].get("ready"):
    error = info_result.get("error") or info_result["data"].get("error")
    status_badge(False, "", f"Model info unavailable — {error}")
else:
    info = info_result["data"]
    st.markdown("#### Model & dataset information")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Model type", info["model_type"], tone="gray")
    with c2:
        metric_card("Training rows", str(info["training_rows"]), tone="gray")
    with c3:
        metric_card("Test rows", str(info["test_rows"]), tone="gray")
    st.caption(f"Trained at: {info['trained_at_utc']}")
    with st.expander("Feature columns used by the model"):
        st.write(info["feature_columns"])
    with st.expander("Hyperparameters"):
        st.json(info["hyperparameters"])

st.info(
    "Good metrics on this historical dataset do not by themselves mean the "
    "model is fair. See the Fairness Audit page for group-level outcome "
    "comparisons.",
    icon="ℹ️",
)

st.divider()
st.markdown("#### Retrain the model")
st.caption(
    "Retrains from the already-prepared train/validation/test data (does not "
    "re-read the raw dataset). Uses the same fixed random seed, so results are "
    "reproducible. The newly trained model is loaded immediately — no restart needed."
)
if st.button("Retrain model now"):
    with st.spinner("Retraining... this takes a few seconds."):
        retrain_result = api_client.retrain_model()
    if not retrain_result["ok"]:
        status_badge(False, "", f"Retraining failed — {retrain_result['error']}")
    else:
        r = retrain_result["data"]
        st.success(f"Model retrained (version {r['version']}). New test ROC-AUC: {r['test_metrics']['roc_auc']:.3f}")
        st.rerun()
