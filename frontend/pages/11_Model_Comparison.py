"""
Model Comparison page (Track A: credit-card default).

Shows the 3 compared models side by side and the transparent, weighted
formula that picked the selected one -- never "highest accuracy wins."
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Model Comparison - FairCreditAI", page_icon="🧮", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("Model Comparison", "Track A — Logistic Regression vs Random Forest vs XGBoost, on the same held-out test set.")

info_result = api_client.cc_model_info()
comparison_result = api_client.cc_model_comparison()

if not info_result["ok"] or not info_result["data"].get("ready"):
    status_badge(False, "", f"Model info unavailable — {info_result.get('error') or info_result['data'].get('error')}")
    st.stop()

info = info_result["data"]

st.markdown("### Why this model was selected")
st.write(info["selection_rationale"])

weights = info["selection_weights"]
st.caption(
    f"Selection formula: {weights['roc_auc']} × ROC-AUC + {weights['f1']} × F1 + "
    f"{weights['calibration']} × calibration + {weights['explainability']} × explainability. "
    "These weights are a documented judgment call, not a measured quantity — changing them "
    "would change which model wins, and that trade-off is intentionally visible here rather than hidden."
)

st.markdown("### Composite scores")
scores = info["selection_scores"]
score_df = pd.DataFrame(scores).T
score_df.index.name = "model"
st.dataframe(score_df, use_container_width=True)

if not comparison_result["ok"] or not comparison_result["data"].get("ready"):
    status_badge(False, "", "Detailed comparison unavailable.")
    st.stop()

results = comparison_result["data"]["results"]

st.markdown("### Held-out test set metrics (uncalibrated final model)")
rows = []
for name, r in results.items():
    m = r["held_out_test_metrics_uncalibrated_final_model"]
    rows.append({
        "Model": name.replace("_", " ").title(),
        "Accuracy": f"{m['accuracy']*100:.1f}%",
        "Precision": f"{m['precision']*100:.1f}%",
        "Recall": f"{m['recall']*100:.1f}%",
        "F1": f"{m['f1']*100:.1f}%",
        "ROC-AUC": f"{m['roc_auc']:.3f}",
        "PR-AUC": f"{m['pr_auc']:.3f}",
        "Brier Score (lower=better calibrated)": f"{m['brier_score']:.4f}",
        "Selected": "✅" if name == info["selected_model"] else "",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(
    "Note the trade-off: the selected model is not necessarily the one with the highest "
    "ROC-AUC — see the composite score table above for why."
)

st.markdown("### ROC and Precision-Recall curves")
model_for_curves = st.selectbox(
    "Model", list(results.keys()), format_func=lambda n: n.replace("_", " ").title(), key="curves_model"
)
curves = results[model_for_curves]["roc_pr_curves"]
col_roc, col_pr = st.columns(2)
with col_roc:
    st.markdown("**ROC curve**")
    roc_df = pd.DataFrame({"False Positive Rate": curves["roc_fpr"], "True Positive Rate": curves["roc_tpr"]})
    st.line_chart(roc_df.set_index("False Positive Rate"))
with col_pr:
    st.markdown("**Precision-Recall curve**")
    pr_df = pd.DataFrame({"Recall": curves["pr_recall"], "Precision": curves["pr_precision"]})
    st.line_chart(pr_df.set_index("Recall"))

st.markdown("### Calibration: raw vs isotonic-calibrated probabilities")
model_for_calibration = st.selectbox("Model", list(results.keys()), format_func=lambda n: n.replace("_", " ").title())
r = results[model_for_calibration]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Uncalibrated**")
    curve = r["calibration_curve_uncalibrated"]
    chart_df = pd.DataFrame({
        "Mean predicted probability": curve["mean_predicted_probability"],
        "Actual fraction positive": curve["fraction_positive"],
    })
    st.line_chart(chart_df.set_index("Mean predicted probability"))
    m = r["held_out_test_metrics_uncalibrated_final_model"]
    metric_card("Brier score (uncalibrated)", f"{m['brier_score']:.4f}", tone="peach")
with col2:
    st.markdown("**Isotonic-calibrated**")
    curve_c = r["calibration_curve_calibrated"]
    chart_df_c = pd.DataFrame({
        "Mean predicted probability": curve_c["mean_predicted_probability"],
        "Actual fraction positive": curve_c["fraction_positive"],
    })
    st.line_chart(chart_df_c.set_index("Mean predicted probability"))
    m_c = r["held_out_test_metrics_calibrated"]
    metric_card("Brier score (calibrated)", f"{m_c['brier_score']:.4f}", tone="green")

st.caption(
    "A perfectly calibrated model's line would sit on the diagonal (predicted probability "
    "== actual fraction positive). Calibration was fit on the validation fold only, never "
    "on the test set shown here."
)

st.divider()
st.markdown("#### Retrain all 3 models")
st.caption("Retrains from the already-prepared data splits and re-runs the selection formula. Takes a bit longer than the HMDA retrain (3 models).")
if st.button("Retrain all 3 models now"):
    with st.spinner("Training Logistic Regression, Random Forest, and XGBoost..."):
        retrain_result = api_client.cc_retrain()
    if not retrain_result["ok"]:
        status_badge(False, "", f"Retraining failed — {retrain_result['error']}")
    else:
        st.success(f"Retrained. Selected model: {retrain_result['data']['selected_model']}")
        st.rerun()
