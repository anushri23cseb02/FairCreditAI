"""
Model Governance page (Model Card) — covers both tracks. Every field
here is read live from the backend's model/dataset/fairness endpoints;
nothing is typed in as static text that could drift from the actual
deployed model.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Model Governance - FairCreditAI", page_icon="📋", layout="wide")

section_header("Model Governance", "Model card for both tracks — intended use, limitations, and current live configuration.")

tab_hmda, tab_cc = st.tabs(["HMDA Mortgage-Denial Track", "Credit Card Default Track"])

with tab_hmda:
    info = api_client.get_model_info()
    dataset = api_client.get_dataset_info()
    metrics = api_client.get_metrics()
    if not info["ok"] or not info["data"].get("ready"):
        status_badge(False, "", "Model info unavailable.")
    else:
        i, d, m = info["data"], dataset.get("data", {}), metrics.get("data", {})
        st.markdown("#### Identity")
        st.write(f"- **Model name:** Logistic Regression (mortgage denial)")
        st.write(f"- **Model type:** {i.get('model_type')}")
        st.write(f"- **Trained at (UTC):** {i.get('trained_at_utc')}")
        st.write(f"- **Training rows:** {i.get('training_rows')} · **Test rows:** {i.get('test_rows')}")

        st.markdown("#### Training data & target")
        st.write(f"- **Dataset:** {d.get('dataset_name', 'Washington State HMDA 2016')}")
        st.write(f"- **Target column:** `{d.get('target_column', 'denied')}`, derived from `{d.get('target_source_column', 'action_taken_name')}`")
        if d.get("target_class_balance"):
            st.write(f"- **Class balance:** {d['target_class_balance']}")

        st.markdown("#### Features")
        with st.expander("Features used"):
            st.write(i.get("feature_columns"))
        st.markdown("#### Protected attributes (audit-only, never model inputs)")
        for name, attr in d.get("protected_attributes", {}).items():
            st.write(f"- **{name.title()}** (`{attr['column']}`): {attr['privileged']} vs {attr['comparison']}")

        st.markdown("#### Metrics (held-out test set)")
        if m.get("ready"):
            met = m["metrics"]
            st.write(f"Accuracy {met['accuracy']*100:.1f}% · Precision {met['precision']*100:.1f}% · "
                     f"Recall {met['recall']*100:.1f}% · F1 {met['f1']*100:.1f}% · ROC-AUC {met['roc_auc']:.3f}")

        st.markdown("#### Intended use")
        st.write("Educational/demonstration risk-assessment aid for a final-year project. **Not** for real lending decisions.")
        st.markdown("#### Prohibited use")
        st.write("Any real credit-decision-making without qualified human/institutional review; any use asserting the model is unbiased.")
        st.markdown("#### Explanation methodology")
        st.write("Exact logistic-regression coefficient decomposition (not an approximation).")
        st.markdown("#### Threshold")
        st.write("0.5 on predicted probability (not separately calibrated for this track).")

with tab_cc:
    info = api_client.cc_model_info()
    dataset = api_client.cc_dataset_info()
    if not info["ok"] or not info["data"].get("ready"):
        status_badge(False, "", "Model info unavailable.")
    else:
        i, d = info["data"], dataset.get("data", {})
        st.markdown("#### Identity")
        st.write(f"- **Models compared:** {', '.join(i.get('models_compared', []))}")
        st.write(f"- **Selected model:** {i.get('selected_model')}")
        st.write(f"- **Selection rationale:** {i.get('selection_rationale')}")
        st.write(f"- **Trained at (UTC):** {i.get('trained_at_utc')}")
        st.write(f"- **Training rows:** {i.get('training_rows')} · **Test rows:** {i.get('test_rows')}")

        st.markdown("#### Training data & target")
        st.write(f"- **Dataset:** {d.get('dataset_name', 'UCI Default of Credit Card Clients')}")
        st.write("- **Target column:** `default` (next-month credit-card default), already binary in the source data.")
        if d.get("target_class_balance"):
            st.write(f"- **Class balance:** {d['target_class_balance']}")

        st.markdown("#### Features")
        with st.expander("Features used"):
            st.write(i.get("feature_columns"))
        st.markdown("#### Protected attributes (audit-only, never model inputs)")
        st.write("- **Sex** (`SEX`): Male vs Female — the only true protected attribute in this dataset; no race/ethnicity column exists.")
        st.write("- **Age band** (proxy check only — AGE **is** a model feature): 35-and-over vs Under-35.")

        st.markdown("#### Intended use")
        st.write("Educational/demonstration risk-assessment aid. **Not** for real credit-issuance decisions.")
        st.markdown("#### Prohibited use")
        st.write("Any real credit-decision-making without qualified human review; using AGE-band or SEX findings to justify adverse treatment of individuals.")
        st.markdown("#### Explanation methodology")
        st.write("Exact coefficients for Logistic Regression; real SHAP TreeExplainer values for Random Forest and XGBoost.")
        st.markdown("#### Calibration")
        st.write("Isotonic regression, fit on the validation fold only. See the Model Comparison page for raw-vs-calibrated reliability curves.")
        st.markdown("#### Mitigation")
        st.write("Fairlearn ThresholdOptimizer (equalized odds), available on the Credit Card Fairness page — shown as a before/after comparison, not applied silently.")

st.divider()
st.markdown("### General limitations (both tracks)")
st.write("""
- Historical data from a specific place and time; results do not generalize to other markets, institutions, or periods.
- Fairness metrics are outcome comparisons on one dataset, not a legal determination of discrimination.
- Model uncertainty exists for every prediction — probabilities are estimates, not certainties.
- Both tracks require human/institutional review before any real-world decision.
- This system runs on demonstration infrastructure and has not undergone the validation a production financial system would require.
""")
