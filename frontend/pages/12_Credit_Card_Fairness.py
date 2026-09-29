"""
Fairness Audit page (Track A: credit-card default).

Shows the SEX audit (the only true protected attribute in this dataset)
and the AGE-band proxy check, then the real before/after
ThresholdOptimizer mitigation.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Credit Card Fairness - FairCreditAI", page_icon="⚖️", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("Fairness Audit — Credit Card Track", "Group outcome differences on the held-out test set, before and after mitigation.")

st.warning(
    "Fairness metrics measure differences in model outcomes between groups. They do not "
    "by themselves establish whether discrimination occurred. Results depend on the dataset, "
    "population, sample size, model, threshold and fairness definition used.",
    icon="⚠️",
)

fairness_result = api_client.cc_fairness()
if not fairness_result["ok"] or not fairness_result["data"].get("ready"):
    status_badge(False, "", f"Fairness audit unavailable — {fairness_result.get('error') or fairness_result['data'].get('error')}")
    st.stop()

data = fairness_result["data"]
st.caption(f"Model audited: {data['model_used']} · Test rows: {data['test_rows']}")

for attr_key, title in [("sex", "Sex"), ("age_band", "Age Band (proxy check)")]:
    attr = data[attr_key]
    if "error" in attr:
        continue
    st.markdown(f"### {title}: {attr['privileged_group']} vs {attr['comparison_group']}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"Rate — {attr['privileged_group']}", f"{attr['selection_rate_by_group'][attr['privileged_group']]*100:.1f}%", tone="blue")
    with c2:
        metric_card(f"Rate — {attr['comparison_group']}", f"{attr['selection_rate_by_group'][attr['comparison_group']]*100:.1f}%", tone="peach")
    with c3:
        metric_card("Demographic parity difference", f"{attr['demographic_parity_difference']:.3f}", tone="gray")
    with c4:
        di = attr["disparate_impact_ratio"]
        tone = "peach" if (di is not None and (di < 0.8 or di > 1.25)) else "green"
        metric_card("Disparate impact ratio", f"{di:.3f}" if di is not None else "—", tone=tone)
    st.caption(
        f"Sample sizes: {attr['privileged_group']}={attr['group_sample_sizes'][attr['privileged_group']]}, "
        f"{attr['comparison_group']}={attr['group_sample_sizes'][attr['comparison_group']]}"
    )
    with st.expander("Definitions and limitations"):
        for k, v in attr["metric_definitions"].items():
            st.write(f"- **{k}**: {v}")
        for lim in attr["limitations"]:
            st.write(f"- {lim}")
    st.divider()

st.markdown("## Mitigation: Fairlearn ThresholdOptimizer")
st.caption("Applied post-hoc to the selected model's probabilities, constrained on equalized odds, using SEX as the sensitive feature. Fit on train, evaluated on held-out test.")

mitigation_result = api_client.cc_fairness_mitigation()
if not mitigation_result["ok"] or not mitigation_result["data"].get("ready"):
    status_badge(False, "", "Mitigation results unavailable.")
    st.stop()

m = mitigation_result["data"]
st.markdown("### Predictive metrics: before vs after")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Before mitigation**")
    for k, v in m["predictive_metrics_before"].items():
        st.write(f"{k}: {v}")
with col2:
    st.markdown("**After mitigation**")
    for k, v in m["predictive_metrics_after"].items():
        st.write(f"{k}: {v}")

st.markdown("### Fairness: before vs after")
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Before mitigation**")
    metric_card("Disparate impact ratio", f"{m['fairness_before']['disparate_impact_ratio']:.3f}", tone="peach")
    metric_card("Demographic parity difference", f"{m['fairness_before']['demographic_parity_difference']:.3f}", tone="peach")
with col4:
    st.markdown("**After mitigation**")
    metric_card("Disparate impact ratio", f"{m['fairness_after']['disparate_impact_ratio']:.3f}", tone="green")
    metric_card("Demographic parity difference", f"{m['fairness_after']['demographic_parity_difference']:.3f}", tone="green")

st.info(m["note"], icon="ℹ️")
st.caption(
    "Fairness metrics indicate the following group-level disparities under the selected "
    "definition — this is not a claim that the model is now 'fair,' only that the "
    "equalized-odds trade-off point has moved."
)
