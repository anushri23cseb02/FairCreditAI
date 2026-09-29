"""
Fairness Audit page.

Displays the group fairness metrics from GET /fairness transparently,
with the exact metric definitions and limitations the backend
documents — never a bare "fair" / "unfair" verdict.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Fairness Audit - FairCreditAI", page_icon="⚖️", layout="wide")

section_header("Fairness Audit", "Group-level outcome metrics on the held-out test set.")

result = api_client.get_fairness_report()

if not result["ok"] or not result["data"].get("ready"):
    error = result.get("error") or result["data"].get("error")
    status_badge(False, "", f"Fairness report unavailable — {error}")
    st.stop()

report = result["data"]
meta = report.get("_meta", {})

st.warning(
    "Fairness metrics are statistical indicators used to identify differences in "
    "model outcomes between groups. They do not by themselves prove discrimination. "
    "This system makes no claim that the model 'is fair' or 'is unbiased.'",
    icon="⚠️",
)

col1, col2 = st.columns(2)
with col1:
    metric_card("Model version", meta.get("model_version", "—"), tone="gray")
with col2:
    metric_card("Test rows evaluated", str(meta.get("test_rows", "—")), tone="gray")

for attribute in ["sex", "race", "ethnicity"]:
    attr = report.get(attribute)
    if not attr or "error" in attr:
        continue

    st.divider()
    st.markdown(f"### {attribute.title()}: {attr['privileged_group']} vs {attr['comparison_group']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            f"Denial rate — {attr['privileged_group']}",
            f"{attr['denial_rate_by_group'][attr['privileged_group']] * 100:.1f}%",
            tone="blue",
        )
    with c2:
        metric_card(
            f"Denial rate — {attr['comparison_group']}",
            f"{attr['denial_rate_by_group'][attr['comparison_group']] * 100:.1f}%",
            tone="peach",
        )
    with c3:
        metric_card("Approval-rate difference between groups", f"{attr['demographic_parity_difference']:.3f}", tone="gray")
    with c4:
        di = attr["disparate_impact_ratio"]
        tone = "peach" if (di is not None and (di < 0.8 or di > 1.25)) else "green"
        metric_card("Disparate impact ratio (denial rate ÷ denial rate)", f"{di:.3f}" if di is not None else "—", tone=tone)

    st.caption(
        f"Group sizes: {attr['privileged_group']}={attr['group_sample_sizes'][attr['privileged_group']]}, "
        f"{attr['comparison_group']}={attr['group_sample_sizes'][attr['comparison_group']]}"
    )

    with st.expander("Metric definitions and limitations"):
        st.markdown("**Definitions**")
        for k, v in attr["metric_definitions"].items():
            st.write(f"- **{k}**: {v}")
        st.markdown("**Limitations**")
        for lim in attr["limitations"]:
            st.write(f"- {lim}")
