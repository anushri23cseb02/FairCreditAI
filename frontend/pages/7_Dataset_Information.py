"""
Dataset & Model Information page.

Shows where the target label came from, which rows were kept/excluded,
which protected attributes exist, and a peek at recent predictions
actually stored in MySQL by the backend.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import metric_card, section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="Dataset Information - FairCreditAI", page_icon="🗂️", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("Dataset Information", "Where the data and the label came from, in plain terms.")

result = api_client.get_dataset_info()

if not result["ok"] or not result["data"].get("ready"):
    error = result.get("error") or result["data"].get("error")
    status_badge(False, "", f"Dataset info unavailable — {error}")
else:
    info = result["data"]

    st.markdown(f"#### Dataset: {info['dataset_name']}")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Raw rows", f"{info['raw_row_count']:,}", tone="gray")
    with c2:
        metric_card("Usable rows (after target filter)", f"{info['rows_after_target_filter']:,}", tone="blue")
    with c3:
        metric_card("Rows excluded (ambiguous outcome)", f"{info['rows_excluded_ambiguous_action']:,}", tone="peach")

    st.markdown("#### How the label was decided")
    st.write(
        f"The target column is **{info['target_column']}**, derived from the raw column "
        f"**{info['target_source_column']}**. Only applications with a completed, "
        "comparable outcome are used to train the model:"
    )
    for raw_value, label in info["target_derivation"].items():
        st.write(f"- \"{raw_value}\" → **{label}**")
    st.caption(
        "Everything else (withdrawn applications, incomplete files, loans purchased "
        "by another institution, and preapproval-only statuses) is left out entirely "
        "rather than guessed at, because none of those are a comparable, adjudicated "
        "denial/non-denial decision."
    )

    if info.get("target_class_balance"):
        st.markdown("#### Class balance in the usable data")
        balance = info["target_class_balance"]
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Not denied", f"{balance.get('0', 0) * 100:.1f}%", tone="green")
        with c2:
            metric_card("Denied", f"{balance.get('1', 0) * 100:.1f}%", tone="peach")

    if info.get("split_sizes"):
        st.caption(
            f"Train/validation/test split: {info['split_sizes']['train']:,} / "
            f"{info['split_sizes']['val']:,} / {info['split_sizes']['test']:,} rows "
            f"— {info.get('split_strategy', '')}"
        )

    st.markdown("#### Protected attributes used only for the fairness evaluation")
    st.caption("These columns are never given to the model as prediction inputs.")
    for name, attr in info["protected_attributes"].items():
        st.write(f"- **{name.title()}** (`{attr['column']}`): {attr['privileged']} vs {attr['comparison']}")

st.divider()
st.info("Recent predictions have moved to the **Prediction History** page in the sidebar.")
