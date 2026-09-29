"""Small reusable pastel-styled UI components for Streamlit pages."""
import streamlit as st


def metric_card(label: str, value: str, tone: str = "blue") -> None:
    """
    Renders a pastel metric card.
    tone: one of "blue", "green", "peach", "gray"
    """
    css_class = f"fc-card fc-card-{tone}"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="fc-metric-label">{label}</div>
            <div class="fc-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(is_ok: bool, ok_text: str, bad_text: str) -> None:
    css_class = "fc-status-ok" if is_ok else "fc-status-bad"
    text = ok_text if is_ok else bad_text
    st.markdown(f'<span class="{css_class}">{text}</span>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="fc-header">
            <h3 style="margin-bottom:0;">{title}</h3>
            <p style="color:#6B6A72; margin-top:0.2rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
