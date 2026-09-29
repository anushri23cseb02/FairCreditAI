"""Sidebar branding block shown on every page."""
import streamlit as st


def render_sidebar_branding() -> None:
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size:1.3rem; font-weight:700; color:#2F2E35;">
                FairCredit AI
            </div>
            <div style="font-size:0.8rem; color:#6B6A72;">
                Explainable &amp; Fair Credit Risk Prediction
            </div>
        </div>
        <hr style="margin-top:0; margin-bottom:1rem; border-color:#E4E2E8;">
        """,
        unsafe_allow_html=True,
    )
