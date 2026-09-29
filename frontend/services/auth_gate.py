"""
Opt-in login gate for the Streamlit app.

Only active when both APP_BASIC_AUTH_USER and APP_BASIC_AUTH_PASSWORD
are set (e.g. on a public cloud deployment) -- local/dev usage via
docker-compose is unaffected unless those variables are also set there.
"""
import os

import streamlit as st

_USER = os.getenv("APP_BASIC_AUTH_USER", "")
_PASSWORD = os.getenv("APP_BASIC_AUTH_PASSWORD", "")


def require_auth() -> None:
    if not _USER or not _PASSWORD:
        return

    if st.session_state.get("authenticated"):
        return

    st.title("FairCredit AI")
    st.caption("Sign in to continue.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if username == _USER and password == _PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.stop()
