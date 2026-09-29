"""
System Status page.

Every status shown here comes from an actual check made right now
(GET /health/detailed, GET /model/info, GET /dataset/info) — nothing is
a static "healthy" label.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import section_header, status_badge
from frontend.services import api_client

st.set_page_config(page_title="System Status - FairCreditAI", page_icon="🩺", layout="wide")
from frontend.services.auth_gate import require_auth
require_auth()

section_header("System Status", "Live status of every component, checked just now.")

health = api_client.get_health_detailed()
model_info = api_client.get_model_info()
dataset_info = api_client.get_dataset_info()

backend_up = health["ok"]
db_connected = health["ok"] and health["data"].get("database") == "connected"
model_loaded = model_info["ok"] and model_info["data"].get("ready", False)
dataset_ready = dataset_info["ok"] and dataset_info["data"].get("ready", False)

rows = [
    ("Frontend", True, "This Streamlit app is rendering, so it's running."),
    ("Backend (FastAPI)", backend_up, "Checked via GET /health/detailed just now."),
    ("Database (MySQL)", db_connected, "Checked via a live SELECT 1 query just now."),
    ("Model", model_loaded, "Checked via GET /model/info — confirms a trained artifact is loaded in memory."),
    ("Dataset", dataset_ready, "Checked via GET /dataset/info — confirms the processed data summary exists."),
]

for name, ok, detail in rows:
    status_badge(ok, f"{name}: Connected", f"{name}: Unreachable / Not ready")
    st.caption(detail)
    st.write("")

if not backend_up:
    st.error(f"Backend error detail: {health.get('error')}")

st.divider()
if backend_up:
    st.markdown("#### Raw health response")
    st.json(health["data"])
