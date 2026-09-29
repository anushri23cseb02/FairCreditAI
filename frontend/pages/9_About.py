"""About FairCredit AI — a short, plain-language page for evaluators."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.components.cards import section_header

st.set_page_config(page_title="About - FairCreditAI", page_icon="📘", layout="wide")

section_header("About FairCredit AI", "Explainable & Fair Credit Risk Prediction System")

st.write(
    """
**FairCredit AI** is a final-year project that predicts whether a mortgage application
is likely to be denied, using real historical lending data (HMDA, Washington State, 2016).

The project was built around one idea: a credit model shouldn't just be accurate — it
should also be possible to explain, and its outcomes should be checked for differences
across applicant groups rather than assumed to be fine.

**What it does:**
- Trains a Logistic Regression model on real mortgage-application data
- Predicts approval/denial for a single applicant or a batch of applicants
- Explains individual predictions using the model's own coefficients (exact, not approximated)
- Reports fairness metrics across sex, race, and ethnicity — with sample sizes and definitions,
  not a bare "fair"/"unfair" label
- Stores every prediction in MySQL so there's a real history to look back on
- Serves everything through a documented REST API (FastAPI) and this dashboard (Streamlit)

**What it is not:**
This is a demonstration and learning project, not a production lending system. It should
never be used to make a real credit decision about a real person.

See the **Dataset Information** page for where the data and the target label came from,
and the **Fairness Audit** page for the actual measured results.
"""
)
