"""Streamlit shell for the delay-risk dashboard (wire up after model + data are ready)."""

import sys
from pathlib import Path

# Allow `import flightdelaycast` when running: streamlit run dashboard/app.py
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import streamlit as st

from flightdelaycast.config import DELAY_THRESHOLD_MINUTES

st.set_page_config(page_title="Flight Delay Cast", layout="wide")
st.title("Flight Delay Cast")
st.caption(f"Target definition: delayed = arrival delay > {DELAY_THRESHOLD_MINUTES} minutes")

st.info(
    "Placeholder dashboard. After Julia delivers merged flight+weather data and trained "
    "models, add inputs (airports, carrier, time) and display predicted delay risk + factors."
)
