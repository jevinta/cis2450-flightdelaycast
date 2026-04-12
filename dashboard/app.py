"""Streamlit dashboard: delay-risk demo + optional EDA figures (for class presentation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import joblib
import pandas as pd
import streamlit as st

from flightdelaycast.config import DELAY_THRESHOLD_MINUTES, MODELS_DIR, REPORTS_FIGURES
from flightdelaycast.model_features import prediction_dataframe

MODEL_PATH = MODELS_DIR / "baseline_logistic.joblib"
FEATURES_PATH = MODELS_DIR / "baseline_features.json"
METRICS_PATH = MODELS_DIR / "baseline_metrics.json"


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file() or not FEATURES_PATH.is_file():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.is_file() else {}
    return model, features, metrics


st.set_page_config(page_title="Flight Delay Cast", layout="wide")
st.title("Flight Delay Cast")
st.caption(f"Delayed = arrival delay **> {DELAY_THRESHOLD_MINUTES} minutes** (per project definition)")

model, feat_meta, metrics = load_model_bundle()

tab_demo, tab_eda = st.tabs(["Delay risk demo", "EDA snapshots"])

with tab_demo:
    if model is None or feat_meta is None:
        st.warning(
            "No trained model found in `models/`. Train locally, then commit or upload artifacts for deployment:\n\n"
            "`python scripts/build_processed.py` → `python scripts/train_baseline.py`\n\n"
            "You need `baseline_logistic.joblib` and `baseline_features.json` next to each other."
        )
    else:
        st.success("Loaded baseline logistic regression pipeline.")
        if metrics:
            st.caption(
                f"Holdout F1: **{metrics.get('f1', 'n/a')}** · "
                f"accuracy: **{metrics.get('accuracy', 'n/a')}** · "
                f"test delay rate: **{metrics.get('delay_rate_test', 'n/a')}**"
            )

        num_cols = feat_meta["numeric"]
        cat_cols = feat_meta["categorical"]
        weather_keys = [c for c in num_cols if c.startswith("w_")]

        c1, c2, c3 = st.columns(3)
        with c1:
            carrier = st.text_input("Carrier (IATA)", value="AA")
            origin = st.text_input("Origin", value="JFK")
            dest = st.text_input("Destination", value="LAX")
        with c2:
            dep_hour = st.slider("Scheduled dep hour (fractional)", 0.0, 23.99, 14.5, 0.25)
            month = st.number_input("Month", 1, 12, 1)
            _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow = st.selectbox("Day of week", list(range(7)), index=0, format_func=lambda i: _days[i])
        with c3:
            distance = st.number_input("Distance (mi)", 50.0, 6000.0, 2475.0, 25.0)

        weather_vals: dict | None = None
        if weather_keys:
            st.subheader("Weather at origin (optional; leave blank → median imputation like training)")
            wc = st.columns(min(len(weather_keys), 4))
            weather_vals = {}
            for i, k in enumerate(weather_keys):
                with wc[i % len(wc)]:
                    raw = st.text_input(k, "", key=f"w_{k}", help="Optional number; empty = missing")
                    if raw.strip():
                        weather_vals[k] = float(raw)

        row = prediction_dataframe(
            num_cols=num_cols,
            cat_cols=cat_cols,
            carrier=carrier,
            origin=origin,
            dest=dest,
            dep_hour=dep_hour,
            month=int(month),
            day_of_week=int(dow),
            distance=distance,
            weather=weather_vals,
        )
        proba = model.predict_proba(row)[0, 1]
        st.metric("Estimated P(delayed)", f"{proba:.1%}")
        st.caption("This is an illustrative baseline; re-train on full BTS + weather for production-quality estimates.")

with tab_eda:
    st.markdown("Figures from `scripts/run_eda.py` (commit `reports/figures/*.png` for Streamlit Cloud).")
    fig_dir = REPORTS_FIGURES
    if fig_dir.is_dir():
        for name in sorted(fig_dir.glob("*.png")):
            st.subheader(name.stem.replace("_", " ").title())
            st.image(str(name), use_container_width=True)
    else:
        st.info(f"No figures at `{fig_dir}` yet.")
