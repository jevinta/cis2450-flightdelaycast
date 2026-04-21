"""Streamlit dashboard: overview + delay-risk demo + EDA (CIS 2450 presentation)."""

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

# Short captions for EDA PNGs (edit as your story evolves)
EDA_CAPTIONS: dict[str, str] = {
    "01_class_balance": "Most flights are on time — class imbalance matters for metrics and modeling choices.",
    "02_delay_rate_by_hour": "Delay rate varies by scheduled departure hour; time-of-day is a strong signal.",
    "03_delay_rate_by_carrier": "Carriers differ systematically; airline is a useful categorical feature.",
    "04_correlation_numeric": "Relationships among numeric fields inform feature scaling and redundancy.",
}

SPEAKER_OVERVIEW = """
- **Hook:** Delays cost passengers and airlines; estimating risk before departure is a practical decision-support problem.
- **Data:** U.S. BTS domestic flights plus optional daily weather at the origin (Meteostat), merged on airport and date.
- **Target:** Binary *delayed* if arrival delay is strictly greater than 15 minutes (matches `DELAY_THRESHOLD_MINUTES` in code).
- **Model (today):** Scikit-learn pipeline with preprocessing and **logistic regression** (`class_weight='balanced'` for imbalance).
- **Demo:** Walk through one realistic itinerary, then contrast a peak-hour hub departure vs an off-peak case if time allows.
"""

SPEAKER_DEMO = """
- **Inputs:** Explain carrier / route / schedule fields; mention distance as a proxy for stage length and complexity.
- **Weather:** If columns exist, note they are optional in the UI and imputed like training when omitted.
- **Output:** Probability is **P(delayed)** on the holdout definition — not the same as “minutes late.”
- **Caveat:** Emphasize illustrative baseline; performance depends on which months/years you trained on.
"""


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file() or not FEATURES_PATH.is_file():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.is_file() else {}
    return model, features, metrics


def _risk_tone(p: float) -> tuple[str, str]:
    if p >= 0.55:
        return "High delay risk", "🔴"
    if p >= 0.35:
        return "Moderate delay risk", "🟡"
    return "Lower delay risk", "🟢"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          div[data-testid="stMetricValue"] { font-size: 2.1rem; }
          .fdc-card {
            border: 1px solid rgba(37, 99, 235, 0.2);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            background: linear-gradient(135deg, rgba(37,99,235,0.06), rgba(255,255,255,0));
            margin-bottom: 1rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tab_overview() -> None:
    st.markdown('<div class="fdc-card">', unsafe_allow_html=True)
    st.subheader("What this project does")
    st.markdown(
        f"""
        **Goal:** Estimate the probability that a U.S. domestic flight will be **delayed**
        (arrival delay **> {DELAY_THRESHOLD_MINUTES} minutes**), using operational fields and optional origin weather.

        **Why it matters:** Passengers miss connections; airlines absorb crew and gate costs. A transparent risk score
        helps explain *which factors* push risk up or down—not just a black-box label.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data sources**")
        st.markdown(
            """
            - **BTS** — U.S. Bureau of Transportation Statistics domestic flights (schedules, carriers, delay outcomes).
            - **Weather** — Daily summaries at the flight **origin** (optional in your processed build).
            """
        )
    with c2:
        st.markdown("**Modeling (current app)**")
        st.markdown(
            """
            - **Preprocessing:** Imputation, scaling for numerics, one-hot encoding for categoricals.
            - **Classifier:** Logistic regression baseline with **balanced class weights** for skewed *on-time* rates.
            """
        )

    with st.expander("Speaker notes — Overview tab", expanded=False):
        st.markdown(SPEAKER_OVERVIEW)

    st.divider()
    st.caption("Tip for your talk: keep this tab visible while you introduce the problem, then switch to **Delay risk demo** for the live walkthrough.")


def tab_demo(model, feat_meta, metrics) -> None:
    if model is None or feat_meta is None:
        st.warning(
            "No trained model found in `models/`. Train locally, then commit or upload artifacts for deployment:\n\n"
            "`python scripts/build_processed.py` → `python scripts/train_baseline.py`\n\n"
            "You need `baseline_logistic.joblib` and `baseline_features.json` next to each other."
        )
        return

    st.success("Loaded baseline logistic regression pipeline.")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Holdout F1", f"{metrics.get('f1', 0):.3f}" if isinstance(metrics.get("f1"), (int, float)) else metrics.get("f1", "n/a"))
        m2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}" if isinstance(metrics.get("accuracy"), (int, float)) else metrics.get("accuracy", "n/a"))
        m3.metric("Precision", f"{metrics.get('precision', 0):.3f}" if isinstance(metrics.get("precision"), (int, float)) else metrics.get("precision", "n/a"))
        m4.metric("Recall", f"{metrics.get('recall', 0):.3f}" if isinstance(metrics.get("recall"), (int, float)) else metrics.get("recall", "n/a"))
        st.caption(f"Test-set delay rate (positives): **{metrics.get('delay_rate_test', 'n/a')}**")

    num_cols = feat_meta["numeric"]
    cat_cols = feat_meta["categorical"]
    weather_keys = [c for c in num_cols if c.startswith("w_")]

    st.markdown("**Scenario inputs**")
    c1, c2, c3 = st.columns(3)
    with c1:
        carrier = st.text_input("Carrier (IATA)", value="AA", help="Two-letter IATA code, uppercase.")
        origin = st.text_input("Origin airport", value="JFK")
        dest = st.text_input("Destination airport", value="LAX")
    with c2:
        dep_hour = st.slider("Scheduled departure (hour)", 0.0, 23.99, 14.5, 0.25)
        month = st.number_input("Month", 1, 12, 1)
        _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = st.selectbox("Day of week", list(range(7)), index=0, format_func=lambda i: _days[i])
    with c3:
        distance = st.number_input("Distance (miles)", 50.0, 6000.0, 2475.0, 25.0)

    weather_vals: dict | None = None
    if weather_keys:
        st.markdown("**Optional weather at origin** (leave blank → same median imputation as training)")
        wc = st.columns(min(len(weather_keys), 4))
        weather_vals = {}
        for i, k in enumerate(weather_keys):
            with wc[i % len(wc)]:
                raw = st.text_input(k.replace("w_", "").replace("_", " "), "", key=f"w_{k}", help="Numeric; empty skips this field.")
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
    proba = float(model.predict_proba(row)[0, 1])
    label, emoji = _risk_tone(proba)

    st.divider()
    left, right = st.columns((1, 1.2), gap="large")
    with left:
        st.markdown(f"### {emoji} {label}")
        st.metric("Estimated **P(delayed)**", f"{proba:.1%}")
    with right:
        st.caption(f"Risk meter · **{proba:.0%}**")
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(
            "This is the probability of the **binary delay class**, not predicted minutes late. "
            "Re-train on your full BTS window (+ weather) for stronger estimates."
        )

    with st.expander("Speaker notes — Live demo", expanded=False):
        st.markdown(SPEAKER_DEMO)


def tab_eda() -> None:
    st.markdown(
        "Figures produced by `scripts/run_eda.py`. Commit `reports/figures/*.png` so **Streamlit Community Cloud** can display them."
    )
    fig_dir = REPORTS_FIGURES
    if not fig_dir.is_dir():
        st.info(f"No figures directory at `{fig_dir}` yet.")
        return

    paths = sorted(fig_dir.glob("*.png"))
    if not paths:
        st.info(f"No `*.png` files in `{fig_dir}` yet.")
        return

    for path in paths:
        key = path.stem
        title = key.replace("_", " ").title()
        caption = EDA_CAPTIONS.get(key, "Discuss what pattern supports your modeling choices.")
        st.subheader(title)
        st.caption(caption)
        st.image(str(path), use_container_width=True)
        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Flight Delay Cast | CIS 2450",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()

    with st.sidebar:
        st.header("Flight Delay Cast")
        st.caption("CIS 2450 · delay risk explorer")
        st.markdown(
            f"**Definition:** *Delayed* means arrival delay **> {DELAY_THRESHOLD_MINUTES} min**."
        )
        st.divider()
        st.markdown("**Presentation flow**")
        st.markdown(
            """
            1. **Overview** — problem & data  
            2. **Delay risk demo** — live inputs  
            3. **EDA snapshots** — evidence behind features
            """
        )

    st.title("Flight Delay Cast")
    st.caption("U.S. domestic flights · BTS + optional origin weather · baseline classifier")

    model, feat_meta, metrics = load_model_bundle()

    tab_ov, tab_live, tab_plots = st.tabs(["Overview", "Delay risk demo", "EDA snapshots"])

    with tab_ov:
        tab_overview()
    with tab_live:
        tab_demo(model, feat_meta, metrics)
    with tab_plots:
        tab_eda()


if __name__ == "__main__":
    main()
