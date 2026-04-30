"""Streamlit dashboard: overview + delay-risk demo + EDA (CIS 2450 presentation)."""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date, time, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import joblib
import pandas as pd
import streamlit as st

from flightdelaycast.config import AIRPORTS_CSV, DELAY_THRESHOLD_MINUTES, MODELS_DIR, REPORTS_DIR, REPORTS_FIGURES
from flightdelaycast.model_features import prediction_dataframe
from flightdelaycast.route_distance import great_circle_miles_between_airports

from manual_weather_numeric import MANUAL_WEATHER_OPTIONS, manual_weather_to_numeric_row
from prediction_explain import approximate_expected_delay_minutes, build_narrative
from weather_policy import (
    SUPPORTED_FORECAST_DAYS,
    WeatherUIMode,
    fetch_live_forecast_weather,
    resolve_weather_ui_mode,
)

# Trained artifacts (train_baseline.py + train_tree_models.py)
MODEL_BUNDLES: list[dict[str, str | Path]] = [
    {
        "id": "logistic",
        "label": "Logistic regression (baseline)",
        "joblib": MODELS_DIR / "baseline_logistic.joblib",
        "features": MODELS_DIR / "baseline_features.json",
        "metrics": MODELS_DIR / "baseline_metrics.json",
    },
    {
        "id": "rf",
        "label": "Random Forest",
        "joblib": MODELS_DIR / "random_forest.joblib",
        "features": MODELS_DIR / "random_forest_features.json",
        "metrics": MODELS_DIR / "random_forest_metrics.json",
    },
    {
        "id": "hgb",
        "label": "Histogram gradient boosting",
        "joblib": MODELS_DIR / "hist_gradient_boosting.joblib",
        "features": MODELS_DIR / "hist_gradient_boosting_features.json",
        "metrics": MODELS_DIR / "hist_gradient_boosting_metrics.json",
    },
]


def _bundle_files_ready(b: dict[str, str | Path]) -> bool:
    return Path(b["joblib"]).is_file() and Path(b["features"]).is_file()


def _bundle_by_id(model_id: str) -> dict[str, str | Path]:
    return next(b for b in MODEL_BUNDLES if b["id"] == model_id)


def _load_json_safe(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except Exception:
        return None


def _default_model_id() -> str:
    for b in MODEL_BUNDLES:
        if _bundle_files_ready(b):
            return str(b["id"])
    return "logistic"


def _classifier_select_label(model_id: str) -> str:
    b = _bundle_by_id(model_id)
    base = str(b["label"])
    return f"{base} ✓" if _bundle_files_ready(b) else f"{base} — not trained yet"


# Short captions for EDA PNGs (edit as your story evolves)
EDA_CAPTIONS: dict[str, str] = {
    "01_class_balance": "Most flights are on time — class imbalance matters for metrics and modeling choices.",
    "02_delay_rate_by_hour": "Delay rate varies by scheduled departure hour; time-of-day is a strong signal.",
    "03_delay_rate_by_carrier": "Carriers differ systematically; airline is a useful categorical feature.",
    "04_correlation_numeric": "Relationships among numeric fields inform feature scaling and redundancy.",
    "05_outlier_boxplots": "IQR-based outlier scan highlights extreme values and informs handling policy.",
}

SPEAKER_OVERVIEW = """
- **Hook:** Delays cost passengers and airlines; estimating risk before departure is a practical decision-support problem.
- **Data:** U.S. BTS domestic flights plus optional daily **origin** and **destination** weather (Meteostat), merged on airport and date.
- **Target:** Binary *delayed* if arrival delay is strictly greater than 15 minutes (matches `DELAY_THRESHOLD_MINUTES` in code).
- **Models:** The live demo can switch among **logistic regression**, **Random Forest**, and **histogram gradient boosting** (whatever `.joblib` files you ship under `models/`).
- **Demo:** Walk through one realistic itinerary, then contrast a peak-hour hub departure vs an off-peak case if time allows.
"""

SPEAKER_DEMO = """
- **Inputs:** Origin/destination + **flight date** + **scheduled departure time**; airline optional; **great-circle distance** is computed from `airports.csv` (run `download_airports.py` if missing).
- **Weather UX:** Inside the forecast horizon, the app stays in **automatic** mode (see `weather_policy.fetch_live_forecast_weather` for the API hook). Farther out, **manual** scenario weather maps to numeric stand-ins (`manual_weather_numeric.py`).
- **Output:** Risk band, **P(delayed)**, optional **heuristic delay minutes**, top factors, and a short recommendation.
- **Caveat:** Illustrative; performance depends on training window and whether weather columns exist in the saved model.
"""

_GFM_TABLE_SEP_LINE = re.compile(r"^\s*\|?[\s\-:|]+\|")


def _gfm_pipe_table_to_html(table_lines: list[str]) -> str:
    """Parse GitHub pipe-table lines into an HTML table with visible white grid (inline styles)."""
    rows: list[list[str]] = []
    for raw in table_lines:
        if _GFM_TABLE_SEP_LINE.match(raw):
            continue
        if not raw.strip().startswith("|"):
            break
        parts = [p.strip() for p in raw.strip().split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if parts:
            rows.append(parts)
    if not rows:
        return ""

    border = "1px solid #ffffff"
    t_style = (
        "border-collapse:collapse;width:100%;margin:0.75rem 0 1.25rem;"
        "border:2px solid #ffffff;background:rgba(8,12,28,0.92);"
        "border-radius:10px;overflow:hidden;"
    )
    chunks: list[str] = [f'<table style="{t_style}">']
    for ri, row in enumerate(rows):
        chunks.append("<tr>")
        for ci, cell in enumerate(row):
            is_head = ri == 0
            tag = "th" if is_head else "td"
            align = "left" if ci == 0 else "right"
            cs = f"border:{border};padding:0.5rem 0.65rem;text-align:{align};color:#f8fafc;"
            if is_head:
                cs += "font-weight:600;background:rgba(6,10,26,0.98);"
            elif ri % 2 == 0:
                cs += "background:rgba(255,255,255,0.06);"
            chunks.append(f'<{tag} style="{cs}">{html.escape(cell)}</{tag}>')
        chunks.append("</tr>")
    chunks.append("</table>")
    return "".join(chunks)


def _eda_summary_segments(md_text: str):
    """Yield ('md', str) and ('html', table_html) — Streamlit's MD renderer drops reliable table borders."""
    lines = md_text.splitlines(keepends=True)
    i = 0
    buf: list[str] = []
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and _GFM_TABLE_SEP_LINE.match(lines[i + 1]):
            if buf:
                yield "md", "".join(buf)
                buf = []
            j = i
            tbl: list[str] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                tbl.append(lines[j])
                j += 1
            yield "html", _gfm_pipe_table_to_html(tbl)
            i = j
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        yield "md", "".join(buf)


def _render_eda_summary_markdown(md_text: str) -> None:
    for kind, segment in _eda_summary_segments(md_text):
        if kind == "md":
            st.markdown(segment)
        else:
            st.markdown(segment, unsafe_allow_html=True)


@st.cache_resource
def load_trained_bundle(model_id: str):
    bundle = next(b for b in MODEL_BUNDLES if b["id"] == model_id)
    if not _bundle_files_ready(bundle):
        return None, None, None, str(bundle["label"])
    model = joblib.load(bundle["joblib"])
    features = json.loads(Path(bundle["features"]).read_text())
    mp = Path(bundle["metrics"])
    metrics = json.loads(mp.read_text()) if mp.is_file() else {}
    return model, features, metrics, str(bundle["label"])


def _risk_tone(p: float) -> tuple[str, str]:
    if p >= 0.55:
        return "High delay risk", "🔴"
    if p >= 0.35:
        return "Moderate delay risk", "🟡"
    return "Lower delay risk", "🟢"


def _inject_style() -> None:
    # Glass / mesh aesthetic inspired by frosted neon gradients (backdrop-filter + noise overlay).
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

          :root {
            --fdc-glass: rgba(18, 22, 45, 0.55);
            --fdc-glass-border: rgba(255, 255, 255, 0.12);
            --fdc-glow-cyan: rgba(0, 240, 255, 0.45);
            --fdc-glow-magenta: rgba(255, 0, 200, 0.35);
            --fdc-text: rgba(245, 248, 255, 0.95);
            --fdc-muted: rgba(200, 210, 230, 0.75);
            --fdc-mesh:
              radial-gradient(ellipse 90% 60% at 12% 25%, rgba(0, 220, 255, 0.38) 0%, transparent 52%),
              radial-gradient(ellipse 70% 50% at 88% 12%, rgba(255, 0, 180, 0.32) 0%, transparent 48%),
              radial-gradient(ellipse 55% 45% at 50% 95%, rgba(120, 60, 255, 0.4) 0%, transparent 50%),
              radial-gradient(ellipse 50% 40% at 75% 60%, rgba(0, 150, 255, 0.2) 0%, transparent 45%),
              linear-gradient(168deg, #050510 0%, #0c0a1a 28%, #0a1228 55%, #10081c 100%);
          }

          /* Never set light text on html/body — Streamlit portals (dropdowns, date picker) attach there and need dark text on white menus. */
          html, body {
            font-family: "DM Sans", "Segoe UI", system-ui, sans-serif;
          }

          [data-testid="stAppViewContainer"],
          .stApp {
            color: rgba(235, 240, 255, 0.94);
          }

          .stApp {
            background: var(--fdc-mesh) !important;
            position: relative;
          }

          /* Light UI surfaces rendered in portals (select menu, calendar, tooltips) */
          [data-baseweb="popover"],
          [data-baseweb="popover"] ul,
          [data-baseweb="popover"] li,
          [data-baseweb="popover"] [role="option"],
          [data-baseweb="menu"],
          [data-baseweb="menu"] li {
            color: #0f172a !important;
            background-color: #f8fafc !important;
          }

          [data-baseweb="popover"] [role="option"]:hover,
          [data-baseweb="menu"] li:hover {
            background-color: #e0f2fe !important;
            color: #0f172a !important;
          }

          [data-baseweb="calendar"] {
            color: #0f172a !important;
            background-color: #f8fafc !important;
          }

          [data-baseweb="calendar"] button {
            color: #0f172a !important;
          }

          /* Subtle grain — avoid mix-blend-mode here: it breaks compositing in Safari/WebKit (blank page). */
          .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            opacity: 0.11;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            background-size: 180px 180px;
          }

          [data-testid="stAppViewContainer"] {
            position: relative;
            z-index: 1;
          }

          /* Hide default Streamlit top chrome (thin header strip above tabs/title) */
          [data-testid="stHeader"],
          [data-testid="stDecoration"] {
            display: none !important;
          }

          section.main {
            position: relative;
            z-index: 2;
            background: transparent !important;
          }

          section.main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 72rem;
          }

          section.main h1 {
            font-weight: 700;
            letter-spacing: -0.03em;
            font-size: clamp(1.85rem, 4vw, 2.35rem);
            background: linear-gradient(105deg, #8ae8ff 0%, #c4b5fd 45%, #f472b6 95%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.35rem;
          }

          section.main h2, section.main h3 {
            color: var(--fdc-text) !important;
            font-weight: 600;
          }

          section.main .stMarkdown p,
          section.main .stMarkdown li {
            color: var(--fdc-muted);
          }

          section.main .stMarkdown strong {
            color: var(--fdc-text) !important;
          }

          section.main a {
            color: #7dd3fc !important;
          }

          section.main code {
            background: rgba(12, 20, 42, 0.95) !important;
            color: #7dd3fc !important;
            padding: 0.15rem 0.45rem;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.14);
          }

          /* Markdown tables — Streamlit puts HTML under [data-testid="stMarkdownContainer"], often WITHOUT .stMarkdown wrapping the table, so target containers directly. */
          section.main [data-testid="stMarkdownContainer"] table,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] table,
          section.main .stMarkdown table,
          .streamlit-expanderContent .stMarkdown table {
            border-collapse: collapse !important;
            border-spacing: 0 !important;
            width: 100%;
            margin: 0.75rem 0 1.25rem;
            font-size: 0.9rem;
            background: rgba(8, 12, 28, 0.88) !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.12) inset;
          }

          section.main [data-testid="stMarkdownContainer"] thead,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] thead,
          section.main .stMarkdown thead,
          .streamlit-expanderContent .stMarkdown thead {
            background: rgba(6, 10, 26, 0.95) !important;
          }

          section.main [data-testid="stMarkdownContainer"] th,
          section.main [data-testid="stMarkdownContainer"] td,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] th,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] td,
          section.main .stMarkdown th,
          section.main .stMarkdown td,
          .streamlit-expanderContent .stMarkdown th,
          .streamlit-expanderContent .stMarkdown td {
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            padding: 0.5rem 0.65rem !important;
            color: rgba(235, 240, 255, 0.96) !important;
          }

          section.main [data-testid="stMarkdownContainer"] th,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] th,
          section.main .stMarkdown th,
          .streamlit-expanderContent .stMarkdown th {
            font-weight: 600;
            color: rgba(210, 225, 255, 0.98) !important;
            text-align: center;
          }

          section.main [data-testid="stMarkdownContainer"] td,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] td,
          section.main .stMarkdown td,
          .streamlit-expanderContent .stMarkdown td {
            text-align: right;
          }

          section.main [data-testid="stMarkdownContainer"] td:first-child,
          section.main [data-testid="stMarkdownContainer"] th:first-child,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] td:first-child,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] th:first-child,
          section.main .stMarkdown td:first-child,
          section.main .stMarkdown th:first-child,
          .streamlit-expanderContent .stMarkdown td:first-child,
          .streamlit-expanderContent .stMarkdown th:first-child {
            text-align: left;
          }

          section.main [data-testid="stMarkdownContainer"] tbody tr:nth-child(even) td,
          .streamlit-expanderContent [data-testid="stMarkdownContainer"] tbody tr:nth-child(even) td,
          section.main .stMarkdown tbody tr:nth-child(even) td,
          .streamlit-expanderContent .stMarkdown tbody tr:nth-child(even) td {
            background: rgba(255, 255, 255, 0.04) !important;
          }

          /* Catch-all: any HTML table rendered inside the main column (Streamlit DOM varies by version) */
          section.main .block-container table {
            border-collapse: collapse !important;
            background: rgba(8, 12, 28, 0.88) !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
          }

          section.main .block-container table th,
          section.main .block-container table td {
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
          }

          section.main .stCaption, section.main [data-testid="stCaptionContainer"] {
            color: var(--fdc-muted) !important;
          }

          /* Sidebar — glass panel */
          [data-testid="stSidebar"] {
            position: relative;
            z-index: 2;
            background: linear-gradient(180deg, rgba(12, 14, 32, 0.72) 0%, rgba(8, 10, 28, 0.58) 100%) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-right: 1px solid var(--fdc-glass-border);
            box-shadow: 4px 0 32px rgba(0, 0, 0, 0.25);
          }

          [data-testid="stSidebar"] > div:first-child {
            background: transparent !important;
          }

          [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: var(--fdc-muted) !important;
          }

          [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--fdc-text) !important;
            font-weight: 700;
          }

          [data-testid="stSidebar"] [data-baseweb="select"] > div,
          [data-testid="stSidebar"] [data-baseweb="input"] input {
            background: rgba(6, 10, 28, 0.88) !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
            color: rgba(245, 248, 255, 0.97) !important;
          }

          [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: rgba(245, 248, 255, 0.95) !important;
          }

          /* Tabs — pill glass bar */
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(12, 16, 40, 0.45);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--fdc-glass-border);
            border-radius: 14px;
            padding: 0.4rem 0.5rem;
            margin-bottom: 1.25rem;
          }

          .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            color: var(--fdc-muted) !important;
            font-weight: 600;
            padding: 0.55rem 1rem !important;
          }

          .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 200, 255, 0.25), rgba(200, 80, 255, 0.2)) !important;
            color: var(--fdc-text) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
          }

          /* Metrics — glass chips */
          [data-testid="stMetricContainer"] {
            background: var(--fdc-glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--fdc-glass-border);
            border-radius: 14px;
            padding: 0.85rem 1rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
          }

          div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700;
            color: #a5f3fc !important;
          }

          div[data-testid="stMetricLabel"] label {
            color: var(--fdc-muted) !important;
          }

          /* Primary buttons */
          .stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #0891b2, #7c3aed) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 20px rgba(8, 145, 178, 0.35);
          }

          .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 28px rgba(124, 58, 237, 0.45);
            border-color: rgba(255, 255, 255, 0.35) !important;
          }

          /* Inputs (control surface = dark; menu pops out separately) */
          .stTextInput input,
          .stSelectbox div[data-baseweb="select"] > div,
          .stDateInput input,
          .stTimeInput input,
          .stNumberInput input {
            border-radius: 10px !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
            background: rgba(8, 12, 30, 0.85) !important;
            color: rgba(245, 248, 255, 0.97) !important;
          }

          /* Time / date pickers — BaseWeb wraps the field in extra divs (fixes solid white box) */
          .stTimeInput > div,
          .stDateInput > div {
            background: transparent !important;
          }

          .stTimeInput [data-baseweb="input"],
          .stTimeInput [data-baseweb="select"],
          .stDateInput [data-baseweb="input"] {
            background: rgba(8, 12, 30, 0.92) !important;
            border-radius: 10px !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
          }

          .stTimeInput [data-baseweb="select"] > div,
          .stTimeInput [data-baseweb="input"] input {
            background: rgba(8, 12, 30, 0.92) !important;
            color: rgba(245, 248, 255, 0.98) !important;
          }

          .stTimeInput [data-baseweb="select"] span,
          .stDateInput [data-baseweb="input"] span {
            color: rgba(245, 248, 255, 0.96) !important;
          }

          .stDateInput [data-baseweb="input"] input {
            background: rgba(8, 12, 30, 0.92) !important;
            color: rgba(245, 248, 255, 0.98) !important;
          }

          .stTextInput input::placeholder,
          .stNumberInput input::placeholder {
            color: rgba(160, 175, 210, 0.75) !important;
          }

          /* Main-area select trigger (not the portal menu) */
          section.main .stSelectbox div[data-baseweb="select"] span {
            color: rgba(245, 248, 255, 0.95) !important;
          }

          /* Secondary buttons */
          .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.1) !important;
            color: rgba(245, 248, 255, 0.96) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
          }

          .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.16) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
          }

          /* Alerts — dark glass + explicit light text (Streamlit defaults are dark gray on our dark bg) */
          div[data-baseweb="notification"],
          .stAlert,
          [data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid var(--fdc-glass-border) !important;
            background: rgba(18, 24, 52, 0.88) !important;
            backdrop-filter: blur(10px);
            color: rgba(225, 232, 255, 0.92) !important;
          }

          .stAlert p,
          .stAlert li,
          .stAlert span,
          .stAlert div[data-testid="stMarkdownContainer"] p,
          [data-testid="stAlert"] p,
          [data-testid="stAlert"] li,
          [data-testid="stAlert"] span {
            color: rgba(225, 232, 255, 0.92) !important;
          }

          /* Expanders — dark glass header (fixes white bar + invisible label text) */
          .streamlit-expander,
          div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 14px !important;
            overflow: hidden;
            background: rgba(10, 14, 32, 0.45) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
          }

          details.streamlit-expander summary,
          .streamlit-expanderHeader,
          div[data-testid="stExpander"] summary,
          div[data-testid="stExpander"] details summary {
            font-weight: 600 !important;
            background: rgba(16, 20, 44, 0.92) !important;
            background-color: rgba(16, 20, 44, 0.92) !important;
            color: rgba(248, 250, 255, 0.98) !important;
            border: none !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
          }

          .streamlit-expanderHeader,
          .streamlit-expanderHeader p,
          .streamlit-expanderHeader span,
          details.streamlit-expander summary,
          details.streamlit-expander summary p,
          details.streamlit-expander summary span,
          div[data-testid="stExpander"] summary,
          div[data-testid="stExpander"] summary *,
          details[data-testid="stExpander"] summary,
          details[data-testid="stExpander"] summary * {
            color: rgba(248, 250, 255, 0.98) !important;
            -webkit-text-fill-color: rgba(248, 250, 255, 0.98) !important;
          }

          .streamlit-expander summary svg,
          div[data-testid="stExpander"] summary svg,
          details[data-testid="stExpander"] summary svg {
            fill: rgba(248, 250, 255, 0.9) !important;
            color: rgba(248, 250, 255, 0.9) !important;
          }

          /* Newer Streamlit: expander trigger can be a button */
          [data-testid="stExpander"] button,
          .streamlit-expander [data-testid="collapsedControl"] {
            background: rgba(16, 20, 44, 0.92) !important;
            color: rgba(248, 250, 255, 0.98) !important;
          }

          .streamlit-expanderContent {
            color: rgba(205, 215, 240, 0.92) !important;
            background: rgba(6, 10, 26, 0.55) !important;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
          }

          .streamlit-expanderContent p,
          .streamlit-expanderContent li,
          .streamlit-expanderContent .stMarkdown p {
            color: rgba(205, 215, 240, 0.92) !important;
          }

          /* Widget labels & helper text */
          label[data-testid="stWidgetLabel"],
          label[data-testid="stWidgetLabel"] p {
            color: rgba(230, 238, 255, 0.92) !important;
          }

          .stTooltipIcon,
          [data-testid="stWidgetLabel"] svg {
            fill: rgba(180, 195, 230, 0.85) !important;
          }

          section.main h4 {
            color: rgba(245, 248, 255, 0.96) !important;
          }

          /* Progress bar */
          .stProgress > div > div {
            background: linear-gradient(90deg, #06b6d4, #a855f7) !important;
            border-radius: 8px;
          }

          .fdc-card {
            border: 1px solid var(--fdc-glass-border);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            background: var(--fdc-glass);
            backdrop-filter: blur(18px) saturate(1.2);
            -webkit-backdrop-filter: blur(18px) saturate(1.2);
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.06);
          }

          .fdc-card strong {
            color: var(--fdc-text);
          }

          /* Dataframes / tables in Methods */
          section.main [data-testid="stDataFrame"], section.main [data-testid="stTable"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--fdc-glass-border);
          }

          /* Images (EDA) — subtle frame */
          section.main [data-testid="stImage"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--fdc-glass-border);
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35);
          }

          /* Dividers */
          hr {
            border-color: rgba(255, 255, 255, 0.08) !important;
          }

          /* Checkbox / radio */
          .stCheckbox label span,
          .stRadio label span,
          .stCheckbox span[data-testid],
          .stRadio span[data-testid] {
            color: rgba(225, 232, 255, 0.9) !important;
          }

          /* Forms */
          [data-testid="stForm"] {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            background: rgba(12, 16, 38, 0.45);
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
        (arrival delay **> {DELAY_THRESHOLD_MINUTES} minutes**), using operational fields and optional origin / destination weather.

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
            - **Weather** — Daily Meteostat summaries at **origin** (`w_*` columns) and/or **destination** (`dw_*`), optional in your processed build.
            """
        )
    with c2:
        st.markdown("**Modeling (current app)**")
        st.markdown(
            """
            - **Preprocessing:** Median imputation, **robust scaling** for numerics (outlier-resistant), one-hot encoding for categoricals, and high-correlation numeric filtering **using the training split only** (no test leakage).
            - **Classifiers:** **Logistic regression** (sparse OHE), **Random Forest**, and **histogram gradient boosting** (dense OHE in training scripts). Use the sidebar selector on the demo tab to compare live **P(delayed)** from each trained pipeline.
            """
        )

    with st.expander("Speaker notes — Overview tab", expanded=False):
        st.markdown(SPEAKER_OVERVIEW)

    st.divider()
    st.caption("Tip for your talk: keep this tab visible while you introduce the problem, then switch to **Delay risk demo** for the live walkthrough.")


def tab_demo(model, feat_meta, metrics, model_label: str, *, model_id: str) -> None:
    if model is None or feat_meta is None:
        if model_id == "rf":
            st.warning(
                "**Random Forest** is not in `models/` yet (need `random_forest.joblib` + `random_forest_features.json`). "
                "After processed flights exist, run:\n\n`python scripts/train_tree_models.py --model rf`\n\n"
                "Or train RF and HGB together: `python scripts/train_tree_models.py`"
            )
        elif model_id == "hgb":
            st.warning(
                "**Histogram gradient boosting** is not in `models/` yet (need `hist_gradient_boosting.joblib` + "
                "`hist_gradient_boosting_features.json`). After processed flights exist, run:\n\n"
                "`python scripts/train_tree_models.py --model hgb`\n\n"
                "Or train both trees: `python scripts/train_tree_models.py`"
            )
        else:
            st.warning(
                "**Logistic regression** artifacts are missing. Train locally, then commit or upload for deployment:\n\n"
                "`python scripts/build_processed.py` → `python scripts/train_baseline.py`\n\n"
                "Expected: `baseline_logistic.joblib` and `baseline_features.json` in `models/`."
            )
        return

    st.success(f"Loaded **{model_label}** pipeline.")

    num_cols = feat_meta["numeric"]
    cat_cols = feat_meta["categorical"]
    origin_wx = [c for c in num_cols if c.startswith("w_") and not c.startswith("dw_")]
    dest_wx = [c for c in num_cols if c.startswith("dw_")]
    wx_numeric_cols = origin_wx + dest_wx

    today = date.today()
    max_flight_date = today + timedelta(days=365)

    st.markdown("### Trip")
    flight_date = st.date_input(
        "Flight date",
        value=today,
        min_value=today,
        max_value=max_flight_date,
        help=f"Past dates are disabled. Automatic forecast mode applies through the next {SUPPORTED_FORECAST_DAYS} days.",
    )

    wx_mode = resolve_weather_ui_mode(flight_date, reference_date=today)
    manual_dep_choice: str | None = None
    manual_dest_choice: str | None = None

    if wx_numeric_cols:
        if wx_mode is WeatherUIMode.AUTOMATIC_FORECAST:
            st.info(f"Live weather forecast will be fetched automatically for this date (within {SUPPORTED_FORECAST_DAYS}-day window).")
        else:
            st.warning(
                "Weather forecast data is unavailable this far in advance, so please choose expected weather conditions."
            )
            _wx_placeholder = "— choose —"
            manual_dep_choice = st.selectbox(
                "Departure weather",
                options=[_wx_placeholder, *MANUAL_WEATHER_OPTIONS],
                help="Mapped to numeric stand-ins for origin `w_*` columns (see `manual_weather_numeric.py`).",
            )
            manual_dest_choice = st.selectbox(
                "Destination weather",
                options=[_wx_placeholder, *MANUAL_WEATHER_OPTIONS],
                help="Mapped to numeric stand-ins for destination `dw_*` columns.",
            )
    else:
        st.caption("This model artifact has **no weather columns**; weather mode UI is skipped.")

    st.markdown("### Itinerary")
    with st.form("delay_prediction_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            origin_in = st.text_input("Origin airport (IATA)", value="JFK", help="Three-letter U.S. airport code.")
            dest_in = st.text_input("Destination airport (IATA)", value="LAX", help="Must differ from origin.")
        with c2:
            dep_time = st.time_input(
                "Scheduled departure time",
                value=time(14, 30),
                help="Converted to fractional hour for the model (scheduled, not actual).",
            )
            airline_in = st.text_input(
                "Airline (optional, IATA)",
                value="",
                placeholder="e.g. AA",
                help="Leave blank if unknown; encoded as UNK for the categorical pipeline.",
            )
        submitted = st.form_submit_button("Run prediction")

    if not submitted:
        st.info("Fill in **origin**, **destination**, and **scheduled departure time**, then click **Run prediction**.")
        return

    errors: list[str] = []
    if flight_date < today or flight_date > max_flight_date:
        errors.append("Flight date must be between today and one year from today.")

    o = origin_in.strip().upper()
    d = dest_in.strip().upper()
    if not o or not d:
        errors.append("Origin and destination are required.")
    if o == d:
        errors.append("Origin and destination cannot be the same.")

    carrier = airline_in.strip().upper() or "UNK"

    dep_hour = dep_time.hour + dep_time.minute / 60.0 + dep_time.second / 3600.0
    month = int(flight_date.month)
    dow = int(flight_date.weekday())

    distance_mi = great_circle_miles_between_airports(o, d, AIRPORTS_CSV)
    if distance_mi is None:
        errors.append(
            "Could not compute distance (unknown IATA codes or missing `data/raw/airports.csv`). "
            "Run `python scripts/download_airports.py`."
        )

    weather_vals: dict[str, float] | None = None
    if wx_numeric_cols:
        if wx_mode is WeatherUIMode.AUTOMATIC_FORECAST:
            fetched = fetch_live_forecast_weather(
                o,
                d,
                flight_date,
                required_numeric_keys=wx_numeric_cols,
            )
            weather_vals = {k: float(v) for k, v in (fetched or {}).items()} if fetched else None
            if weather_vals:
                wx_display = {k: f"{v:.1f}" for k, v in weather_vals.items()}
                st.success(f"Live forecast fetched: {wx_display}")
            else:
                st.warning("Could not fetch live forecast — weather will be median-imputed.")
        else:
            _ph = "— choose —"
            if manual_dep_choice in (None, _ph) or manual_dest_choice in (None, _ph):
                errors.append("Select both departure and destination weather for manual mode.")
            else:
                weather_vals = manual_weather_to_numeric_row(
                    str(manual_dep_choice),
                    str(manual_dest_choice),
                    origin_keys=origin_wx,
                    dest_keys=dest_wx,
                )

    if errors:
        for e in errors:
            st.error(e)
        return

    assert distance_mi is not None
    row = prediction_dataframe(
        num_cols=num_cols,
        cat_cols=cat_cols,
        carrier=carrier,
        origin=o,
        dest=d,
        dep_hour=float(dep_hour),
        month=month,
        day_of_week=dow,
        distance=float(distance_mi),
        weather=weather_vals,
    )
    proba = float(model.predict_proba(row)[0, 1])
    label, emoji = _risk_tone(proba)
    approx_min = approximate_expected_delay_minutes(proba)
    narrative = build_narrative(
        model,
        model_id,
        row,
        dep_hour=float(dep_hour),
        month=month,
        distance_mi=float(distance_mi),
        origin=o,
        dest=d,
        carrier=carrier,
        flight_date=flight_date,
        delay_probability=proba,
        manual_dep_wx=manual_dep_choice if wx_mode is WeatherUIMode.MANUAL_SCENARIO else None,
        manual_dest_wx=manual_dest_choice if wx_mode is WeatherUIMode.MANUAL_SCENARIO else None,
    )

    st.divider()
    st.markdown("### Prediction")
    left, right = st.columns((1, 1.15), gap="large")
    with left:
        st.markdown(f"#### {emoji} {label}")
        st.metric("Probability of delay ( >15 min )", f"{proba:.1%}")
        if approx_min is not None:
            st.metric("Approx. delay if late (heuristic)", f"~{approx_min:.0f} min")
            st.caption("Heuristic from P(delayed), not a trained regression on minutes.")
        else:
            st.caption("Estimated delay minutes: *not shown* for very low delay probability.")
    with right:
        st.caption(f"Risk meter · **{proba:.0%}**")
        st.progress(min(max(proba, 0.0), 1.0))
        st.markdown("**Top contributing factors**")
        for line in narrative.factors:
            st.markdown(f"- {line}")
        st.markdown("**Recommendation**")
        st.markdown(narrative.recommendation)
        st.caption(
            "P(delayed) is from the trained classifier on historical BTS-style features, not live ATC data."
        )

    with st.expander("Speaker notes — Live demo", expanded=False):
        st.markdown(SPEAKER_DEMO)


def tab_eda() -> None:
    st.markdown(
        "Figures produced by `scripts/run_eda.py`. Commit `reports/figures/*.png` so **Streamlit Community Cloud** can display them."
    )
    summary_path = REPORTS_DIR / "eda_summary.md"
    if summary_path.is_file():
        with st.expander("EDA findings summary (rubric-aligned)", expanded=True):
            _render_eda_summary_markdown(summary_path.read_text(encoding="utf-8"))
    else:
        st.info(
            "No EDA findings summary found yet. Run `python scripts/run_eda.py` to generate `reports/eda_summary.md`."
        )

    st.divider()
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


def tab_methods() -> None:
    st.markdown("## Difficulty Concepts")
    st.caption("Three advanced techniques applied in this project, with justification, implementation details, and measured results.")

    # ── 1. Ensemble Models ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("1. Ensemble Models")
    st.markdown(
        """
        **Why:** A single decision tree overfits because it memorises the training split.
        Ensemble methods address this in two complementary ways:

        - **Random Forest** trains many trees on random bootstrap samples with random feature subsets,
          then averages their votes. This reduces *variance* without meaningfully increasing bias,
          making it robust to noisy features like flight schedules.
        - **Histogram Gradient Boosting** builds trees *sequentially* — each tree corrects the
          residuals of the previous one. The learning rate (shrinkage) controls how aggressively
          each tree is applied, balancing bias reduction against overfitting.

        Using both lets us compare a variance-reducing ensemble (RF) against a bias-reducing one (HGB),
        and measure whether either justifies the added complexity over logistic regression.
        """
    )

    comparison = _load_json_safe(MODELS_DIR / "tree_models_comparison.json")
    baseline_m = _load_json_safe(MODELS_DIR / "baseline_metrics.json")

    if comparison or baseline_m:
        rows = []
        if baseline_m:
            rows.append({
                "Model": "Logistic Regression (baseline)",
                "F1": f"{baseline_m.get('f1', 0):.3f}",
                "Accuracy": f"{baseline_m.get('accuracy', 0):.3f}",
                "Precision": f"{baseline_m.get('precision', 0):.3f}",
                "Recall": f"{baseline_m.get('recall', 0):.3f}",
            })
        for key, m in (comparison or {}).items():
            rows.append({
                "Model": key.replace("_", " ").title(),
                "F1": f"{m.get('f1', 0):.3f}",
                "Accuracy": f"{m.get('accuracy', 0):.3f}",
                "Precision": f"{m.get('precision', 0):.3f}",
                "Recall": f"{m.get('recall', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Both ensemble models outperform logistic regression on F1. HGB achieves the best "
            "balance of precision and recall; Random Forest is competitive with more interpretable "
            "feature importances."
        )
    else:
        st.info("Run `python scripts/train_tree_models.py` to generate model comparison data.")

    st.markdown(
        "**Where in code:** `scripts/train_tree_models.py` — `_train_rf()` and `_train_hgb()`; "
        "`sklearn.ensemble.RandomForestClassifier` and `HistGradientBoostingClassifier`."
    )

    # ── 2. Feature Importance & Selection ───────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Feature Importance & Feature Selection")
    st.markdown(
        """
        **Why:** Not all features contribute equally. After training Random Forest, each feature
        receives a *mean decrease in impurity* importance score. Because one-hot encoding expands
        categorical columns into many binary dummies, importance scores are *aggregated back* to
        the original column level (e.g., all `ORIGIN_*` dummies are summed into one `ORIGIN` score).

        This serves two purposes:
        1. **Confirm EDA signals** — verifies that departure hour, carrier, and distance really are
           the strongest predictors, as the visualisations suggested.
        2. **Narrow the feature space** — retraining on only the top-k most important original
           columns tests whether low-importance features are truly redundant. If the reduced model
           preserves most of the F1, it confirms the selection and supports a leaner deployment model.
        """
    )

    sel = _load_json_safe(MODELS_DIR / "random_forest_feature_selection.json")
    rf_m = _load_json_safe(MODELS_DIR / "random_forest_metrics.json")

    if sel:
        top_k = sel.get("top_k", 6)
        all_feats = sel.get("all_feature_importances", sel.get("top_features", []))
        reduced = sel.get("reduced_metrics", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**All features ranked by aggregated importance**")
            df_imp = pd.DataFrame(all_feats)
            if not df_imp.empty:
                df_imp["importance"] = df_imp["importance"].map(lambda x: f"{x:.4f}")
                st.dataframe(df_imp, hide_index=True, use_container_width=True)
        with col2:
            st.markdown(f"**Full model vs. top-{top_k} features only**")
            if rf_m and reduced:
                comp_rows = [
                    {"Model": "All features", "F1": f"{rf_m['f1']:.3f}", "Accuracy": f"{rf_m['accuracy']:.3f}"},
                    {"Model": f"Top-{top_k} features", "F1": f"{reduced['f1']:.3f}", "Accuracy": f"{reduced['accuracy']:.3f}"},
                ]
                st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
                delta = abs(rf_m["f1"] - reduced["f1"])
                st.caption(
                    f"Keeping only the {top_k} most important original features drops F1 by "
                    f"just {delta:.3f}, confirming the remaining features contribute marginally."
                )
    else:
        st.info("Re-run `python scripts/train_tree_models.py` to generate feature selection results.")

    st.markdown(
        "**Where in code:** `scripts/train_tree_models.py` — `_col_importances()` aggregates "
        "post-OHE importances; `_feature_selection_comparison()` retrains and compares. "
        "Results saved to `models/random_forest_feature_selection.json`."
    )

    # ── 3. Hyperparameter Tuning ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Hyperparameter Tuning (RandomizedSearchCV)")
    st.markdown(
        """
        **Why:** Default hyperparameters are a starting point, not a solution. Key knobs like
        tree depth, learning rate, and minimum samples per leaf create a bias-variance tradeoff
        that depends on dataset size and class balance.

        `RandomizedSearchCV` samples randomly from a defined parameter space and uses
        *k*-fold cross-validation (k=3) to estimate out-of-sample F1 for each candidate.
        This is preferred over grid search for large spaces (fewer evaluations needed) and
        avoids overfitting to the test set because the test set is never used during search.
        Scoring on **F1** — not accuracy — is critical here because the target is imbalanced
        (~23 % delayed).
        """
    )

    log_params = _load_json_safe(MODELS_DIR / "baseline_best_params.json")
    rf_params = _load_json_safe(MODELS_DIR / "random_forest_best_params.json")
    hgb_params = _load_json_safe(MODELS_DIR / "hist_gradient_boosting_best_params.json")

    if log_params or rf_params or hgb_params:
        if log_params:
            st.markdown("**Logistic regression (baseline) — best params (C regularization, cv=3, scoring=F1)**")
            st.json(log_params)
        if rf_params:
            st.markdown("**Random Forest — best params found by RandomizedSearchCV (cv=3, scoring=F1)**")
            st.json(rf_params)
        if hgb_params:
            st.markdown("**Histogram Gradient Boosting — best params**")
            st.json(hgb_params)
    else:
        st.info(
            "Tuning results not yet generated. By default, training runs **RandomizedSearchCV** "
            "(F1, cv=3). Run:\n\n"
            "`python scripts/train_baseline.py`\n\n"
            "`python scripts/train_tree_models.py`\n\n"
            "Use `--no-tune` on either script only for a faster dev run. "
            "Best params are written to `models/baseline_best_params.json` and "
            "`models/*_best_params.json` for trees."
        )

    with st.expander("Parameter spaces searched (RF_PARAM_DIST / HGB_PARAM_DIST)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Random Forest**")
            st.json({
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, 24],
                "min_samples_leaf": [5, 10, 20],
                "max_features": ["sqrt", "log2"],
            })
        with c2:
            st.markdown("**Histogram Gradient Boosting**")
            st.json({
                "learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2],
                "max_depth": [5, 7, 9],
                "max_leaf_nodes": [31, 48, 63],
                "l2_regularization": [0.0, 0.1, 0.5],
                "min_samples_leaf": [10, 20, 30],
            })

    st.markdown(
        "**Where in code:** `scripts/train_baseline.py` (`_tune_logistic`) and "
        "`scripts/train_tree_models.py` (`_tune_pipe`) wrap "
        "`sklearn.model_selection.RandomizedSearchCV` (on by default; pass `--no-tune` to skip)."
    )

    # ── Conclusion ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Conclusion")

    baseline_f1 = (baseline_m or {}).get("f1")
    rf_f1 = (rf_m or {}).get("f1")
    hgb_m = _load_json_safe(MODELS_DIR / "hist_gradient_boosting_metrics.json")
    hgb_f1 = (hgb_m or {}).get("f1")

    parts = []
    if baseline_f1 and rf_f1 and hgb_f1:
        parts.append(
            f"Logistic regression (F1={baseline_f1:.3f}) establishes a stable baseline. "
            f"Both ensemble models exceed it: Random Forest (F1={rf_f1:.3f}) reduces variance "
            f"through bagging and feature randomisation, while Histogram Gradient Boosting "
            f"(F1={hgb_f1:.3f}) achieves the highest score by sequentially correcting residuals."
        )
    if sel and rf_m:
        top_k = sel.get("top_k", 6)
        top_names = ", ".join(f['feature'] for f in sel["top_features"][:3])
        full_f1 = rf_m["f1"]
        red_f1 = sel["reduced_metrics"]["f1"]
        delta = abs(full_f1 - red_f1)
        parts.append(
            f"Feature importance analysis (aggregated across OHE dummies) confirmed that "
            f"{top_k} original features — led by {top_names} — drive most of the signal. "
            f"Retraining on only those {top_k} features preserved F1 within {delta:.3f} of "
            f"the full-feature model, validating the importance rankings from EDA."
        )
    if log_params or rf_params or hgb_params:
        parts.append(
            "RandomizedSearchCV identified hyperparameters that refine the bias-variance "
            "tradeoff beyond the initial defaults, with tuning guided by 3-fold cross-validated F1."
        )
    else:
        parts.append(
            "Training scripts run RandomizedSearchCV by default (3-fold CV on F1). "
            "Re-run `train_baseline.py` and `train_tree_models.py` to materialize best-parameter JSON."
        )

    st.markdown(" ".join(parts))


def main() -> None:
    st.set_page_config(
        page_title="Flight Delay Cast | CIS 2450",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()

    model_id_key = "fdc_model_id"
    all_ids = [str(b["id"]) for b in MODEL_BUNDLES]
    ready_n = sum(1 for b in MODEL_BUNDLES if _bundle_files_ready(b))

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
        st.divider()
        st.markdown("**Delay risk demo**")
        if model_id_key in st.session_state and st.session_state[model_id_key] not in all_ids:
            del st.session_state[model_id_key]
        default_ix = all_ids.index(_default_model_id())
        chosen_id = st.selectbox(
            "Classifier",
            options=all_ids,
            index=default_ix,
            format_func=_classifier_select_label,
            key=model_id_key,
        )
        st.caption(f"Trained bundles in `models/`: **{ready_n}/{len(MODEL_BUNDLES)}**")
        if ready_n < len(MODEL_BUNDLES):
            st.caption("Add trees: `python scripts/train_tree_models.py` (needs processed flight CSV).")

    st.title("Flight Delay Cast")
    st.caption("U.S. domestic flights · BTS + optional origin/destination weather · pick a classifier in the sidebar")

    tab_ov, tab_live, tab_plots, tab_meth = st.tabs(["Overview", "Delay risk demo", "EDA snapshots", "Methods"])

    model, feat_meta, metrics, label = load_trained_bundle(chosen_id)

    with tab_ov:
        tab_overview()
    with tab_live:
        tab_demo(model, feat_meta, metrics, label, model_id=chosen_id)
    with tab_plots:
        tab_eda()
    with tab_meth:
        tab_methods()


if __name__ == "__main__":
    main()
