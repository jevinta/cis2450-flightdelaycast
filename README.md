# Flight Delay Cast (CIS 2450)

Predict whether a U.S. domestic flight will be **delayed** (arrival delay **> 15 minutes**) using BTS flight data and NOAA weather, with a small Streamlit dashboard.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

## Layout

- `data/raw` — downloaded CSVs (gitignored)
- `data/processed` — cleaned / merged tables
- `notebooks` — EDA
- `src/flightdelaycast` — shared Python package (cleaning, config)
- `dashboard` — Streamlit app (`streamlit run dashboard/app.py`)

## Deploying the dashboard (presentation)

Your UI is **Streamlit** (`dashboard/app.py`). Typical flow:

1. **Push to GitHub** (same layout as this repo: `src/flightdelaycast`, `dashboard/app.py`, `pyproject.toml`).
2. **Include model artifacts** for the live demo (or the app falls back to instructions only):
   - `models/baseline_logistic.joblib`
   - `models/baseline_features.json` (generated with `scripts/train_baseline.py`)
   - Optional: `models/baseline_metrics.json` for on-screen scores
3. **Include EDA images** if you want the second tab to show charts: commit `reports/figures/*.png` (they are not gitignored).
4. **[Streamlit Community Cloud](https://streamlit.io/cloud)** (free): sign in with GitHub → New app → pick repo → **Main file** `dashboard/app.py` → **Advanced** → Python version 3.10+ → **Requirements file** `requirements-app.txt` (slimmer than full dev `requirements.txt`).
5. Wait for the build; share the public URL for your presentation.

**Alternatives:** [Render](https://render.com) or [Railway](https://railway.app) can run Streamlit with a `streamlit run dashboard/app.py` start command and the same requirements file.

**Local preview (what the grader sees on your laptop):**

```bash
pip install -r requirements-app.txt
streamlit run dashboard/app.py
```