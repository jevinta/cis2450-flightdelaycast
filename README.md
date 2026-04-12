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