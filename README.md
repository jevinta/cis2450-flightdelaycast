# Flight Delay Cast (CIS 2450)

Predict whether a U.S. domestic flight will be **delayed** (arrival delay **> 15 minutes**) using BTS flight data and optional daily weather at origin (`w_*`) and destination (`dw_*`) from Meteostat, with an interactive Streamlit dashboard.

## Project layout

```
data/
  raw/bts/          BTS CSV downloads (gitignored)
  raw/airports.csv  IATA airport coordinates
  processed/        Cleaned + merged tables (.csv.gz, gitignored except .gitkeep)
models/             Trained .joblib files + metrics/params JSON
reports/figures/    EDA plots (.png, committed)
scripts/            Data download, processing, EDA, and training scripts
src/flightdelaycast/  Shared Python package
dashboard/          Streamlit app
notebooks/          Jupyter notebooks (gitignored except .gitkeep)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

## Training pipeline

### 1. Download raw data

```bash
python scripts/download_bts.py --year 2023 --months 1 2 3   # BTS flight CSVs
python scripts/download_airports.py                           # airports.csv (IATA + coords)
```

### 2. Build processed flights

```bash
# Flights only (fastest):
python scripts/build_processed.py

# With origin weather:
python scripts/build_processed.py --weather

# With origin + destination weather (slow — Meteostat API calls):
python scripts/build_processed.py --weather --weather-dest --weather-max-pairs 500
```

Outputs `data/processed/flights_wrangled.csv.gz` (and weather caches if requested).

### 3. Run EDA

```bash
python scripts/run_eda.py
```

Writes `reports/eda_summary.md` and `reports/figures/*.png`.

### 4. Train models

```bash
python scripts/train_baseline.py        # Logistic regression
python scripts/train_tree_models.py     # Random Forest + Histogram Gradient Boosting
```

Both scripts run **RandomizedSearchCV** (cv=3, scoring=F1) by default. Pass `--no-tune` for a faster run without hyperparameter search. Individual tree models: `--model rf` or `--model hgb`.

Trained artifacts land in `models/`:

| File | Description |
|---|---|
| `baseline_logistic.joblib` | Logistic regression pipeline |
| `random_forest.joblib` | Random Forest pipeline |
| `hist_gradient_boosting.joblib` | HGB pipeline |
| `*_features.json` | Feature column lists (numeric + categorical) |
| `*_metrics.json` | Test-set metrics |
| `*_best_params.json` | Best hyperparameters from RandomizedSearchCV |
| `*_threshold.json` | Tuned decision threshold (RF and HGB) |
| `random_forest_feature_selection.json` | Aggregated importances + reduced-model comparison |
| `tree_models_comparison.json` | Side-by-side RF vs. HGB metrics |

## Model results

Trained on 120 000 flights, tested on 30 000 (~20.6 % delayed).

| Model | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.397 | 0.608 | 0.291 | 0.627 |
| Random Forest | 0.436 | 0.708 | 0.362 | 0.548 |
| Histogram gradient boosting | **0.453** | 0.708 | 0.369 | 0.587 |

F1 is the primary metric because the target is imbalanced (roughly 4:1 on-time vs. delayed).

## Dashboard

```bash
streamlit run dashboard/app.py
```

Four tabs:

- **Overview** — problem framing, data sources, preprocessing summary
- **Delay risk demo** — enter origin, destination, date, and departure time; pick a classifier in the sidebar; get P(delayed), risk band, heuristic delay minutes, top factors, and a recommendation
- **EDA snapshots** — class balance, delay rate by hour and carrier, feature correlations, outlier boxplots
- **Methods** — ensemble models, feature importance & selection, hyperparameter tuning — all backed by the live JSON artifacts in `models/`

### Weather in the demo

- **Within 7 days:** live forecast fetched automatically from [Open-Meteo](https://open-meteo.com) (free, no API key).
- **Beyond 7 days:** manual scenario picker ("Clear", "Rainy", etc.) mapped to numeric stand-ins (`dashboard/manual_weather_numeric.py`).
- **No weather columns in model:** weather UI is skipped; the model runs on operational features only.

### Sidebar classifier selector

The sidebar shows all three classifiers. Those with `.joblib` + `_features.json` in `models/` are selectable; missing ones display a training hint.

## Shared package (`src/flightdelaycast`)

| Module | Purpose |
|---|---|
| `config.py` | Paths and `DELAY_THRESHOLD_MINUTES` (15) |
| `cleaning.py` | BTS column renaming and type coercions |
| `wrangle.py` | Merge flights with weather, derive features |
| `model_features.py` | Feature column selection, high-correlation drop, `prediction_dataframe` |
| `meteostat_daily.py` | Fetch daily weather summaries from Meteostat |
| `weather_origin.py` | Build/cache origin weather table |
| `weather_destination.py` | Build/cache destination weather table |
| `route_distance.py` | Great-circle distance between IATA airports |

## Deploying to Streamlit Community Cloud

1. Push this repo (with `models/` artifacts and `reports/figures/`) to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app → pick repo → **Main file:** `dashboard/app.py` → **Advanced** → Python 3.10+ → **Requirements file:** `requirements-app.txt`.
3. Wait for build; share the public URL.

`requirements-app.txt` is a slim install (no Meteostat, no Jupyter) suitable for cloud hosts. For local dev use `requirements.txt`.

**Alternatives:** Render or Railway — start command `streamlit run dashboard/app.py`, requirements file `requirements-app.txt`.
