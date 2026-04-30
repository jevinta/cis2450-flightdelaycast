# Flight Delay Cast

A flight delay predictor built for CIS 2450. Given a U.S. domestic itinerary — origin, destination, airline, and scheduled departure time — it estimates the probability that the flight arrives more than 15 minutes late. Three classifiers are available: logistic regression as a baseline, Random Forest, and Histogram Gradient Boosting.

The project combines Bureau of Transportation Statistics (BTS) flight records with daily weather from Meteostat at both the departure and arrival airport. The interactive Streamlit dashboard lets you run predictions against any of the trained models, explore the EDA findings, and see a breakdown of the advanced methods used.

---

## Data

**Source:** BTS domestic flight records + Meteostat daily weather summaries.

We trained on 150,000 U.S. domestic flights spanning multiple months and seasons. The target is binary: a flight is "delayed" if its actual arrival is more than 15 minutes late, which matches the industry-standard reportable threshold. About 1 in 5 flights in the dataset is delayed (20.6%), so we optimize for F1 rather than accuracy — a model that always predicts "on time" would hit 80% accuracy but catch zero delays.

Key patterns in the data:
- Early morning flights (around 6 AM) have delay rates near 10–12%. Risk climbs through the day and peaks around 7 PM at roughly 29%, as scheduling disruptions cascade across aircraft rotations.
- Airline identity is one of the strongest predictors — about 15 percentage points separate the most and least punctual carriers in the dataset.
- Great-circle distance, departure hour, month, and day of week are the most important operational features. Weather (especially precipitation and temperature at origin) adds signal for weather-sensitive airports.

---

## Results

| Model | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.397 | 0.608 | 0.291 | 0.627 |
| Random Forest | 0.436 | 0.708 | 0.362 | 0.548 |
| Histogram gradient boosting | **0.453** | 0.708 | 0.369 | 0.587 |

The ensemble models both outperform logistic regression on F1. HGB edges out Random Forest and has a better precision/recall balance at its tuned threshold (0.55). RF is competitive and provides more interpretable feature importances.

Feature importance analysis (aggregated across one-hot encoded dummies back to original column level) confirmed that departure hour, carrier, origin airport, and distance drive most of the signal — consistent with the EDA findings. Retraining Random Forest on only the top features preserved F1 within 0.003 of the full model.

All three models use **RandomizedSearchCV** (3-fold CV, scoring=F1) for hyperparameter tuning by default. RF and HGB also get a tuned decision threshold to better balance precision and recall on the imbalanced target.

---

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

---

## Running the training pipeline

### 1. Get the raw data

```bash
python scripts/download_bts.py --year 2023 --months 1 2 3
python scripts/download_airports.py
```

BTS CSVs go into `data/raw/bts/`. `airports.csv` (IATA codes + coordinates for distance calculations) goes into `data/raw/`.

### 2. Build the processed dataset

```bash
# Fastest — no weather:
python scripts/build_processed.py

# With origin weather:
python scripts/build_processed.py --weather

# With origin + destination weather (slower, Meteostat API calls):
python scripts/build_processed.py --weather --weather-dest --weather-max-pairs 500
```

Output: `data/processed/flights_wrangled.csv.gz`. Weather caches are written alongside it if you use those flags.

### 3. Explore the data

```bash
python scripts/run_eda.py
```

Writes `reports/eda_summary.md` and five figures under `reports/figures/`.

### 4. Train

```bash
python scripts/train_baseline.py        # logistic regression
python scripts/train_tree_models.py     # Random Forest + HGB (or --model rf / --model hgb)
```

Both scripts tune hyperparameters by default. Pass `--no-tune` to skip RandomizedSearchCV for faster iteration.

Artifacts written to `models/`: `.joblib` pipelines, `_features.json` (column lists), `_metrics.json`, `_best_params.json`, and `_threshold.json` for the tree models. The dashboard reads these at runtime.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Pick a classifier in the sidebar, then use the four tabs:

- **Overview** — problem framing, data sources, and a summary of the preprocessing approach
- **Delay risk demo** — enter origin, destination, flight date, and departure time; get P(delayed), a risk band, heuristic expected delay, top contributing factors, and a recommendation
- **EDA snapshots** — the five EDA figures with plain-language captions, plus the full EDA findings summary
- **Methods** — covers the three advanced concepts: ensemble models, feature importance & selection, and hyperparameter tuning, each backed by the live metrics from `models/`

For flights within 7 days, the demo fetches a live weather forecast from Open-Meteo (free, no API key needed). Beyond that window, a manual scenario picker maps plain-language conditions ("Clear", "Rainy", etc.) to numeric stand-ins. If the loaded model has no weather columns, the weather UI is skipped entirely.

For local preview with the slim dependency set:

```bash
pip install -r requirements-app.txt
streamlit run dashboard/app.py
```

---

## Project structure

```
data/
  raw/bts/              BTS CSV downloads (gitignored)
  raw/airports.csv      IATA airport lookup with lat/lon
  processed/            Cleaned and merged tables (gitignored)
models/                 Trained .joblib files and JSON metrics/params
reports/
  eda_summary.md        Written by run_eda.py
  figures/              EDA plots (committed)
scripts/
  download_bts.py       Fetch BTS flight CSVs
  download_airports.py  Fetch airports.csv
  build_processed.py    Merge flights + weather into training data
  run_eda.py            Generate EDA summary and figures
  train_baseline.py     Train and tune logistic regression
  train_tree_models.py  Train and tune RF and HGB
src/flightdelaycast/    Shared package (cleaning, wrangling, feature engineering, weather fetching)
dashboard/
  app.py                Streamlit app entry point
  weather_policy.py     Open-Meteo forecast fetching and weather UI mode logic
  manual_weather_numeric.py  Maps scenario labels to numeric weather stand-ins
  prediction_explain.py     Narrative and factor generation for the demo output
  style.css             Custom dashboard styles
```
