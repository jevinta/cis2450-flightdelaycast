"""Paths and project constants."""

from pathlib import Path

# Repo root = parent of `src/`
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_RAW_BTS = DATA_RAW / "bts"
AIRPORTS_CSV = DATA_RAW / "airports.csv"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_FIGURES = REPORTS_DIR / "figures"
MODELS_DIR = REPO_ROOT / "models"
PROCESSED_FLIGHTS = DATA_PROCESSED / "flights_wrangled.csv.gz"
WEATHER_CACHE = DATA_PROCESSED / "weather_origin_daily.csv.gz"
WEATHER_DEST_CACHE = DATA_PROCESSED / "weather_destination_daily.csv.gz"

# Proposal: binary target — 1 if arrival delay strictly exceeds this (minutes)
DELAY_THRESHOLD_MINUTES = 15
