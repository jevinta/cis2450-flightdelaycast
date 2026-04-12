"""Paths and project constants."""

from pathlib import Path

# Repo root = parent of `src/`
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# Proposal: binary target — 1 if arrival delay strictly exceeds this (minutes)
DELAY_THRESHOLD_MINUTES = 15
