"""Data cleaning and preprocessing for BTS flight tables (extend as columns are finalized)."""

from __future__ import annotations

import pandas as pd

from flightdelaycast.config import DELAY_THRESHOLD_MINUTES


def add_delay_target(df: pd.DataFrame, arr_delay_col: str = "ARR_DELAY") -> pd.DataFrame:
    """Add binary target: 1 if arrival delay > threshold minutes, else 0.

    Drops or masks rows with missing arrival delay when building the target
    (adjust if your team keeps cancelled/diverted flights differently).
    """
    out = df.copy()
    delay = pd.to_numeric(out[arr_delay_col], errors="coerce")
    out["is_delayed"] = (delay > DELAY_THRESHOLD_MINUTES).astype("Int64")
    return out


def basic_flight_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Example sanity pass: sort by time columns if present (customize to your schema)."""
    out = df.copy()
    for col in ("FL_DATE", "DEP_TIME", "ARR_TIME"):
        if col in out.columns:
            out = out.sort_values(col, kind="mergesort")
            break
    return out
