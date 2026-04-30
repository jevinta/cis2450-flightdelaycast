"""BTS flight data cleaning helpers."""

from __future__ import annotations

import pandas as pd

from flightdelaycast.config import DELAY_THRESHOLD_MINUTES


def add_delay_target(df: pd.DataFrame, arr_delay_col: str = "ARR_DELAY") -> pd.DataFrame:
    """Create `is_delayed` from arrival delay minutes."""
    out = df.copy()
    delay = pd.to_numeric(out[arr_delay_col], errors="coerce")
    out["is_delayed"] = (delay > DELAY_THRESHOLD_MINUTES).astype("Int64")
    return out


def basic_flight_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by the first available time column, if present."""
    out = df.copy()
    for col in ("FL_DATE", "DEP_TIME", "ARR_TIME"):
        if col in out.columns:
            out = out.sort_values(col, kind="mergesort")
            break
    return out
