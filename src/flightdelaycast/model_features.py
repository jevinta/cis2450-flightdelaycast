"""Feature column lists shared by training scripts and the Streamlit app."""

from __future__ import annotations

import pandas as pd

WEATHER_NUMERIC = ("w_tmax", "w_tmin", "w_prcp", "w_wspd")


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = ["dep_hour", "month", "day_of_week", "DISTANCE"]
    for c in WEATHER_NUMERIC:
        if c in df.columns and df[c].notna().any():
            numeric.append(c)
    categorical = ["OP_CARRIER", "ORIGIN", "DEST"]
    return numeric, categorical


def prediction_dataframe(
    *,
    num_cols: list[str],
    cat_cols: list[str],
    carrier: str,
    origin: str,
    dest: str,
    dep_hour: float,
    month: int,
    day_of_week: int,
    distance: float,
    weather: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    import numpy as np

    base = {
        "dep_hour": float(dep_hour),
        "month": int(month),
        "day_of_week": int(day_of_week),
        "DISTANCE": float(distance),
    }
    row: dict = {}
    for c in num_cols:
        if c in base:
            row[c] = base[c]
        elif weather and c in weather and weather[c] is not None:
            row[c] = float(weather[c])
        else:
            row[c] = np.nan
    for c in cat_cols:
        if c == "OP_CARRIER":
            row[c] = carrier.strip().upper()
        elif c == "ORIGIN":
            row[c] = origin.strip().upper()
        elif c == "DEST":
            row[c] = dest.strip().upper()
    ordered = num_cols + cat_cols
    return pd.DataFrame([{k: row[k] for k in ordered}])[ordered]
