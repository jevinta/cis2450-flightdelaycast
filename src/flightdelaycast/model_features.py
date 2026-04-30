"""Feature column lists shared by training scripts and the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_ORIGIN_NUMERIC = ("w_tmax", "w_tmin", "w_prcp", "w_wspd")
WEATHER_DEST_NUMERIC = ("dw_tmax", "dw_tmin", "dw_prcp", "dw_wspd")
# Back-compat alias
WEATHER_NUMERIC = WEATHER_ORIGIN_NUMERIC


def drop_highly_correlated_numeric(
    df: pd.DataFrame, numeric_cols: list[str], threshold: float = 0.9
) -> tuple[list[str], list[str]]:
    """Drop redundant numeric columns using Pearson correlation on df (pass training rows only)."""
    usable = [c for c in numeric_cols if c in df.columns]
    if len(usable) < 2:
        return usable, []
    corr = df[usable].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    return [c for c in usable if c not in to_drop], to_drop


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = ["dep_hour", "month", "day_of_week", "DISTANCE"]
    for c in ("dep_hour_sin", "dep_hour_cos"):
        if c in df.columns:
            numeric.append(c)
    for c in WEATHER_ORIGIN_NUMERIC + WEATHER_DEST_NUMERIC:
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
        "dep_hour_sin": float(np.sin(2 * np.pi * dep_hour / 24)),
        "dep_hour_cos": float(np.cos(2 * np.pi * dep_hour / 24)),
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
