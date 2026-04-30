"""Daily station weather via Meteostat (shared for origin / destination airport-day keys)."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from meteostat import Daily, Stations
from meteostat.interface.base import Base
from tqdm import tqdm

# Default Meteostat cache is under ~/.meteostat/cache.
# Use a local project cache to avoid permission issues in sandboxed environments.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_METEOSTAT_CACHE = _REPO_ROOT / ".cache" / "meteostat" / "cache"
if os.environ.get("FLIGHTDELAYCAST_USE_LOCAL_METEOSTAT_CACHE", "1") not in ("0", "false", "False"):
    _LOCAL_METEOSTAT_CACHE.mkdir(parents=True, exist_ok=True)
    Base.cache_dir = str(_LOCAL_METEOSTAT_CACHE)


def _nearest_station_id(lat: float, lon: float) -> str | None:
    stations = Stations()
    nearby = stations.nearby(lat, lon).fetch(1)
    if nearby is None or nearby.empty:
        return None
    return str(nearby.index[0])


def fetch_daily_weather_at_airports(
    pairs: pd.DataFrame,
    airport_coords: pd.DataFrame,
    *,
    airport_col: str,
    lat_col: str,
    lon_col: str,
    value_prefix: str,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """Fetch one Meteostat Daily row per (airport, calendar day).

    pairs: columns ``airport_col``, ``FL_DATE`` (datetime-like).
    airport_coords: columns ``airport_col``, ``lat_col``, ``lon_col`` (one row per airport).
    value_prefix: e.g. ``\"w_\"`` -> ``w_tmax``, ``\"dw_\"`` -> ``dw_tmax``.
    """
    need = pairs[[airport_col, "FL_DATE"]].drop_duplicates()
    coord = airport_coords.drop_duplicates(airport_col).set_index(airport_col)[[lat_col, lon_col]]

    station_by_airport: dict[str, str] = {}
    for ap in tqdm(need[airport_col].unique(), desc=f"Resolve stations ({airport_col})"):
        if ap not in coord.index:
            continue
        lat = float(coord.loc[ap, lat_col])
        lon = float(coord.loc[ap, lon_col])
        if pd.isna(lat) or pd.isna(lon):
            continue
        sid = _nearest_station_id(lat, lon)
        if sid:
            station_by_airport[str(ap)] = sid
        time.sleep(sleep_s)

    field_map = (
        ("tmax", f"{value_prefix}tmax"),
        ("tmin", f"{value_prefix}tmin"),
        ("prcp", f"{value_prefix}prcp"),
        ("wspd", f"{value_prefix}wspd"),
    )

    rows: list[dict] = []
    for _, r in tqdm(need.iterrows(), total=len(need), desc=f"Fetch daily weather ({airport_col})"):
        ap_code = str(r[airport_col])
        day = pd.Timestamp(r["FL_DATE"]).to_pydatetime().date()
        sid = station_by_airport.get(ap_code)
        if not sid:
            continue
        d0 = datetime(day.year, day.month, day.day)
        data = Daily(sid, d0, d0).fetch()
        time.sleep(sleep_s)
        if data is None or data.empty:
            continue
        row = data.iloc[0]
        rec: dict = {airport_col: ap_code, "FL_DATE": pd.Timestamp(day)}
        for src, dst in field_map:
            rec[dst] = float(row.get(src)) if pd.notna(row.get(src)) else None
        rows.append(rec)

    return pd.DataFrame(rows)
