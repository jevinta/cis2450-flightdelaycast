"""Fetch daily weather at flight origin (Meteostat) for unique (ORIGIN, FL_DATE) keys."""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
from meteostat import Daily, Stations
from tqdm import tqdm

def _nearest_station_id(lat: float, lon: float) -> str | None:
    stations = Stations()
    nearby = stations.nearby(lat, lon).fetch(1)
    if nearby is None or nearby.empty:
        return None
    return str(nearby.index[0])


def fetch_origin_weather_for_pairs(
    pairs: pd.DataFrame,
    origin_coords: pd.DataFrame,
    *,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """pairs: columns ORIGIN, FL_DATE (datetime). origin_coords: ORIGIN, origin_lat, origin_lon."""
    need = pairs[["ORIGIN", "FL_DATE"]].drop_duplicates()
    coord = origin_coords.drop_duplicates("ORIGIN").set_index("ORIGIN")[["origin_lat", "origin_lon"]]

    station_by_origin: dict[str, str] = {}
    for origin in tqdm(need["ORIGIN"].unique(), desc="Resolve Meteostat stations"):
        if origin not in coord.index:
            continue
        lat = float(coord.loc[origin, "origin_lat"])
        lon = float(coord.loc[origin, "origin_lon"])
        if pd.isna(lat) or pd.isna(lon):
            continue
        sid = _nearest_station_id(lat, lon)
        if sid:
            station_by_origin[str(origin)] = sid
        time.sleep(sleep_s)

    rows: list[dict] = []
    for _, r in tqdm(need.iterrows(), total=len(need), desc="Fetch daily weather"):
        origin = str(r["ORIGIN"])
        day = pd.Timestamp(r["FL_DATE"]).to_pydatetime().date()
        sid = station_by_origin.get(origin)
        if not sid:
            continue
        d0 = datetime(day.year, day.month, day.day)
        data = Daily(sid, d0, d0).fetch()
        time.sleep(sleep_s)
        if data is None or data.empty:
            continue
        row = data.iloc[0]
        rows.append(
            {
                "ORIGIN": origin,
                "FL_DATE": pd.Timestamp(day),
                "w_tmax": float(row.get("tmax")) if pd.notna(row.get("tmax")) else None,
                "w_tmin": float(row.get("tmin")) if pd.notna(row.get("tmin")) else None,
                "w_prcp": float(row.get("prcp")) if pd.notna(row.get("prcp")) else None,
                "w_wspd": float(row.get("wspd")) if pd.notna(row.get("wspd")) else None,
            }
        )

    return pd.DataFrame(rows)
