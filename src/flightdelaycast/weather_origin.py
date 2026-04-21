"""Fetch daily weather at flight origin (Meteostat) for unique (ORIGIN, FL_DATE) keys."""

from __future__ import annotations

from flightdelaycast.meteostat_daily import fetch_daily_weather_at_airports


def fetch_origin_weather_for_pairs(
    pairs: pd.DataFrame,
    origin_coords: pd.DataFrame,
    *,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """pairs: columns ORIGIN, FL_DATE. origin_coords: ORIGIN, origin_lat, origin_lon."""
    return fetch_daily_weather_at_airports(
        pairs,
        origin_coords,
        airport_col="ORIGIN",
        lat_col="origin_lat",
        lon_col="origin_lon",
        value_prefix="w_",
        sleep_s=sleep_s,
    )
