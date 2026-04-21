"""Fetch daily weather at flight destination (Meteostat) for unique (DEST, FL_DATE) keys."""

from __future__ import annotations

from flightdelaycast.meteostat_daily import fetch_daily_weather_at_airports


def fetch_destination_weather_for_pairs(
    pairs: pd.DataFrame,
    dest_coords: pd.DataFrame,
    *,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """pairs: columns DEST, FL_DATE. dest_coords: DEST, dest_lat, dest_lon."""
    return fetch_daily_weather_at_airports(
        pairs,
        dest_coords,
        airport_col="DEST",
        lat_col="dest_lat",
        lon_col="dest_lon",
        value_prefix="dw_",
        sleep_s=sleep_s,
    )
