"""Weather UI mode and live forecast via Open-Meteo (free, no API key required)."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from flightdelaycast.config import AIRPORTS_CSV


class WeatherUIMode(str, Enum):
    """How the Streamlit demo supplies numeric ``w_*`` / ``dw_*`` columns for the model."""

    AUTOMATIC_FORECAST = "automatic"
    MANUAL_SCENARIO = "manual"


# Open-Meteo free tier covers 16 days ahead; cap at 7 ("a week or less") per project spec.
SUPPORTED_FORECAST_DAYS = 7


def resolve_weather_ui_mode(
    flight_date: date,
    *,
    reference_date: date,
) -> WeatherUIMode:
    if flight_date < reference_date:
        return WeatherUIMode.MANUAL_SCENARIO
    last_auto = reference_date + timedelta(days=SUPPORTED_FORECAST_DAYS)
    if flight_date <= last_auto:
        return WeatherUIMode.AUTOMATIC_FORECAST
    return WeatherUIMode.MANUAL_SCENARIO


@lru_cache(maxsize=512)
def _airport_latlon(iata: str) -> tuple[float, float] | None:
    """Cached airport coordinate lookup from airports.csv."""
    if not AIRPORTS_CSV.is_file():
        return None
    try:
        import pandas as pd
        ap = pd.read_csv(AIRPORTS_CSV, low_memory=False)
    except Exception:
        return None
    ap = ap[ap["iata_code"].notna()]
    ap = ap.drop_duplicates(subset=["iata_code"])
    idx = ap.set_index(ap["iata_code"].astype(str).str.strip().str.upper())
    code = iata.strip().upper()
    if code not in idx.index:
        return None
    row = idx.loc[code]
    try:
        return float(row["latitude_deg"]), float(row["longitude_deg"])
    except Exception:
        return None


def _open_meteo_daily(lat: float, lon: float, date_str: str) -> dict[str, float] | None:
    """Fetch one day of weather from Open-Meteo forecast API.

    Maps API fields to Meteostat-compatible names used in training:
      temperature_2m_max  → tmax (°C)
      temperature_2m_min  → tmin (°C)
      precipitation_sum   → prcp (mm)
      wind_speed_10m_max  → wspd (km/h)
    """
    params = urllib.parse.urlencode({
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "flightdelaycast/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    daily = data.get("daily", {})

    def _first(key: str) -> float | None:
        vals = daily.get(key)
        if vals and len(vals) > 0 and vals[0] is not None:
            return float(vals[0])
        return None

    tmax = _first("temperature_2m_max")
    if tmax is None:
        return None

    tmin = _first("temperature_2m_min")
    prcp = _first("precipitation_sum")
    wspd = _first("wind_speed_10m_max")
    return {
        "tmax": tmax,
        "tmin": tmin if tmin is not None else tmax - 8.0,
        "prcp": prcp if prcp is not None else 0.0,
        "wspd": wspd if wspd is not None else 15.0,
    }


def fetch_live_forecast_weather(
    origin_iata: str,
    dest_iata: str,
    flight_date: date,
    *,
    required_numeric_keys: list[str],
) -> dict[str, float] | None:
    """Fetch forecast weather from Open-Meteo for origin and/or destination on flight_date.

    Returns a dict keyed by the model's weather column names (e.g. ``w_tmax``, ``dw_prcp``),
    or ``None`` if both lookups fail (pipeline will median-impute missing keys).
    """
    date_str = flight_date.isoformat()
    origin_keys = [k for k in required_numeric_keys if k.startswith("w_") and not k.startswith("dw_")]
    dest_keys = [k for k in required_numeric_keys if k.startswith("dw_")]

    result: dict[str, float] = {}

    if origin_keys:
        coords = _airport_latlon(origin_iata)
        if coords:
            wx = _open_meteo_daily(coords[0], coords[1], date_str)
            if wx:
                for k in origin_keys:
                    suffix = k.removeprefix("w_")
                    if suffix in wx:
                        result[k] = wx[suffix]

    if dest_keys:
        coords = _airport_latlon(dest_iata)
        if coords:
            wx = _open_meteo_daily(coords[0], coords[1], date_str)
            if wx:
                for k in dest_keys:
                    suffix = k.removeprefix("dw_")
                    if suffix in wx:
                        result[k] = wx[suffix]

    return result if result else None
