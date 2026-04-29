"""Weather UI mode (automatic forecast vs manual) and stub for live forecast API."""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
class WeatherUIMode(str, Enum):
    """How the Streamlit demo supplies numeric ``w_*`` / ``dw_*`` columns for the model."""

    AUTOMATIC_FORECAST = "automatic"
    MANUAL_SCENARIO = "manual"


# Product rule: only dates within this many days from "today" are eligible for live forecast UX.
# **Tune this** to match your provider (many free tiers expose ~7–16 days of daily forecast).
SUPPORTED_FORECAST_DAYS = 14


def resolve_weather_ui_mode(
    flight_date: date,
    *,
    reference_date: date,
) -> WeatherUIMode:
    """Return which weather UX path applies for ``flight_date`` relative to ``reference_date``.

    **Automatic mode:** ``reference_date <= flight_date <= reference_date + SUPPORTED_FORECAST_DAYS``.
    The UI hides manual weather dropdowns and should call :func:`fetch_live_forecast_weather`.

    **Manual mode:** dates farther out than the supported forecast horizon (still capped by the app
    at one year). The UI shows departure + destination condition dropdowns mapped to numeric
    stand-ins in :mod:`dashboard.manual_weather_numeric`.
    """
    if flight_date < reference_date:
        # Caller should block invalid dates; treat as manual to avoid implying live data exists.
        return WeatherUIMode.MANUAL_SCENARIO
    last_auto = reference_date + timedelta(days=SUPPORTED_FORECAST_DAYS)
    if flight_date <= last_auto:
        return WeatherUIMode.AUTOMATIC_FORECAST
    return WeatherUIMode.MANUAL_SCENARIO


def fetch_live_forecast_weather(
    origin_iata: str,
    dest_iata: str,
    flight_date: date,
    *,
    required_numeric_keys: list[str],
) -> dict[str, float] | None:
    """Fetch real forecast-derived numerics for the trained schema.

    **REAL WEATHER API — connect here:** replace the body with HTTP calls to your provider
    (OpenWeather, Tomorrow.io, NOAA, etc.). Map responses onto the keys your model expects
    (e.g. ``w_tmax``, ``w_tmin``, ``w_prcp``, ``w_wspd`` at origin and ``dw_*`` at destination).

    - Use ``origin_iata`` / ``dest_iata`` and ``flight_date`` to query **forecast** endpoints
      (historical Meteostat **Daily** is *not* a substitute for future dates).
    - Return a **dense** ``dict`` of floats for every key listed in ``required_numeric_keys`` that
      your API can populate; omit keys you cannot fill (the model pipeline will **median-impute**
      missing numerics like training).

    **Fallback:** return ``None`` on failure; the app still stays in **automatic** mode (no manual
    dropdowns) and passes ``NaN``s so imputation applies, with a small caption that the provider
    is not configured or the call failed.
    """
    del origin_iata, dest_iata, flight_date, required_numeric_keys
    # ------------------------------------------------------------------ live API integration stub
    return None
