"""Map manual weather condition labels to numeric stand-ins for ``w_*`` / ``dw_*`` columns."""

from __future__ import annotations

from typing import Final

# Values are on scales compatible with Meteostat-style daily fields used in training
# (°C for temperatures, mm/day precipitation, km/h wind).
MANUAL_WEATHER_OPTIONS: Final[tuple[str, ...]] = (
    "Clear",
    "Cloudy",
    "Rain",
    "Thunderstorm",
    "Snow",
    "Fog",
    "Windy",
)

_PROFILES: Final[dict[str, dict[str, float]]] = {
    "Clear": {"tmax": 26.0, "tmin": 14.0, "prcp": 0.0, "wspd": 10.0},
    "Cloudy": {"tmax": 18.0, "tmin": 12.0, "prcp": 0.0, "wspd": 18.0},
    "Rain": {"tmax": 16.0, "tmin": 11.0, "prcp": 12.0, "wspd": 22.0},
    "Thunderstorm": {"tmax": 17.0, "tmin": 13.0, "prcp": 28.0, "wspd": 35.0},
    "Snow": {"tmax": -1.0, "tmin": -7.0, "prcp": 4.0, "wspd": 25.0},
    "Fog": {"tmax": 14.0, "tmin": 12.0, "prcp": 0.2, "wspd": 6.0},
    "Windy": {"tmax": 17.0, "tmin": 9.0, "prcp": 0.0, "wspd": 40.0},
}


def manual_weather_to_numeric_row(
    departure_condition: str,
    destination_condition: str,
    *,
    origin_keys: list[str],
    dest_keys: list[str],
) -> dict[str, float]:
    """Build a flat dict of numeric weather columns present in the trained model.

    **MANUAL FALLBACK — handled here:** categorical dropdowns are converted into plausible
    daily numerics so the existing sklearn pipeline can consume them. This is **not** learned
    from BTS; it is a **scenario** input for demos when forecasts are unavailable far in advance.
    """
    o_prof = _PROFILES.get(departure_condition, _PROFILES["Clear"])
    d_prof = _PROFILES.get(destination_condition, _PROFILES["Clear"])
    out: dict[str, float] = {}
    for k in origin_keys:
        suffix = k.removeprefix("w_")
        if suffix in o_prof:
            out[k] = float(o_prof[suffix])
    for k in dest_keys:
        suffix = k.removeprefix("dw_")
        if suffix in d_prof:
            out[k] = float(d_prof[suffix])
    return out
