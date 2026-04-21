"""Great-circle distance between two IATA airports using OurAirports-style CSV."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    km = r_km * c
    return km * 0.621371


def great_circle_miles_between_airports(
    origin_iata: str,
    dest_iata: str,
    airports_csv: Path,
    *,
    iso_country: str = "US",
) -> float | None:
    """Return stage length in miles, or ``None`` if codes are missing or file unreadable."""
    o = str(origin_iata).strip().upper()
    d = str(dest_iata).strip().upper()
    if not o or not d or o == d:
        return None
    if not airports_csv.is_file():
        return None
    try:
        ap = pd.read_csv(airports_csv, low_memory=False)
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    if "iata_code" not in ap.columns or "latitude_deg" not in ap.columns or "longitude_deg" not in ap.columns:
        return None
    if "iso_country" in ap.columns:
        ap = ap[ap["iso_country"] == iso_country]
    ap = ap[ap["iata_code"].notna() & (ap["iata_code"].astype(str).str.strip() != "")]
    ap = ap.drop_duplicates(subset=["iata_code"], keep="first")
    idx = ap.set_index(ap["iata_code"].astype(str).str.strip().str.upper())
    if o not in idx.index or d not in idx.index:
        return None
    lat1, lon1 = float(idx.loc[o, "latitude_deg"]), float(idx.loc[o, "longitude_deg"])
    lat2, lon2 = float(idx.loc[d, "latitude_deg"]), float(idx.loc[d, "longitude_deg"])
    if any(map(math.isnan, (lat1, lon1, lat2, lon2))):
        return None
    return float(_haversine_miles(lat1, lon1, lat2, lon2))
