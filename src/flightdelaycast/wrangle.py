"""Load BTS CSVs, clean, engineer features, optionally merge airport coords + origin weather."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from flightdelaycast.cleaning import add_delay_target
from flightdelaycast.config import DELAY_THRESHOLD_MINUTES

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        c2 = re.sub(r"\s+", "_", str(c).strip())
        rename[c] = c2
    out = out.rename(columns=rename)
    lower_map = {c.lower(): c for c in out.columns}
    # Common BTS aliases
    aliases = {
        "flightdate": "FL_DATE",
        "reporting_airline": "OP_CARRIER",
        "iata_code_reporting_airline": "OP_CARRIER",
    }
    for alias, canonical in aliases.items():
        if alias in lower_map and canonical not in out.columns:
            out = out.rename(columns={lower_map[alias]: canonical})
    return out


def _require_columns(out: pd.DataFrame) -> None:
    required = [
        "FL_DATE",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "ARR_DELAY",
        "CANCELLED",
        "DIVERTED",
        "DISTANCE",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        preview = list(out.columns)[:60]
        raise ValueError(f"Missing columns {missing}. First columns in file: {preview}")


def parse_crs_dep_time(series: pd.Series) -> pd.Series:
    """CRS_DEP_TIME is HHMM (e.g. 925 = 09:25). Returns hour 0-23."""

    def one(v: float | int | str | None) -> float:
        if pd.isna(v):
            return np.nan
        try:
            x = int(float(v))
        except (TypeError, ValueError):
            return np.nan
        if x < 0 or x > 2400:
            return np.nan
        h = x // 100
        m = x % 100
        if m > 59 or h > 23:
            return np.nan
        return float(h) + m / 60.0

    return series.map(one)


def load_bts_csvs(bts_dir: Path) -> pd.DataFrame:
    paths = sorted(
        p
        for p in bts_dir.rglob("*.csv")
        if not p.name.lower().startswith("readme")
    )
    if not paths:
        raise FileNotFoundError(f"No CSV files under {bts_dir} (extract BTS zips here).")
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled/diverted; require ARR_DELAY; add target and time features."""
    out = normalize_columns(df)
    _require_columns(out)

    for c in ("CANCELLED", "DIVERTED"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out = out[out["CANCELLED"] == 0]
    out = out[out["DIVERTED"] == 0]
    out["ARR_DELAY"] = pd.to_numeric(out["ARR_DELAY"], errors="coerce")
    out = out.dropna(subset=["ARR_DELAY"])
    out["DISTANCE"] = pd.to_numeric(out["DISTANCE"], errors="coerce")

    out["FL_DATE"] = pd.to_datetime(out["FL_DATE"], errors="coerce")
    out = out.dropna(subset=["FL_DATE"])

    out = add_delay_target(out, "ARR_DELAY")
    out = out.dropna(subset=["is_delayed"])
    out["is_delayed"] = out["is_delayed"].astype(int)

    out["dep_hour"] = parse_crs_dep_time(out["CRS_DEP_TIME"])
    out["month"] = out["FL_DATE"].dt.month
    out["day_of_week"] = out["FL_DATE"].dt.dayofweek
    for col in ("OP_CARRIER", "ORIGIN", "DEST"):
        out[col] = out[col].astype(str).str.strip()
    out = out.dropna(subset=["dep_hour", "OP_CARRIER", "ORIGIN", "DEST"])

    return out


def merge_airport_coords(flights: pd.DataFrame, airports_path: Path) -> pd.DataFrame:
    ap = pd.read_csv(airports_path, low_memory=False)
    ap = ap[(ap["iso_country"] == "US") & ap["iata_code"].notna() & (ap["iata_code"] != "")]
    ap = ap.rename(columns={"latitude_deg": "origin_lat", "longitude_deg": "origin_lon"})
    use = ap[["iata_code", "origin_lat", "origin_lon"]].drop_duplicates("iata_code")
    out = flights.merge(use, left_on="ORIGIN", right_on="iata_code", how="left")
    out = out.drop(columns=["iata_code"], errors="ignore")
    return out


def sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def wrangle_pipeline(
    bts_dir: Path,
    airports_path: Path,
    *,
    max_rows: int | None = 150_000,
    seed: int = 42,
) -> pd.DataFrame:
    raw = load_bts_csvs(bts_dir)
    clean = clean_flights(raw)
    clean = merge_airport_coords(clean, airports_path)
    clean = sample_rows(clean, max_rows, seed=seed)
    return clean
