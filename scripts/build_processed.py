#!/usr/bin/env python3
"""Clean BTS CSVs, merge airport coordinates, optional origin weather, write parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from flightdelaycast.config import (  # noqa: E402
    AIRPORTS_CSV,
    DATA_RAW_BTS,
    PROCESSED_FLIGHTS,
    WEATHER_CACHE,
)
from flightdelaycast.weather_origin import fetch_origin_weather_for_pairs  # noqa: E402
from flightdelaycast.wrangle import clean_flights, load_bts_csvs, merge_airport_coords, sample_rows  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bts-dir", type=Path, default=DATA_RAW_BTS)
    p.add_argument("--airports", type=Path, default=AIRPORTS_CSV)
    p.add_argument("--out", type=Path, default=PROCESSED_FLIGHTS)
    p.add_argument("--max-rows", type=int, default=150_000, help="Random subsample after cleaning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weather", action="store_true", help="Fetch Meteostat daily weather at origin")
    p.add_argument(
        "--weather-max-pairs",
        type=int,
        default=2500,
        help="Cap unique (ORIGIN, date) pairs for weather API calls",
    )
    args = p.parse_args()

    if not args.airports.exists():
        print(f"Airports file missing: {args.airports}", file=sys.stderr)
        print("Run: python scripts/download_airports.py", file=sys.stderr)
        raise SystemExit(1)

    raw = load_bts_csvs(args.bts_dir)
    clean = clean_flights(raw)
    clean = merge_airport_coords(clean, args.airports)
    clean = sample_rows(clean, args.max_rows, seed=args.seed)

    if args.weather:
        pairs = clean[["ORIGIN", "FL_DATE"]].drop_duplicates()
        if len(pairs) > args.weather_max_pairs:
            pairs = pairs.sample(n=args.weather_max_pairs, random_state=args.seed)
        wx = fetch_origin_weather_for_pairs(pairs, clean[["ORIGIN", "origin_lat", "origin_lon"]])
        WEATHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        wx.to_csv(WEATHER_CACHE, index=False, compression="gzip")
        clean["FL_DATE_ONLY"] = pd.to_datetime(clean["FL_DATE"]).dt.normalize()
        wx["FL_DATE_ONLY"] = pd.to_datetime(wx["FL_DATE"]).dt.normalize()
        wx_small = wx.drop(columns=["FL_DATE"], errors="ignore")
        clean = clean.merge(wx_small, on=["ORIGIN", "FL_DATE_ONLY"], how="left")
        clean = clean.drop(columns=["FL_DATE_ONLY"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False, compression="gzip")
    print(f"Wrote {args.out} ({len(clean):,} rows, {len(clean.columns)} cols)")


if __name__ == "__main__":
    main()
