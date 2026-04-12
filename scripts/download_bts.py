#!/usr/bin/env python3
"""Download BTS Marketing Carrier On-Time Performance zips (Jan 2018+).

Example:
  python scripts/download_bts.py --year 2024 --months 1 2 3

If automated download fails (503/maintenance), use the TranStats site manually:
  https://www.transtats.bts.gov/PREZIP/
Place extracted .csv files under data/raw/bts/
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from flightdelaycast.config import DATA_RAW_BTS  # noqa: E402

BASE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
)
USER_AGENT = (
    "Mozilla/5.0 (compatible; cis2450-flightdelaycast/0.1; educational project; +https://github.com/)"
)


def download_one(year: int, month: int, dest_dir: Path, session: requests.Session) -> Path:
    url = BASE.format(year=year, month=month)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"bts_{year}_{month:02d}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 1000:
        print(f"Skip existing {zip_path.name}")
        return zip_path

    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            r = session.get(url, stream=True, timeout=120)
            if r.status_code == 503:
                raise RuntimeError(f"HTTP 503 (server busy); retry later or download manually: {url}")
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            break
        except Exception as e:
            last_err = e
            wait = min(30, 2**attempt)
            print(f"Attempt {attempt} failed ({e}); sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    else:
        raise last_err  # type: ignore[misc]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"Downloaded and extracted: {url} -> {dest_dir}")
    return zip_path


def main() -> None:
    p = argparse.ArgumentParser(description="Download BTS on-time CSV zips.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--months", type=int, nargs="+", required=True, help="Month numbers 1-12")
    p.add_argument(
        "--out",
        type=Path,
        default=DATA_RAW_BTS,
        help=f"Output directory (default: {DATA_RAW_BTS})",
    )
    args = p.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for m in args.months:
        if not 1 <= m <= 12:
            raise SystemExit(f"Invalid month: {m}")
        download_one(args.year, m, args.out, session)


if __name__ == "__main__":
    main()
