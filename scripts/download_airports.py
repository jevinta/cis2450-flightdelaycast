#!/usr/bin/env python3
"""Download OurAirports open data (IATA codes + coordinates) for US airports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from flightdelaycast.config import AIRPORTS_CSV  # noqa: E402

# Redirect target used by ourairports.com (stable for scripting)
URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=AIRPORTS_CSV)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    args.out.write_bytes(r.content)
    print(f"Wrote {args.out} ({len(r.content) // 1024} KB)")


if __name__ == "__main__":
    main()
