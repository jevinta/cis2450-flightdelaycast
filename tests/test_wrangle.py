import unittest
from pathlib import Path

import pandas as pd

from flightdelaycast.wrangle import clean_flights, load_bts_csvs, merge_airport_coords

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BTS_FIXTURE_DIR = FIXTURES / "bts"


class TestWrangle(unittest.TestCase):
    def test_clean_and_target(self) -> None:
        raw = load_bts_csvs(BTS_FIXTURE_DIR)
        clean = clean_flights(raw)
        self.assertIn("is_delayed", clean.columns)
        self.assertEqual(clean["is_delayed"].isin([0, 1]).all(), True)
        # Two rows with ARR_DELAY > 15 in fixture
        self.assertGreaterEqual(clean["is_delayed"].sum(), 1)

    def test_merge_airports(self) -> None:
        raw = load_bts_csvs(BTS_FIXTURE_DIR)
        clean = clean_flights(raw)
        merged = merge_airport_coords(clean, FIXTURES / "airports_sample.csv")
        self.assertIn("origin_lat", merged.columns)
        self.assertGreater(merged["origin_lat"].notna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
