from pathlib import Path

from flightdelaycast.route_distance import great_circle_miles_between_airports

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "airports_sample.csv"


def test_jfk_lax_distance_positive() -> None:
    mi = great_circle_miles_between_airports("JFK", "LAX", FIXTURE)
    assert mi is not None
    assert 2400 < mi < 2500


def test_same_airport_returns_none() -> None:
    assert great_circle_miles_between_airports("JFK", "JFK", FIXTURE) is None
