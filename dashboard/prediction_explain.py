"""Heuristic explanations: logistic contributions when possible, else rule-based factors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class PredictionNarrative:
    factors: list[str]
    recommendation: str


def approximate_expected_delay_minutes(delay_probability: float) -> float | None:
    """Return a bounded delay estimate from delay probability for UI display."""
    p = float(np.clip(delay_probability, 0.0, 1.0))
    if p <= 0.02:
        return None
    # Smooth mapping: higher P(delay) means higher expected delay if late.
    base = 15.0 + 95.0 * (p**1.15)
    return float(min(240.0, max(16.0, base)))


def _top_linear_factors(model: Pipeline, row: pd.DataFrame, k: int = 5) -> list[str] | None:
    if "prep" not in model.named_steps or "clf" not in model.named_steps:
        return None
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    if not hasattr(clf, "coef_"):
        return None
    coef = np.asarray(clf.coef_, dtype=float).ravel()
    try:
        Xt = prep.transform(row)
    except Exception:
        return None
    if sp_sparse.issparse(Xt):
        if Xt.shape[1] != coef.shape[0]:
            return None
        # np.asarray(sparse) does not densify, so convert explicitly before ravel().
        prod = Xt.multiply(coef)
        contrib = np.asarray(prod.toarray(), dtype=np.float64).ravel()
        names = prep.get_feature_names_out()
        order = np.argsort(np.abs(contrib))[::-1][:k]
        lines: list[str] = []
        for i in order:
            c = float(contrib[i])
            nm = str(names[i])
            direction = "increases" if c > 0 else "decreases"
            lines.append(f"**{nm}** {direction} delay log-odds for this row (≈ {c:+.4f}).")
        return lines

    x = np.asarray(Xt, dtype=float).ravel()
    if x.shape[0] != coef.shape[0]:
        return None
    contrib = x * coef
    names = prep.get_feature_names_out()
    order = np.argsort(np.abs(contrib))[::-1][:k]
    lines: list[str] = []
    for i in order:
        c = float(contrib[i])
        nm = str(names[i])
        direction = "increases" if c > 0 else "decreases"
        lines.append(f"**{nm}** {direction} delay risk for this row (linear term ≈ {c:+.3f}).")
    return lines


def _heuristic_factors(
    *,
    dep_hour: float,
    month: int,
    distance_mi: float,
    origin: str,
    dest: str,
    carrier: str,
    manual_departure_weather: str | None,
    manual_destination_weather: str | None,
    flight_date: date,
) -> list[str]:
    factors: list[str] = []
    if dep_hour >= 17 or dep_hour <= 6:
        factors.append("Scheduled departure is in a **peak or overnight** window where delays cluster in historical data.")
    if distance_mi > 1500:
        factors.append("**Long stage length** adds exposure to schedule recovery issues.")
    elif distance_mi < 250:
        factors.append("**Short hop** — less airborne time, but ground/taxi congestion can still matter.")
    if manual_departure_weather in ("Thunderstorm", "Snow", "Rain"):
        factors.append(f"Departure-side scenario **{manual_departure_weather}** pushes risk up in the manual-weather mapping.")
    if manual_destination_weather in ("Thunderstorm", "Snow", "Rain"):
        factors.append(f"Destination scenario **{manual_destination_weather}** is mapped to adverse conditions.")
    if manual_departure_weather == "Fog" or manual_destination_weather == "Fog":
        factors.append("**Fog** reduces throughput at affected airports.")
    if carrier and carrier != "UNK":
        factors.append(f"**Carrier {carrier}** shifts the baseline versus the network average (categorical effect in the model).")
    factors.append(f"**Route {origin} → {dest}** on **{flight_date.isoformat()}** anchors hub/congestion effects learned from data.")
    if month in (6, 7, 8, 12):
        factors.append("**Summer / holiday season** months often align with higher operational load.")
    return factors[:6]


def build_narrative(
    model: Any,
    model_id: str,
    row: pd.DataFrame,
    *,
    dep_hour: float,
    month: int,
    distance_mi: float,
    origin: str,
    dest: str,
    carrier: str,
    flight_date: date,
    delay_probability: float,
    manual_dep_wx: str | None,
    manual_dest_wx: str | None,
) -> PredictionNarrative:
    factors: list[str] | None = None
    if isinstance(model, Pipeline) and model_id == "logistic":
        factors = _top_linear_factors(model, row)

    if not factors:
        factors = _heuristic_factors(
            dep_hour=dep_hour,
            month=month,
            distance_mi=distance_mi,
            origin=origin,
            dest=dest,
            carrier=carrier,
            manual_departure_weather=manual_dep_wx,
            manual_destination_weather=manual_dest_wx,
            flight_date=flight_date,
        )

    if delay_probability >= 0.55:
        rec = (
            "Treat this as **elevated risk**: add connection buffer if transferring, "
            "check alternate same-day options, and monitor airline notifications."
        )
    elif delay_probability >= 0.35:
        rec = (
            "**Moderate risk** — arrive at the airport with normal buffer plus a little extra; "
            "watch inbound aircraft and posted delays."
        )
    else:
        rec = "**Lower modeled risk** — routine planning is reasonable, but real operations can still surprise."

    return PredictionNarrative(factors=factors, recommendation=rec)
