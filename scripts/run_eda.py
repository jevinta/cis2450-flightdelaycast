#!/usr/bin/env python3
"""Exploratory plots + summary stats -> reports/figures (and stdout)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Avoid crashes when ~/.matplotlib is not writable (CI/sandbox); must run before pyplot import.
_mpl_dir = REPO_ROOT / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
# Headless / script-safe backend (avoids GUI toolkit + some font-cache issues).
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flightdelaycast.config import PROCESSED_FLIGHTS, REPORTS_FIGURES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=PROCESSED_FLIGHTS)
    p.add_argument("--out-dir", type=Path, default=REPORTS_FIGURES)
    args = p.parse_args()

    if not args.data.exists():
        print(f"Missing processed data: {args.data}", file=sys.stderr)
        print("Run: python scripts/build_processed.py", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(args.data, low_memory=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    delay_rate = df["is_delayed"].mean()
    print(f"Rows: {len(df):,}")
    print(f"Delay rate P(arr_delay > 15 min): {delay_rate:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    df["is_delayed"].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_xticklabels(["On time / small delay", "Delayed"], rotation=0)
    ax.set_ylabel("Flights")
    ax.set_title("Class balance (target)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "01_class_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    hourly = df.assign(dep_h=np.floor(df["dep_hour"]).clip(0, 23)).groupby("dep_h")["is_delayed"].mean()
    hourly.plot(ax=ax, marker="o")
    ax.set_xlabel("Scheduled dep hour (approx)")
    ax.set_ylabel("P(delayed)")
    ax.set_title("Delay rate by hour of day")
    fig.tight_layout()
    fig.savefig(args.out_dir / "02_delay_rate_by_hour.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = df["OP_CARRIER"].value_counts().head(12).index
    sub = df[df["OP_CARRIER"].isin(top)]
    order = sub.groupby("OP_CARRIER")["is_delayed"].mean().sort_values(ascending=False).index
    sns.barplot(data=sub, x="OP_CARRIER", y="is_delayed", order=order, ax=ax, color="#54a24b")
    ax.set_ylabel("P(delayed)")
    ax.set_xlabel("Carrier (top 12 by volume)")
    ax.set_title("Delay rate by airline")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "03_delay_rate_by_carrier.png", dpi=150)
    plt.close(fig)

    num_cols = [c for c in ["dep_hour", "month", "day_of_week", "DISTANCE", "w_tmax", "w_prcp"] if c in df.columns]
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df[num_cols + ["is_delayed"]].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
        ax.set_title("Correlation (numeric features + target)")
        fig.tight_layout()
        fig.savefig(args.out_dir / "04_correlation_numeric.png", dpi=150)
        plt.close(fig)

    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
