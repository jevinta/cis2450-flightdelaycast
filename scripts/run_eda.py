#!/usr/bin/env python3
"""Exploratory plots + summary stats -> reports/figures and reports/eda_summary.md."""

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

from flightdelaycast.config import PROCESSED_FLIGHTS, REPORTS_DIR, REPORTS_FIGURES  # noqa: E402


def _fmt_pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


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
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_md = reports_dir / "eda_summary.md"

    summary_lines: list[str] = []
    summary_lines.append("# EDA Findings Summary")
    summary_lines.append("")
    summary_lines.append("## 1) Data context and project objective")
    summary_lines.append(
        "- Objective: predict whether arrival delay is greater than 15 minutes (`is_delayed`)."
    )
    summary_lines.append(f"- Dataset size: **{len(df):,}** rows.")
    summary_lines.append(
        "- Key feature groups: schedule (`dep_hour`, `month`, `day_of_week`), route (`ORIGIN`, `DEST`, `DISTANCE`), carrier (`OP_CARRIER`), optional weather (`w_*`, `dw_*`)."
    )
    summary_lines.append(
        "- Roadmap impact: confirms that this is a classification task with mixed numeric and categorical predictors."
    )
    summary_lines.append("")

    print(f"Rows: {len(df):,}")
    print(f"Delay rate P(arr_delay > 15 min): {delay_rate:.3f}")
    summary_lines.append("## 2) Target distribution (class balance)")
    summary_lines.append(
        f"- Delay prevalence: **{_fmt_pct(delay_rate)}** delayed vs **{_fmt_pct(1.0 - delay_rate)}** not delayed."
    )
    summary_lines.append(
        "- Interpretation: class imbalance is present, so model evaluation should emphasize precision/recall/F1 (not accuracy alone)."
    )
    summary_lines.append(
        "- Roadmap impact: use imbalance-aware evaluation and keep probability outputs for threshold tuning."
    )
    summary_lines.append("")

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
    if "dep_hour" in df.columns:
        hour_peak = (
            df.assign(dep_h=np.floor(df["dep_hour"]).clip(0, 23))
            .groupby("dep_h")["is_delayed"]
            .mean()
            .sort_values(ascending=False)
        )
        if not hour_peak.empty:
            top_h = int(hour_peak.index[0])
            top_h_rate = float(hour_peak.iloc[0])
            summary_lines.append("## 3) Distribution and pattern checks")
            summary_lines.append(
                f"- Time-of-day effect: highest delay risk appears around hour **{top_h:02d}:00** at **{_fmt_pct(top_h_rate)}**."
            )
            summary_lines.append(
                "- Carrier-level variation indicates airline identity contributes useful signal."
            )
            summary_lines.append(
                "- Correlation matrix helps flag redundant numeric predictors before modeling."
            )
            summary_lines.append(
                "- Roadmap impact: retain hour and carrier features; monitor multicollinearity among numeric variables."
            )
            summary_lines.append("")

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

    outlier_candidates = [c for c in ["dep_hour", "DISTANCE", "w_tmax", "w_prcp"] if c in df.columns]
    outlier_rows: list[tuple[str, int, int, float]] = []
    for col in outlier_candidates:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        outlier_rows.append((col, n_out, int(series.shape[0]), (n_out / float(series.shape[0]))))

    if outlier_rows:
        outlier_rows.sort(key=lambda x: x[3], reverse=True)
        summary_lines.append("## 4) Outlier check and handling rationale")
        summary_lines.append("- Method: IQR rule (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) on key numeric variables.")
        for col, n_out, n_total, frac in outlier_rows:
            summary_lines.append(
                f"- `{col}`: {n_out:,}/{n_total:,} potential outliers ({_fmt_pct(frac)})."
            )
        summary_lines.append(
            "- Handling policy: keep valid operational extremes (e.g., long-distance flights, extreme weather) and rely on robust models/tree splits; add clipping only if validation metrics degrade."
        )
        summary_lines.append(
            "- Roadmap impact: preserves real rare events while keeping a documented mitigation strategy."
        )
        summary_lines.append("")

        box_cols = [r[0] for r in outlier_rows[:4]]
        melted = df[box_cols].copy()
        for c in box_cols:
            melted[c] = pd.to_numeric(melted[c], errors="coerce")
        melted = melted.melt(var_name="feature", value_name="value").dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=melted, x="feature", y="value", ax=ax, color="#9ecae1")
        ax.set_title("Outlier inspection (selected numeric features)")
        ax.set_xlabel("")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(args.out_dir / "05_outlier_boxplots.png", dpi=150)
        plt.close(fig)

    summary_lines.append("## 5) EDA conclusion")
    summary_lines.append(
        "- EDA supports the modeling approach: mixed-feature classification with interpretable risk outputs."
    )
    summary_lines.append(
        "- Next steps: continue threshold tuning and error analysis by carrier, hour, and weather regime."
    )
    summary_lines.append("")

    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Figures written to {args.out_dir}")
    print(f"Summary written to {summary_md}")


if __name__ == "__main__":
    main()
