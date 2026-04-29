#!/usr/bin/env python3
"""Train a baseline logistic regression; save metrics and sklearn pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from flightdelaycast.config import MODELS_DIR, PROCESSED_FLIGHTS  # noqa: E402
from flightdelaycast.model_features import feature_columns  # noqa: E402


def _drop_highly_correlated_numeric(
    df: pd.DataFrame, numeric_cols: list[str], threshold: float = 0.9
) -> tuple[list[str], list[str]]:
    """Keep numeric columns while dropping high-correlation duplicates."""
    usable = [c for c in numeric_cols if c in df.columns]
    if len(usable) < 2:
        return usable, []
    corr = df[usable].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    keep = [c for c in usable if c not in to_drop]
    return keep, to_drop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=PROCESSED_FLIGHTS)
    p.add_argument("--out-model", type=Path, default=MODELS_DIR / "baseline_logistic.joblib")
    p.add_argument("--out-metrics", type=Path, default=MODELS_DIR / "baseline_metrics.json")
    p.add_argument("--out-features", type=Path, default=MODELS_DIR / "baseline_features.json")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.data.exists():
        print(f"Missing processed data: {args.data}", file=sys.stderr)
        print("Run: python scripts/build_processed.py", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(args.data, low_memory=False)
    num_cols, cat_cols = feature_columns(df)
    num_cols, dropped_corr = _drop_highly_correlated_numeric(df, num_cols, threshold=0.9)
    use_cols = num_cols + cat_cols
    miss = [c for c in use_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns for modeling: {miss}")

    X = df[use_cols]
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=True),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocess),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="saga",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "delay_rate_test": float(y_test.mean()),
        "corr_drop_threshold": 0.9,
        "dropped_high_corr_numeric": dropped_corr,
    }

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_model)

    args.out_metrics.write_text(json.dumps(metrics, indent=2))
    feat_meta = {"numeric": num_cols, "categorical": cat_cols}
    args.out_features.write_text(json.dumps(feat_meta, indent=2))
    print(json.dumps(metrics, indent=2))
    if dropped_corr:
        print(f"Dropped high-correlation numeric features: {dropped_corr}")
    print(classification_report(y_test, pred, digits=3))
    print(f"Saved model -> {args.out_model}")
    print(f"Saved metrics -> {args.out_metrics}")
    print(f"Saved feature schema -> {args.out_features}")


if __name__ == "__main__":
    main()
