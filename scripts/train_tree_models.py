#!/usr/bin/env python3
"""Train Random Forest and/or Histogram Gradient Boosting on processed flights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flightdelaycast.config import MODELS_DIR, PROCESSED_FLIGHTS  # noqa: E402
from flightdelaycast.model_features import feature_columns  # noqa: E402


def _preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    # Dense OHE: HistGradientBoostingClassifier does not accept sparse X.
    # RandomForest works with dense as well.
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False),
                cat_cols,
            ),
        ]
    )


def _metrics(y_test: pd.Series, pred: np.ndarray, *, n_train: int, n_test: int) -> dict:
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n_train": n_train,
        "n_test": n_test,
        "delay_rate_test": float(y_test.mean()),
    }


def _save_importances(pipe: Pipeline, out: Path, top_n: int = 25) -> None:
    clf = pipe.named_steps["clf"]
    prep = pipe.named_steps["prep"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = prep.get_feature_names_out()
    imp = np.asarray(clf.feature_importances_, dtype=float)
    order = np.argsort(-imp)[:top_n]
    rows = [{"feature": str(names[i]), "importance": float(imp[i])} for i in order]
    out.write_text(json.dumps(rows, indent=2))


def _train_rf(X_train, X_test, y_train, y_test, num_cols, cat_cols, seed: int):
    pipe = Pipeline(
        steps=[
            ("prep", _preprocess(num_cols, cat_cols)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=24,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = _metrics(y_test, pred, n_train=len(X_train), n_test=len(X_test))
    return pipe, metrics, pred


def _train_hgb(X_train, X_test, y_train, y_test, num_cols, cat_cols, seed: int):
    pipe = Pipeline(
        steps=[
            ("prep", _preprocess(num_cols, cat_cols)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.08,
                    max_depth=7,
                    max_leaf_nodes=48,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=20,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = _metrics(y_test, pred, n_train=len(X_train), n_test=len(X_test))
    return pipe, metrics, pred


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PROCESSED_FLIGHTS)
    p.add_argument("--model", choices=("rf", "hgb", "both"), default="both")
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.data.exists():
        print(f"Missing processed data: {args.data}", file=sys.stderr)
        print("Run: python scripts/build_processed.py", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(args.data, low_memory=False)
    num_cols, cat_cols = feature_columns(df)
    use_cols = num_cols + cat_cols
    miss = [c for c in use_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns for modeling: {miss}")

    X = df[use_cols]
    y = df["is_delayed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feat_meta = {"numeric": num_cols, "categorical": cat_cols}
    results: dict[str, dict] = {}

    if args.model in ("rf", "both"):
        pipe, metrics, pred = _train_rf(X_train, X_test, y_train, y_test, num_cols, cat_cols, args.seed)
        stem = args.out_dir / "random_forest"
        joblib.dump(pipe, stem.with_suffix(".joblib"))
        stem.with_name("random_forest_metrics.json").write_text(json.dumps(metrics, indent=2))
        stem.with_name("random_forest_features.json").write_text(json.dumps(feat_meta, indent=2))
        _save_importances(pipe, stem.with_name("random_forest_importances.json"))
        results["random_forest"] = metrics
        print("=== Random Forest ===")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, digits=3))

    if args.model in ("hgb", "both"):
        pipe, metrics, pred = _train_hgb(X_train, X_test, y_train, y_test, num_cols, cat_cols, args.seed)
        stem = args.out_dir / "hist_gradient_boosting"
        joblib.dump(pipe, stem.with_name("hist_gradient_boosting.joblib"))
        stem.with_name("hist_gradient_boosting_metrics.json").write_text(json.dumps(metrics, indent=2))
        stem.with_name("hist_gradient_boosting_features.json").write_text(json.dumps(feat_meta, indent=2))
        _save_importances(pipe, stem.with_name("hist_gradient_boosting_importances.json"))
        results["hist_gradient_boosting"] = metrics
        print("=== HistGradientBoosting ===")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, digits=3))

    if len(results) > 1:
        comp = args.out_dir / "tree_models_comparison.json"
        comp.write_text(json.dumps(results, indent=2))
        print(f"Wrote comparison -> {comp}")


if __name__ == "__main__":
    main()
