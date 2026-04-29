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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flightdelaycast.config import MODELS_DIR, PROCESSED_FLIGHTS  # noqa: E402
from flightdelaycast.model_features import feature_columns  # noqa: E402

# Parameter spaces for RandomizedSearchCV (--tune flag).
# n_estimators, depth, leaf size, and feature-split strategy cover the main
# bias-variance knobs for RF; learning rate, depth, and regularisation for HGB.
RF_PARAM_DIST = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [10, 15, 20, 24],
    "clf__min_samples_leaf": [5, 10, 20],
    "clf__max_features": ["sqrt", "log2"],
}

HGB_PARAM_DIST = {
    "clf__learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2],
    "clf__max_depth": [5, 7, 9],
    "clf__max_leaf_nodes": [31, 48, 63],
    "clf__l2_regularization": [0.0, 0.1, 0.5],
    "clf__min_samples_leaf": [10, 20, 30],
}


def _preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    # Dense OHE: HistGradientBoostingClassifier does not accept sparse X.
    # RandomForest works with dense as well.
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False),
            cat_cols,
        ))
    return ColumnTransformer(transformers=transformers)


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


def _col_importances(prep, clf, num_cols: list[str], cat_cols: list[str]) -> dict[str, float]:
    """Aggregate post-OHE feature importances back to each original column name.

    ColumnTransformer names features as "num__<col>" and "cat__<col>_<value>".
    Summing across OHE dummies gives each original column a single importance score.
    """
    names = prep.get_feature_names_out()
    imp = np.asarray(clf.feature_importances_, dtype=float)
    col_imp: dict[str, float] = {}
    for feat_name, feat_imp in zip(names, imp):
        orig = None
        if feat_name.startswith("num__"):
            orig = feat_name[5:]
        elif feat_name.startswith("cat__"):
            remainder = feat_name[5:]
            # Match the longest column name that is a prefix (handles underscores in names).
            for c in sorted(cat_cols, key=len, reverse=True):
                if remainder.startswith(c + "_") or remainder == c:
                    orig = c
                    break
        if orig:
            col_imp[orig] = col_imp.get(orig, 0.0) + feat_imp
    return col_imp


def _feature_selection_comparison(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
    *,
    top_k: int = 6,
    seed: int = 42,
) -> dict:
    """Select top-k original features by aggregated RF importance; retrain and compare.

    This demonstrates that feature importance scores are actionable: the model
    retrained on the reduced feature set should preserve most predictive power,
    validating that low-importance features contribute marginally.
    """
    clf = pipe.named_steps["clf"]
    prep = pipe.named_steps["prep"]
    col_imp = _col_importances(prep, clf, num_cols, cat_cols)
    sorted_cols = sorted(col_imp.items(), key=lambda x: -x[1])

    top_cols = {c for c, _ in sorted_cols[:top_k]}
    top_num = [c for c in num_cols if c in top_cols]
    top_cat = [c for c in cat_cols if c in top_cols]

    pipe_small = Pipeline([
        ("prep", _preprocess(top_num, top_cat)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )),
    ])
    pipe_small.fit(X_train[top_num + top_cat], y_train)
    pred_small = pipe_small.predict(X_test[top_num + top_cat])
    reduced_metrics = _metrics(y_test, pred_small, n_train=len(X_train), n_test=len(X_test))

    return {
        "top_k": top_k,
        "top_features": [
            {"feature": c, "importance": round(float(col_imp[c]), 4)}
            for c, _ in sorted_cols[:top_k]
        ],
        "all_feature_importances": [
            {"feature": c, "importance": round(float(v), 4)}
            for c, v in sorted_cols
        ],
        "reduced_metrics": reduced_metrics,
    }


def _tune_pipe(
    pipe: Pipeline,
    param_dist: dict,
    X_train,
    y_train,
    *,
    n_iter: int = 12,
    cv: int = 3,
    seed: int = 42,
) -> tuple[Pipeline, dict]:
    """Run RandomizedSearchCV; return (best_estimator, best_clf_params).

    Scoring on F1 rather than accuracy because the target is imbalanced (~23 % delayed).
    Cross-validation prevents the chosen params from overfitting to the single train split.
    """
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=seed,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    # Strip "clf__" prefix so the saved JSON maps cleanly to classifier kwargs.
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def _train_rf(
    X_train,
    X_test,
    y_train,
    y_test,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int,
    *,
    tune: bool = False,
    n_iter: int = 12,
) -> tuple[Pipeline, dict, np.ndarray, dict | None]:
    pipe = Pipeline(steps=[
        ("prep", _preprocess(num_cols, cat_cols)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )),
    ])
    best_params = None
    if tune:
        print("Running RandomizedSearchCV for Random Forest …")
        pipe, best_params = _tune_pipe(pipe, RF_PARAM_DIST, X_train, y_train, n_iter=n_iter, seed=seed)
        print(f"RF best params: {best_params}")
    else:
        pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = _metrics(y_test, pred, n_train=len(X_train), n_test=len(X_test))
    return pipe, metrics, pred, best_params


def _train_hgb(
    X_train,
    X_test,
    y_train,
    y_test,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int,
    *,
    tune: bool = False,
    n_iter: int = 12,
) -> tuple[Pipeline, dict, np.ndarray, dict | None]:
    pipe = Pipeline(steps=[
        ("prep", _preprocess(num_cols, cat_cols)),
        ("clf", HistGradientBoostingClassifier(
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
        )),
    ])
    best_params = None
    if tune:
        print("Running RandomizedSearchCV for Histogram Gradient Boosting …")
        # Disable early stopping during search: CV folds already provide the
        # hold-out estimate, and early_stopping + validation_fraction conflicts
        # with how RandomizedSearchCV splits data internally.
        pipe.named_steps["clf"].set_params(early_stopping=False, validation_fraction=None)
        pipe, best_params = _tune_pipe(pipe, HGB_PARAM_DIST, X_train, y_train, n_iter=n_iter, seed=seed)
        print(f"HGB best params: {best_params}")
    else:
        pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = _metrics(y_test, pred, n_train=len(X_train), n_test=len(X_test))
    return pipe, metrics, pred, best_params


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PROCESSED_FLIGHTS)
    p.add_argument("--model", choices=("rf", "hgb", "both"), default="both")
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV for hyperparameter tuning (slower; saves *_best_params.json).",
    )
    p.add_argument(
        "--tune-iter",
        type=int,
        default=12,
        help="Number of parameter settings sampled by RandomizedSearchCV.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Top-k original features to keep in the feature-selection comparison (RF only).",
    )
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
        pipe, metrics, pred, best_params = _train_rf(
            X_train, X_test, y_train, y_test, num_cols, cat_cols, args.seed,
            tune=args.tune, n_iter=args.tune_iter,
        )
        stem = args.out_dir / "random_forest"
        joblib.dump(pipe, stem.with_suffix(".joblib"))
        stem.with_name("random_forest_metrics.json").write_text(json.dumps(metrics, indent=2))
        stem.with_name("random_forest_features.json").write_text(json.dumps(feat_meta, indent=2))
        _save_importances(pipe, stem.with_name("random_forest_importances.json"))

        if best_params is not None:
            out_params = stem.with_name("random_forest_best_params.json")
            out_params.write_text(json.dumps(best_params, indent=2))
            print(f"Saved tuning results → {out_params.name}")

        # Always run feature-importance-based selection to show which features matter.
        print(f"Running feature importance selection (top {args.top_k} original features) …")
        sel = _feature_selection_comparison(
            pipe, X_train, X_test, y_train, y_test, num_cols, cat_cols,
            top_k=args.top_k, seed=args.seed,
        )
        out_sel = stem.with_name("random_forest_feature_selection.json")
        out_sel.write_text(json.dumps(sel, indent=2))
        print(
            f"Feature selection: full F1={metrics['f1']:.3f} → "
            f"top-{args.top_k} F1={sel['reduced_metrics']['f1']:.3f}"
        )

        results["random_forest"] = metrics
        print("=== Random Forest ===")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, digits=3))

    if args.model in ("hgb", "both"):
        pipe, metrics, pred, best_params = _train_hgb(
            X_train, X_test, y_train, y_test, num_cols, cat_cols, args.seed,
            tune=args.tune, n_iter=args.tune_iter,
        )
        stem = args.out_dir / "hist_gradient_boosting"
        joblib.dump(pipe, stem.with_name("hist_gradient_boosting.joblib"))
        stem.with_name("hist_gradient_boosting_metrics.json").write_text(json.dumps(metrics, indent=2))
        stem.with_name("hist_gradient_boosting_features.json").write_text(json.dumps(feat_meta, indent=2))
        _save_importances(pipe, stem.with_name("hist_gradient_boosting_importances.json"))

        if best_params is not None:
            out_params = stem.with_name("hist_gradient_boosting_best_params.json")
            out_params.write_text(json.dumps(best_params, indent=2))
            print(f"Saved tuning results → {out_params.name}")

        results["hist_gradient_boosting"] = metrics
        print("=== HistGradientBoosting ===")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, digits=3))

    if len(results) > 1:
        comp = args.out_dir / "tree_models_comparison.json"
        comp.write_text(json.dumps(results, indent=2))
        print(f"Wrote comparison → {comp}")


if __name__ == "__main__":
    main()
