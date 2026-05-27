"""Model building, evaluation, persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from loan_mlops.features import build_preprocessor

logger = logging.getLogger(__name__)


def build_baseline_pipeline(
    X: pd.DataFrame, model_params: dict[str, Any], random_state: int = 42
) -> Pipeline:
    """Build preprocessor + LogisticRegression pipeline."""
    preprocessor = build_preprocessor(X)
    classifier = LogisticRegression(random_state=random_state, **model_params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    logger.info(
        "Starting cross-validation",
        extra={"folds": cv_folds, "scoring": scoring},
    )
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)

    result = {
        "cv_score_mean": float(scores.mean()),
        "cv_score_std": float(scores.std()),
        "cv_score_folds": scores.tolist(),
    }
    logger.info(
        "Cross-validation complete",
        extra={
            "mean": result["cv_score_mean"],
            "std": result["cv_score_std"],
        },
    )
    return result


def evaluate(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate a fitted pipeline on held-out test data."""
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "test_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "test_avg_precision": float(average_precision_score(y_test, y_pred_proba)),
    }
    logger.info("Test evaluation", extra=metrics)
    logger.info(
        "Classification report:\n" + classification_report(y_test, y_pred, zero_division=0)
    )
    return metrics


def save_model(pipeline: Pipeline, output_dir: Path, name: str) -> Path:
    """Save a fitted pipeline to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("Saved model", extra={"path": str(path)})
    return path


def load_model(model_dir: Path, name: str) -> Pipeline:
    """Load a saved pipeline from disk."""
    path = model_dir / f"{name}.joblib"
    logger.info("Loading model", extra={"path": str(path)})
    pipeline = joblib.load(path)
    return pipeline  # type: ignore[no-any-return]