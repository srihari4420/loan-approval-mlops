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
from xgboost import XGBClassifier

from loan_mlops.features import build_preprocessor

logger = logging.getLogger(__name__)


def _scale_pos_weight(y: pd.Series) -> float:
    """XGBoost's preferred way to handle class imbalance."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / pos) if pos else 1.0


def build_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    model_params: dict[str, Any],
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_preprocessor(X)

    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=random_state, **model_params)
    elif model_type == "xgboost":
        clf = XGBClassifier(
            random_state=random_state,
            scale_pos_weight=_scale_pos_weight(y),
            n_jobs=-1,
            **model_params,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


# Kept for backward compat with tests written against the LogReg-only API.
def build_baseline_pipeline(
    X: pd.DataFrame, model_params: dict[str, Any], random_state: int = 42
) -> Pipeline:
    preprocessor = build_preprocessor(X)
    clf = LogisticRegression(random_state=random_state, **model_params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    logger.info("CV starting", extra={"folds": cv_folds, "scoring": scoring})
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)
    result = {
        "cv_score_mean": float(scores.mean()),
        "cv_score_std": float(scores.std()),
        "cv_score_folds": scores.tolist(),
    }
    logger.info("CV done", extra={"mean": result["cv_score_mean"], "std": result["cv_score_std"]})
    return result


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    metrics = {
        "test_auc": float(roc_auc_score(y_test, proba)),
        "test_avg_precision": float(average_precision_score(y_test, proba)),
    }
    logger.info("Test metrics", extra=metrics)
    logger.info("Classification report:\n" + classification_report(y_test, pred, zero_division=0))
    return metrics


def save_model(pipeline: Pipeline, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("Model saved", extra={"path": str(path)})
    return path


def load_model(model_dir: Path, name: str) -> Pipeline:
    path = model_dir / f"{name}.joblib"
    logger.info("Loading model", extra={"path": str(path)})
    return joblib.load(path)