"""Tests for the model module: pipeline construction, training, evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from loan_mlops.model import (
    build_baseline_pipeline,
    cross_validate,
    evaluate,
    load_model,
    save_model,
)


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification data with a learnable signal."""
    rng = np.random.RandomState(42)
    n = 500

    # Make sure there's actual signal so models can learn
    X = pd.DataFrame(
        {
            "num_1": rng.randn(n),
            "num_2": rng.randn(n),
            "cat_1": rng.choice(["A", "B", "C"], size=n),
        }
    )
    # Target depends on num_1 + noise
    logits = 1.5 * X["num_1"] + 0.5 * rng.randn(n)
    y = pd.Series((logits > 0).astype(int))

    return X, y


@pytest.fixture
def default_model_params() -> dict:
    return {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "C": 1.0,
    }


def test_build_baseline_pipeline_returns_sklearn_pipeline(
    synthetic_data: tuple, default_model_params: dict
) -> None:
    """The output of build_baseline_pipeline must be a sklearn Pipeline."""
    X, _ = synthetic_data
    pipeline = build_baseline_pipeline(X, default_model_params)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_predicts_probabilities_in_valid_range(
    synthetic_data: tuple, default_model_params: dict
) -> None:
    """predict_proba output must be valid probabilities (in [0, 1], summing to 1)."""
    X, y = synthetic_data
    pipeline = build_baseline_pipeline(X, default_model_params)
    pipeline.fit(X, y)
    probas = pipeline.predict_proba(X)

    assert probas.shape == (len(X), 2)
    assert ((probas >= 0) & (probas <= 1)).all()
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_pipeline_learns_signal(synthetic_data: tuple, default_model_params: dict) -> None:
    """On data with real signal, the model must outperform random guessing."""
    from sklearn.metrics import roc_auc_score

    X, y = synthetic_data
    pipeline = build_baseline_pipeline(X, default_model_params)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]

    # Random guessing = 0.5; a working model on synthetic signal should be much higher
    assert roc_auc_score(y, proba) > 0.85


def test_cross_validate_returns_expected_keys(
    synthetic_data: tuple, default_model_params: dict
) -> None:
    """cross_validate result schema must be stable."""
    X, y = synthetic_data
    pipeline = build_baseline_pipeline(X, default_model_params)
    result = cross_validate(pipeline, X, y, cv_folds=3)

    assert "cv_score_mean" in result
    assert "cv_score_std" in result
    assert "cv_score_folds" in result
    assert len(result["cv_score_folds"]) == 3
    assert 0.0 <= result["cv_score_mean"] <= 1.0


def test_evaluate_returns_test_metrics(synthetic_data: tuple, default_model_params: dict) -> None:
    """evaluate must return both AUC and average precision."""
    X, y = synthetic_data
    pipeline = build_baseline_pipeline(X, default_model_params)
    pipeline.fit(X, y)
    metrics = evaluate(pipeline, X, y)

    assert "test_auc" in metrics
    assert "test_avg_precision" in metrics
    assert 0.0 <= metrics["test_auc"] <= 1.0


def test_save_and_load_round_trip(
    tmp_path: Path, synthetic_data: tuple, default_model_params: dict
) -> None:
    """A saved model must load back identically and produce the same predictions."""
    X, y = synthetic_data
    pipeline = build_baseline_pipeline(X, default_model_params)
    pipeline.fit(X, y)
    original_preds = pipeline.predict_proba(X)[:, 1]

    save_model(pipeline, output_dir=tmp_path, name="test_model")
    loaded = load_model(model_dir=tmp_path, name="test_model")
    loaded_preds = loaded.predict_proba(X)[:, 1]

    np.testing.assert_array_almost_equal(original_preds, loaded_preds)
