from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_mlops.explain import compute_shap_values, explain_single
from loan_mlops.model import build_pipeline


@pytest.fixture
def trained_xgb() -> tuple:
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame(
        {
            "num_1": rng.randn(n),
            "num_2": rng.randn(n),
            "cat_1": rng.choice(["A", "B", "C"], size=n),
        }
    )
    y = pd.Series((1.5 * X["num_1"] + 0.5 * rng.randn(n) > 0).astype(int))

    pipeline = build_pipeline(
        X,
        y,
        model_type="xgboost",
        model_params={"n_estimators": 50, "max_depth": 3, "tree_method": "hist"},
    )
    pipeline.fit(X, y)
    return pipeline, X, y


def test_compute_shap_returns_one_value_per_feature(trained_xgb: tuple) -> None:
    pipeline, X, _ = trained_xgb
    values, names = compute_shap_values(pipeline, X.head(10), sample_size=None)
    assert values.values.shape[0] == 10
    assert values.values.shape[1] == len(names)


def test_explain_single_returns_expected_structure(trained_xgb: tuple) -> None:
    pipeline, X, _ = trained_xgb
    result = explain_single(pipeline, X.head(1), top_k=3)
    assert 0.0 <= result["default_probability"] <= 1.0
    assert isinstance(result["risk_factors"], list)
    assert isinstance(result["protective_factors"], list)


def test_explain_single_rejects_multiple_rows(trained_xgb: tuple) -> None:
    pipeline, X, _ = trained_xgb
    with pytest.raises(ValueError, match="exactly 1 row"):
        explain_single(pipeline, X.head(2))


def test_shap_values_correlate_with_predictions(trained_xgb: tuple) -> None:
    pipeline, X, _ = trained_xgb
    values, _ = compute_shap_values(pipeline, X.head(50), sample_size=None)
    proba = pipeline.predict_proba(X.head(50))[:, 1]
    shap_totals = values.values.sum(axis=1)
    corr = np.corrcoef(shap_totals, proba)[0, 1]
    assert corr > 0.5
