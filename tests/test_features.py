"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from loan_mlops.features import (
    HIGH_CARDINALITY_THRESHOLD,
    build_preprocessor,
    classify_columns,
    split_xy,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small mixed-type DataFrame for testing."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(100),
            "TARGET": rng.randint(0, 2, size=100),
            "num_feat_1": rng.randn(100),
            "num_feat_2": rng.randn(100) * 100,
            "low_card_cat": rng.choice(["A", "B", "C"], size=100),
            "high_card_cat": rng.choice([f"cat_{i}" for i in range(20)], size=100),
        }
    )


def test_split_xy_separates_target() -> None:
    """split_xy returns target as Series and features as DataFrame."""
    df = pd.DataFrame(
        {"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "f1": [10, 20, 30]}
    )
    X, y = split_xy(df, target_col="TARGET", id_col="SK_ID_CURR")
    assert list(y) == [0, 1, 0]
    assert "TARGET" not in X.columns
    assert "SK_ID_CURR" not in X.columns
    assert "f1" in X.columns


def test_classify_columns_separates_by_cardinality(sample_df: pd.DataFrame) -> None:
    """Categorical columns split correctly by cardinality threshold."""
    X = sample_df.drop(columns=["TARGET", "SK_ID_CURR"])
    cols = classify_columns(X)

    assert "num_feat_1" in cols["numeric"]
    assert "num_feat_2" in cols["numeric"]
    assert "low_card_cat" in cols["low_card"]
    assert "high_card_cat" in cols["high_card"]


def test_classify_columns_uses_threshold_constant() -> None:
    """Cardinality boundary respects the module-level constant."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        {
            "exactly_at_threshold": rng.choice(
                [f"c{i}" for i in range(HIGH_CARDINALITY_THRESHOLD)], size=50
            ),
            "below_threshold": rng.choice(
                [f"c{i}" for i in range(HIGH_CARDINALITY_THRESHOLD - 1)], size=50
            ),
        }
    )
    cols = classify_columns(X)
    # Exactly-at-threshold should land in high_card (>= threshold)
    assert "exactly_at_threshold" in cols["high_card"]
    assert "below_threshold" in cols["low_card"]


def test_build_preprocessor_returns_column_transformer(sample_df: pd.DataFrame) -> None:
    """The preprocessor must be a ColumnTransformer for sklearn Pipeline compatibility."""
    X = sample_df.drop(columns=["TARGET", "SK_ID_CURR"])
    preprocessor = build_preprocessor(X)
    assert isinstance(preprocessor, ColumnTransformer)


def test_build_preprocessor_handles_missing_values(sample_df: pd.DataFrame) -> None:
    """Preprocessor must successfully fit_transform data containing NaNs."""
    X = sample_df.drop(columns=["TARGET", "SK_ID_CURR"]).copy()
    y = sample_df["TARGET"]
    # Inject NaNs
    X.loc[0, "num_feat_1"] = np.nan
    X.loc[1, "low_card_cat"] = np.nan

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X, y)

    # No NaNs in the output — imputation worked
    assert not np.isnan(transformed).any()