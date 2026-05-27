"""Tests for data loading, validation, and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from loan_mlops.data import (
    ApplicationTrainSchema,
    clean_sentinels,
    drop_high_missing,
    load_clean,
    load_raw,
)

# ---------- Fixtures ----------


@pytest.fixture
def valid_raw_df() -> pd.DataFrame:
    """A minimal DataFrame matching the production schema."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003, 100004],
            "TARGET": [0, 1, 0, 1],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0, 150000.0, 50000.0],
            "AMT_CREDIT": [500000.0, 1000000.0, 750000.0, 300000.0],
            "DAYS_BIRTH": [-12000, -15000, -10000, -20000],
            "DAYS_EMPLOYED": [-2000, 365243, -5000, -1000],
        }
    )


# ---------- Schema validation tests ----------


def test_schema_accepts_valid_data(valid_raw_df: pd.DataFrame) -> None:
    """Schema should validate clean data without errors."""
    validated = ApplicationTrainSchema.validate(valid_raw_df, lazy=True)
    assert len(validated) == len(valid_raw_df)


def test_schema_rejects_invalid_target(valid_raw_df: pd.DataFrame) -> None:
    """TARGET must be 0 or 1 — anything else should fail."""
    bad_df = valid_raw_df.copy()
    bad_df.loc[0, "TARGET"] = 2
    with pytest.raises(pandera.errors.SchemaErrors):
        ApplicationTrainSchema.validate(bad_df, lazy=True)


def test_schema_rejects_duplicate_ids(valid_raw_df: pd.DataFrame) -> None:
    """SK_ID_CURR must be unique."""
    bad_df = valid_raw_df.copy()
    bad_df.loc[0, "SK_ID_CURR"] = bad_df.loc[1, "SK_ID_CURR"]
    with pytest.raises(pandera.errors.SchemaErrors):
        ApplicationTrainSchema.validate(bad_df, lazy=True)


def test_schema_rejects_negative_income(valid_raw_df: pd.DataFrame) -> None:
    """AMT_INCOME_TOTAL must be non-negative."""
    bad_df = valid_raw_df.copy()
    bad_df.loc[0, "AMT_INCOME_TOTAL"] = -100.0
    with pytest.raises(pandera.errors.SchemaErrors):
        ApplicationTrainSchema.validate(bad_df, lazy=True)


# ---------- Cleaning tests ----------


def test_clean_sentinels_replaces_with_nan(valid_raw_df: pd.DataFrame) -> None:
    """The sentinel value 365243 should be replaced with NaN."""
    result = clean_sentinels(valid_raw_df, sentinel_value=365243)
    assert pd.isna(result.loc[1, "DAYS_EMPLOYED"])
    assert result.loc[0, "DAYS_EMPLOYED"] == -2000  # unaffected


def test_clean_sentinels_adds_flag(valid_raw_df: pd.DataFrame) -> None:
    """A binary flag column should mark which rows had the sentinel."""
    result = clean_sentinels(valid_raw_df, sentinel_value=365243)
    assert "DAYS_EMPLOYED_ANOM" in result.columns
    assert result["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0, 0]


def test_clean_sentinels_preserves_row_count(valid_raw_df: pd.DataFrame) -> None:
    """Cleaning must not add or remove rows."""
    result = clean_sentinels(valid_raw_df, sentinel_value=365243)
    assert len(result) == len(valid_raw_df)


def test_clean_sentinels_does_not_mutate_input(valid_raw_df: pd.DataFrame) -> None:
    """Cleaning must not modify the original DataFrame (immutability)."""
    original = valid_raw_df.copy()
    _ = clean_sentinels(valid_raw_df, sentinel_value=365243)
    pd.testing.assert_frame_equal(valid_raw_df, original)


# ---------- Drop high-missing tests ----------


def test_drop_high_missing_removes_sparse_columns() -> None:
    """Columns with >threshold missing values should be dropped."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "mostly_missing": [1.0, np.nan, np.nan, np.nan],  # 75% missing
            "mostly_present": [1.0, 2.0, 3.0, np.nan],  # 25% missing
        }
    )
    result = drop_high_missing(df, threshold=0.5)
    assert "mostly_missing" not in result.columns
    assert "mostly_present" in result.columns


def test_drop_high_missing_protects_target_and_id() -> None:
    """Even if TARGET/ID are mostly missing, they must not be dropped (would never happen)."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, np.nan, np.nan, np.nan],
            "TARGET": [0, np.nan, np.nan, np.nan],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = drop_high_missing(df, threshold=0.5)
    assert "TARGET" in result.columns
    assert "SK_ID_CURR" in result.columns


def test_drop_high_missing_zero_threshold_drops_everything_with_any_missing() -> None:
    """Edge case: threshold=0 drops every column with at least one NaN."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "no_missing": [1.0, 2.0, 3.0],
            "one_missing": [1.0, 2.0, np.nan],
        }
    )
    result = drop_high_missing(df, threshold=0.0)
    assert "no_missing" in result.columns
    assert "one_missing" not in result.columns


# ---------- Integration tests (filesystem) ----------


def test_load_raw_reads_csv_and_validates(tmp_path, valid_raw_df: pd.DataFrame) -> None:
    """load_raw should read a CSV from disk and pass schema validation."""
    csv_path = tmp_path / "test.csv"
    valid_raw_df.to_csv(csv_path, index=False)

    result = load_raw(csv_path, validate=True)
    assert len(result) == len(valid_raw_df)
    assert set(result.columns) == set(valid_raw_df.columns)


def test_load_raw_skip_validation(tmp_path, valid_raw_df: pd.DataFrame) -> None:
    """load_raw with validate=False should not raise even on malformed data."""
    csv_path = tmp_path / "test.csv"
    # Inject invalid TARGET — schema would reject this if validation ran
    bad_df = valid_raw_df.copy()
    bad_df.loc[0, "TARGET"] = 5
    bad_df.to_csv(csv_path, index=False)

    result = load_raw(csv_path, validate=False)
    assert len(result) == len(bad_df)


def test_load_raw_raises_on_invalid_data(tmp_path, valid_raw_df: pd.DataFrame) -> None:
    """load_raw with validate=True should raise on invalid data."""
    csv_path = tmp_path / "test.csv"
    bad_df = valid_raw_df.copy()
    bad_df.loc[0, "TARGET"] = 5
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(pandera.errors.SchemaErrors):
        load_raw(csv_path, validate=True)


def test_load_clean_full_pipeline(tmp_path, valid_raw_df: pd.DataFrame) -> None:
    """load_clean integration: load + clean sentinels + drop high-missing."""
    # Add a high-missing column to test the drop step
    df = valid_raw_df.copy()
    df["very_sparse"] = [None, None, None, 1.0]  # 75% missing

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    result = load_clean(
        raw_path=csv_path,
        sentinel_value=365243,
        missing_threshold=0.5,
        validate=True,
    )
    # Sentinel should be replaced
    assert pd.isna(result.loc[1, "DAYS_EMPLOYED"])
    # High-missing column should be dropped
    assert "very_sparse" not in result.columns
    # Anomaly flag should be added
    assert "DAYS_EMPLOYED_ANOM" in result.columns
    # Row count preserved
    assert len(result) == len(valid_raw_df)
