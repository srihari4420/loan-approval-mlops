"""Data loading, validation, and basic cleaning.

Validates schema on load using Pandera — catches data drift and bad files early.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

logger = logging.getLogger(__name__)


# Schema validation — declares what the raw data MUST look like.
# If the schema breaks, training fails loudly instead of producing a broken model.
class ApplicationTrainSchema(pa.DataFrameModel):
    """Pandera schema for application_train.csv.

    Only validates the columns we depend on. Lets others pass through unvalidated.
    """

    SK_ID_CURR: Series[int] = pa.Field(unique=True, nullable=False)
    TARGET: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    AMT_INCOME_TOTAL: Series[float] = pa.Field(ge=0, nullable=True)
    AMT_CREDIT: Series[float] = pa.Field(ge=0, nullable=True)
    DAYS_BIRTH: Series[int] = pa.Field(le=0, nullable=False)  # negative = days before now
    DAYS_EMPLOYED: Series[int] = pa.Field(nullable=True)

    class Config:
        strict = False  # extra columns are allowed
        coerce = True


def load_raw(path: Path, validate: bool = True) -> pd.DataFrame:
    """Load raw CSV, optionally validating schema.

    Args:
        path: Path to CSV file.
        validate: If True (default), validate against ApplicationTrainSchema.

    Returns:
        Loaded DataFrame.

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    logger.info("Loading raw data", extra={"path": str(path)})
    df = pd.read_csv(path)
    logger.info(
        "Loaded data",
        extra={"rows": len(df), "columns": df.shape[1]},
    )

    if validate:
        logger.info("Validating schema with Pandera")
        df = ApplicationTrainSchema.validate(df, lazy=True)
        logger.info("Schema validation passed")

    return df


def clean_sentinels(df: pd.DataFrame, sentinel_value: int = 365243) -> pd.DataFrame:
    """Replace DAYS_EMPLOYED sentinel with NaN and add anomaly flag."""
    df = df.copy()
    mask = df["DAYS_EMPLOYED"] == sentinel_value
    n_sentinel = int(mask.sum())
    logger.info(
        "Replacing DAYS_EMPLOYED sentinel values",
        extra={"count": n_sentinel, "sentinel_value": sentinel_value},
    )
    df["DAYS_EMPLOYED_ANOM"] = mask.astype(int)
    df.loc[mask, "DAYS_EMPLOYED"] = np.nan
    return df


def drop_high_missing(
    df: pd.DataFrame,
    threshold: float,
    protected: tuple[str, ...] = ("TARGET", "SK_ID_CURR"),
) -> pd.DataFrame:
    """Drop columns where missing fraction exceeds threshold. Protect target and ID."""
    missing_pct = df.isnull().mean()
    to_drop = [col for col in missing_pct[missing_pct > threshold].index if col not in protected]
    logger.info(
        "Dropping high-missing columns",
        extra={"count": len(to_drop), "threshold": threshold},
    )
    return df.drop(columns=to_drop)


def load_clean(
    raw_path: Path,
    sentinel_value: int = 365243,
    missing_threshold: float = 0.6,
    validate: bool = True,
) -> pd.DataFrame:
    """Full data prep: load → validate → clean sentinels → drop high-missing."""
    df = load_raw(raw_path, validate=validate)
    df = clean_sentinels(df, sentinel_value=sentinel_value)
    df = drop_high_missing(df, threshold=missing_threshold)
    logger.info("Final cleaned shape", extra={"shape": str(df.shape)})
    return df
