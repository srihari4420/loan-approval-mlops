"""Feature engineering: encoding, imputation, scaling.

All preprocessing lives inside a sklearn ColumnTransformer so it's fit on train
folds only — no data leakage during cross-validation.
"""

from __future__ import annotations

import logging

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

HIGH_CARDINALITY_THRESHOLD = 10


def split_xy(df: pd.DataFrame, target_col: str, id_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features and target, dropping the ID column."""
    y = df[target_col]
    X = df.drop(columns=[target_col, id_col])
    return X, y


def classify_columns(X: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into numeric, low-cardinality, high-cardinality categorical."""
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_all = X.select_dtypes(include=["object"]).columns.tolist()

    low_card = [c for c in cat_all if X[c].nunique() < HIGH_CARDINALITY_THRESHOLD]
    high_card = [c for c in cat_all if X[c].nunique() >= HIGH_CARDINALITY_THRESHOLD]

    logger.info(
        "Column classification",
        extra={
            "numeric": len(numeric),
            "low_card_categorical": len(low_card),
            "high_card_categorical": len(high_card),
        },
    )
    return {"numeric": numeric, "low_card": low_card, "high_card": high_card}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer wired up correctly for each column type."""
    cols = classify_columns(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    low_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", TargetEncoder(handle_unknown="value", handle_missing="value")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, cols["numeric"]),
            ("low_cat", low_card_pipeline, cols["low_card"]),
            ("high_cat", high_card_pipeline, cols["high_card"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
