"""Loaded model and the column schema it expects."""

from __future__ import annotations

import logging
from functools import lru_cache

from sklearn.pipeline import Pipeline

from loan_mlops.api.settings import get_settings
from loan_mlops.model import load_model

logger = logging.getLogger(__name__)


@lru_cache
def get_model() -> Pipeline:
    settings = get_settings()
    logger.info("Loading model", extra={"model_name": settings.model_name})
    return load_model(model_dir=settings.models_dir, name=settings.model_name)


@lru_cache
def get_expected_columns() -> tuple[str, ...]:
    """Columns the trained pipeline expects, in the order it expects them.

    Pulled from the ColumnTransformer at startup so we know what to pad.
    """
    pipeline = get_model()
    pre = pipeline.named_steps["preprocessor"]
    cols: list[str] = []
    for _, _, columns in pre.transformers_:
        cols.extend(columns)
    return tuple(cols)
