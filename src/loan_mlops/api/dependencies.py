"""Loaded models, expected columns, and shared resources."""

from __future__ import annotations

import logging
from functools import lru_cache

from sklearn.pipeline import Pipeline

from loan_mlops.api.settings import get_settings
from loan_mlops.model import load_model

logger = logging.getLogger(__name__)


@lru_cache
def get_champion() -> Pipeline:
    s = get_settings()
    logger.info("Loading champion model", extra={"model_name": s.model_name})
    return load_model(model_dir=s.models_dir, name=s.model_name)


@lru_cache
def get_challenger() -> Pipeline | None:
    s = get_settings()
    if not s.challenger_model_name:
        return None
    try:
        logger.info("Loading challenger model", extra={"model_name": s.challenger_model_name})
        return load_model(model_dir=s.models_dir, name=s.challenger_model_name)
    except FileNotFoundError:
        logger.warning(
            "Challenger model not found, disabling",
            extra={"model_name": s.challenger_model_name},
        )
        return None


@lru_cache
def get_expected_columns() -> tuple[str, ...]:
    """Champion's expected columns. Challenger uses the same preprocessor in our setup."""
    pipeline = get_champion()
    pre = pipeline.named_steps["preprocessor"]
    cols: list[str] = []
    for _, _, columns in pre.transformers_:
        cols.extend(columns)
    return tuple(cols)


# Keep `get_model` as an alias so existing /predict callers still work
get_model = get_champion
