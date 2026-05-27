from __future__ import annotations

import logging
from functools import lru_cache

from sklearn.pipeline import Pipeline

from loan_mlops.api import settings
from loan_mlops.api.settings import get_settings
from loan_mlops.model import load_model

logger = logging.getLogger(__name__)


@lru_cache
def get_model() -> Pipeline:
    """Load the model once and cache it for the lifetime of the process."""
    settings = get_settings()
    logger.info("Loading model", extra={"model_name": settings.model_name})
    return load_model(model_dir=settings.models_dir, name=settings.model_name)