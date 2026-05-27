from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOAN_API_", env_file=".env")

    # Model loading
    models_dir: Path = Path("models")
    model_name: str = "xgboost_v1"

    # Service
    service_name: str = "loan-scoring"
    log_level: str = "INFO"
    log_json: bool = False

    # Behaviour
    decision_threshold: float = 0.5  # probability above which we reject
    enable_explanations: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached so it's only constructed once per process."""
    return Settings()