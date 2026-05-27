from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOAN_API_", env_file=".env")

    # Champion (default model)
    models_dir: Path = Path("models")
    model_name: str = "xgboost_v1"

    # Challenger
    challenger_model_name: str | None = "logistic_regression_v1"
    challenger_traffic_pct: float = 0.10  # 10% of traffic to challenger

    # Service
    service_name: str = "loan-scoring"
    log_level: str = "INFO"
    log_json: bool = False

    # Behaviour
    decision_threshold: float = 0.5
    enable_explanations: bool = True

    # Database
    database_url: str = "sqlite:///predictions.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
