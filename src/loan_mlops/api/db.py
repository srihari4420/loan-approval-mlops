"""Database setup and audit-log model.

Uses SQLAlchemy 2.0 declarative style. SQLite for dev, Postgres for prod —
the connection string is the only thing that changes."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from loan_mlops.api.settings import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    """One row per scored application.

    Stored fields are the minimum a credit risk auditor would need to reconstruct
    a decision: who scored what, when, with which model, and why."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    correlation_id = Column(String(64), nullable=False, index=True)
    application_id = Column(String(64), nullable=True)
    model_version = Column(String(64), nullable=False)
    cohort = Column(String(16), nullable=False)  # "champion" or "challenger"
    default_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    decision = Column(String(16), nullable=False)
    risk_factors = Column(JSON, nullable=False, default=list)
    protective_factors = Column(JSON, nullable=False, default=list)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("ix_predictions_model_created", "model_version", "created_at"),
        Index("ix_predictions_cohort_created", "cohort", "created_at"),
    )


_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        logger.info("Creating DB engine", extra={"url": settings.database_url})
        _engine = create_engine(settings.database_url, echo=False, future=True)
        Base.metadata.create_all(_engine)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_for_tests() -> None:
    """Drop and recreate tables. Tests only."""
    global _engine, _SessionFactory
    if _engine is not None:
        Base.metadata.drop_all(_engine)
        Base.metadata.create_all(_engine)
    _SessionFactory = None
