from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from loan_mlops.api import db as db_module
from loan_mlops.api.db import Base, PredictionRecord, session_scope


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch: pytest.MonkeyPatch):
    """Replace the module-level engine with an in-memory SQLite for each test."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr(db_module, "_engine", engine)
    monkeypatch.setattr(db_module, "_SessionFactory", factory)
    yield
    Base.metadata.drop_all(engine)


def test_session_scope_commits_on_success() -> None:
    with session_scope() as session:
        session.add(_sample_record())

    with session_scope() as session:
        assert session.query(PredictionRecord).count() == 1


def test_session_scope_rolls_back_on_exception() -> None:
    with pytest.raises(RuntimeError), session_scope() as session:
        session.add(_sample_record())
        raise RuntimeError("simulated failure")

    with session_scope() as session:
        assert session.query(PredictionRecord).count() == 0


def test_prediction_record_stores_all_fields() -> None:
    record = _sample_record()
    with session_scope() as session:
        session.add(record)

    with session_scope() as session:
        row = session.query(PredictionRecord).one()
        assert row.correlation_id == "corr-1"
        assert row.cohort == "champion"
        assert row.decision == "approve"
        assert row.risk_factors == [{"feature": "EXT_SOURCE_2", "shap": 0.1}]
        assert row.latency_ms == 42.5
        assert row.created_at is not None


def test_prediction_record_requires_non_null_fields(session_factory_cleanup: None = None) -> None:
    """A record without correlation_id must fail to insert."""
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError), session_scope() as session:
        session.add(
            PredictionRecord(
                application_id="app-1",
                model_version="m1",
                cohort="champion",
                default_probability=0.1,
                threshold=0.5,
                decision="approve",
                risk_factors=[],
                protective_factors=[],
                latency_ms=10.0,
            )
        )


def _sample_record() -> PredictionRecord:
    return PredictionRecord(
        correlation_id="corr-1",
        application_id="app-1",
        model_version="xgboost_v1",
        cohort="champion",
        default_probability=0.12,
        threshold=0.5,
        decision="approve",
        risk_factors=[{"feature": "EXT_SOURCE_2", "shap": 0.1}],
        protective_factors=[],
        latency_ms=42.5,
        created_at=datetime.now(UTC),
    )
