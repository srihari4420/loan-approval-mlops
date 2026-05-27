"""End-to-end integration tests.

Real model, real DB, real HTTP layer. Catches wiring bugs that unit tests with
mocks miss — column mismatches, schema drift, dependency override gaps."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from loan_mlops.api import db as db_module
from loan_mlops.api.app import create_app
from loan_mlops.api.db import Base, PredictionRecord

REAL_MODEL_PATH = Path("models/xgboost_v1.joblib")

requires_model = pytest.mark.skipif(
    not REAL_MODEL_PATH.exists(),
    reason="Real model not available — train it locally to run integration tests",
)


@pytest.fixture
def integration_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Shared in-memory SQLite — using `StaticPool` so every connection sees the same DB.
    # Without this, each new connection gets its own empty in-memory DB and tables vanish.
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_module, "_engine", engine)
    monkeypatch.setattr(db_module, "_SessionFactory", factory)

    app = create_app()
    return TestClient(app)


@pytest.fixture
def realistic_application() -> dict:
    return {
        "application_id": "integration-001",
        "CODE_GENDER": "M",
        "DAYS_BIRTH": -12000,
        "CNT_CHILDREN": 1,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "AMT_INCOME_TOTAL": 180000.0,
        "DAYS_EMPLOYED": -2000,
        "OCCUPATION_TYPE": "Core staff",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 24000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "EXT_SOURCE_1": 0.7,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.5,
    }


@requires_model
def test_end_to_end_prediction_with_real_model(
    integration_client: TestClient, realistic_application: dict
) -> None:
    r = integration_client.post("/predict", json=realistic_application)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in {"approve", "decline"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["model_version"] in {"xgboost_v1", "logistic_regression_v1"}
    assert isinstance(body["risk_factors"], list)
    assert len(body["correlation_id"]) > 10


@requires_model
def test_audit_log_written_on_prediction(
    integration_client: TestClient, realistic_application: dict
) -> None:
    integration_client.post("/predict", json=realistic_application)

    with db_module.session_scope() as session:
        rows = session.query(PredictionRecord).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.application_id == "integration-001"
        assert row.cohort in {"champion", "challenger"}
        assert row.latency_ms > 0


@requires_model
def test_same_application_id_routes_to_same_cohort(
    integration_client: TestClient, realistic_application: dict
) -> None:
    responses = []
    for _ in range(10):
        r = integration_client.post("/predict", json=realistic_application)
        responses.append(r.json()["model_version"])
    assert len(set(responses)) == 1, f"got mixed models: {set(responses)}"


@requires_model
def test_stats_aggregates_across_cohorts(
    integration_client: TestClient, realistic_application: dict
) -> None:
    for i in range(50):
        app = dict(realistic_application)
        app["application_id"] = f"integration-batch-{i}"
        r = integration_client.post("/predict", json=app)
        assert r.status_code == 200

    stats = integration_client.get("/stats").json()
    total = sum(c["predictions"] for c in stats["by_cohort"])
    assert total == 50


@requires_model
def test_correlation_id_round_trips_with_real_model(
    integration_client: TestClient, realistic_application: dict
) -> None:
    cid = "integration-trace-xyz"
    r = integration_client.post(
        "/predict",
        json=realistic_application,
        headers={"x-correlation-id": cid},
    )
    assert r.headers["x-correlation-id"] == cid

    with db_module.session_scope() as session:
        row = session.query(PredictionRecord).first()
        assert row is not None
        assert row.correlation_id == cid
