from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from loan_mlops.api.app import create_app
from loan_mlops.api.dependencies import get_expected_columns, get_model


@pytest.fixture
def fake_model() -> MagicMock:
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.58, 0.42]])
    return m


@pytest.fixture
def fake_columns() -> tuple[str, ...]:
    return ("CODE_GENDER", "DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT")


@pytest.fixture
def client(fake_model: MagicMock, fake_columns: tuple[str, ...]) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_model] = lambda: fake_model
    app.dependency_overrides[get_expected_columns] = lambda: fake_columns
    return TestClient(app)


@pytest.fixture
def valid_application() -> dict:
    return {
        "application_id": "test-001",
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


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response(client: TestClient, valid_application: dict) -> None:
    r = client.post("/predict", json=valid_application)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in {"approve", "decline"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["application_id"] == "test-001"


def test_predict_decision_matches_threshold(
    client: TestClient, valid_application: dict, fake_model: MagicMock
) -> None:
    fake_model.predict_proba.return_value = np.array([[0.58, 0.42]])
    r = client.post("/predict", json=valid_application)
    assert r.json()["decision"] == "approve"

    fake_model.predict_proba.return_value = np.array([[0.25, 0.75]])
    r = client.post("/predict", json=valid_application)
    assert r.json()["decision"] == "decline"


def test_predict_rejects_missing_required_field(
    client: TestClient, valid_application: dict
) -> None:
    del valid_application["CODE_GENDER"]
    r = client.post("/predict", json=valid_application)
    assert r.status_code == 422


def test_predict_rejects_invalid_enum(client: TestClient, valid_application: dict) -> None:
    valid_application["CODE_GENDER"] = "Z"
    r = client.post("/predict", json=valid_application)
    assert r.status_code == 422


def test_predict_rejects_negative_income(
    client: TestClient, valid_application: dict
) -> None:
    valid_application["AMT_INCOME_TOTAL"] = -1000.0
    r = client.post("/predict", json=valid_application)
    assert r.status_code == 422


def test_predict_rejects_extra_fields(client: TestClient, valid_application: dict) -> None:
    valid_application["nefarious_extra_field"] = "drop_table"
    r = client.post("/predict", json=valid_application)
    assert r.status_code == 422


def test_correlation_id_returned_in_response(
    client: TestClient, valid_application: dict
) -> None:
    r = client.post("/predict", json=valid_application)
    assert "x-correlation-id" in r.headers


def test_correlation_id_respects_inbound_header(
    client: TestClient, valid_application: dict
) -> None:
    inbound = "trace-abc-123"
    r = client.post("/predict", json=valid_application, headers={"x-correlation-id": inbound})
    assert r.headers["x-correlation-id"] == inbound


def test_scoring_failure_returns_500(
    client: TestClient, valid_application: dict, fake_model: MagicMock
) -> None:
    fake_model.predict_proba.side_effect = RuntimeError("model crashed")
    r = client.post("/predict", json=valid_application)
    assert r.status_code == 500
