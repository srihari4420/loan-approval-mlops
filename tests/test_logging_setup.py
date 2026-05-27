"""Tests for structured logging setup."""

from __future__ import annotations

import json
import logging
import uuid
from io import StringIO

import pytest

from loan_mlops.logging_setup import (
    JsonFormatter,
    correlation_id_var,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_correlation_id() -> None:
    """Each test starts with a clean correlation ID context."""
    correlation_id_var.set("")


def test_set_correlation_id_generates_uuid_when_none_provided() -> None:
    """set_correlation_id() with no arg should generate a UUID."""
    cid = set_correlation_id()
    # Should be a valid UUID string
    parsed = uuid.UUID(cid)
    assert str(parsed) == cid


def test_set_correlation_id_uses_provided_value() -> None:
    """An explicit correlation ID should be used as-is."""
    custom_id = "my-custom-correlation-id-123"
    result = set_correlation_id(custom_id)
    assert result == custom_id
    assert correlation_id_var.get() == custom_id


def test_json_formatter_outputs_valid_json() -> None:
    """JsonFormatter should produce parseable JSON."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "test message"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert "timestamp" in parsed


def test_json_formatter_includes_correlation_id() -> None:
    """The correlation ID from context must appear in the JSON output."""
    set_correlation_id("abc-123")
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="t.py",
        lineno=1, msg="hello", args=(), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["correlation_id"] == "abc-123"


def test_json_formatter_includes_extra_fields() -> None:
    """Extra fields passed via logger.info(..., extra={...}) must be preserved."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="t.py",
        lineno=1, msg="hello", args=(), exc_info=None,
    )
    record.user_id = "user_42"
    record.duration_ms = 123
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["user_id"] == "user_42"
    assert parsed["duration_ms"] == 123


def test_json_formatter_handles_exceptions() -> None:
    """Exception info should be captured in the JSON output."""
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="t.py",
            lineno=1, msg="error happened", args=(),
            exc_info=sys.exc_info(),
        )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_setup_logging_human_readable_format() -> None:
    """Human-readable mode produces non-JSON output."""
    stream = StringIO()
    setup_logging(level="DEBUG", json_format=False)
    # Replace handler stream so we can capture
    logging.getLogger().handlers[0].stream = stream

    logger = logging.getLogger("test_human")
    logger.info("hello world")

    output = stream.getvalue()
    assert "hello world" in output
    # Should NOT be JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.strip())


def test_setup_logging_json_format() -> None:
    """JSON mode produces valid JSON lines."""
    stream = StringIO()
    setup_logging(level="INFO", json_format=True)
    logging.getLogger().handlers[0].stream = stream

    logger = logging.getLogger("test_json")
    logger.info("structured message")

    output = stream.getvalue().strip()
    parsed = json.loads(output)
    assert parsed["message"] == "structured message"