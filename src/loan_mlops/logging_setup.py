"""Structured JSON logging setup for production-grade observability.

JSON logs are machine-parsable by log aggregators (Datadog, CloudWatch, Splunk).
Includes correlation IDs so requests can be traced across services.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Correlation ID for tracing a single request/run through the system
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


_STANDARD_FIELDS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
        "getMessage",
    }
)

# These cannot be passed via extra={} — Python's logging module reserves them.
# We intercept and rename them to avoid the KeyError that would otherwise crash logging.
_RESERVED_EXTRA_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "message",
        "asctime",
    }
)


def _safe_extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_FIELDS}


def set_correlation_id(cid: str | None = None) -> str:
    """Set a correlation ID for the current context. Generates one if not provided."""
    cid = cid or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for ingestion by log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Include any extra fields passed via logger.info(..., extra={...})
        log_obj.update(_safe_extras(record))
        return json.dumps(log_obj)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON. If False, human-readable (use in local dev).
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
