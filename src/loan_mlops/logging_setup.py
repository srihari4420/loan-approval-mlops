"""Structured logging with correlation IDs.

Human-readable in dev, JSON in production. Both formats surface extra fields
passed via logger.info(..., extra={...}) — that's the bit production needs."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# Standard LogRecord attributes — anything else in record.__dict__ is user-supplied extras
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


def set_correlation_id(cid: str | None = None) -> str:
    cid = cid or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def _extract_extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_FIELDS}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        log_obj.update(_extract_extras(record))
        return json.dumps(log_obj)


class HumanFormatter(logging.Formatter):
    """Standard format plus inline 'key=value' pairs for any extras."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = _extract_extras(record)
        if not extras:
            return base
        kv = " | ".join(f"{k}={v}" for k, v in extras.items())
        return f"{base} | {kv}"


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            HumanFormatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
