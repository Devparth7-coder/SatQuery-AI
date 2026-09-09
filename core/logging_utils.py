"""SatQuery AI v2 — structured logging.

Single-line JSON-ish log events. A ``request_id`` context variable lets the
API middleware tag every log line belonging to one HTTP request.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_configured = False


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


def get_logger(name: str = "satquery") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _configured = True
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    payload = {"ts": round(time.time(), 3), "event": event,
               "request_id": get_request_id()}
    payload.update(fields)
    try:
        line = json.dumps(payload, default=str)
    except Exception:
        line = f'{{"event": "{event}"}}'
    logger.log(level, line)
