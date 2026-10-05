"""Telemetry helpers: every log line and span attribute passes the PII scrubber.

Use :func:`log_event` for structured JSON logs and :func:`set_span_attr` for
custom span attributes instead of calling ``logger``/``span`` directly, so the
redaction pipeline in :mod:`app.redaction` can't be bypassed by accident.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import trace

from app.redaction import redact_text, redact_value

logger = logging.getLogger("pantrypal")


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emits one redacted JSON log line (Cloud Logging parses it as jsonPayload)."""
    logger.log(level, json.dumps({"event": event, **redact_value(fields)}, default=str))


def set_span_attr(key: str, value: Any) -> None:
    """Sets ``pantrypal.<key>`` on the current span, redacting string values."""
    if isinstance(value, str):
        value = redact_text(value)
    elif isinstance(value, (list, tuple)):
        value = [redact_text(v) if isinstance(v, str) else v for v in value]
    trace.get_current_span().set_attribute(f"pantrypal.{key}", value)
