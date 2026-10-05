"""PII redaction pipeline: one scrubber for logs, spans and BigQuery rows.

Every telemetry sink in PantryPal goes through this module before data leaves
the process:

* **Structured logs** — :func:`app.observability.log_event` redacts every field,
  and :func:`install_log_redaction` installs a ``LogRecord`` factory so even
  plain ``logger.info("...")`` lines (ours or a library's) are scrubbed.
* **Trace spans** — :func:`app.observability.set_span_attr` redacts string
  attributes, ``ToolAuditPlugin`` records *redacted* prompts, replies and
  tool arguments on spans, and :class:`RedactingSpanProcessor` scrubs ADK's own
  content attributes before any exporter sees them.
* **BigQuery Agent Analytics** — :func:`bq_content_formatter` is plugged into
  ``BigQueryLoggerConfig.content_formatter`` and rewrites each event payload
  (user messages, LLM requests/responses, tool args/results) before it is
  written. The plugin treats the formatter as a privacy boundary: if it raises,
  the row is written as ``[FORMATTER_FAILED]``, never as raw content.

Two complementary techniques are applied:

1. **Pattern detection** in free text: e-mail addresses, phone numbers, payment
   card numbers (Luhn-checked to avoid masking prices or IDs), US SSNs, IPv4
   addresses, street addresses and self-introduced names ("my name is ...").
2. **Sensitive keys** in structured data (``address``, ``email``, ``phone`` ...):
   the whole value is masked whatever it contains.

Domain data the agent needs to be debuggable (prices, servings, recipe and
order IDs, allergens, prep times) is intentionally left untouched.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

MASK = "[REDACTED]"

# Keys whose values are always masked, regardless of content.
SENSITIVE_KEYS = frozenset(
    {
        "address",
        "delivery_address",
        "street",
        "email",
        "phone",
        "phone_number",
        # Not bare "name": recipes and ingredients use it in tool results.
        "full_name",
        "first_name",
        "last_name",
        "card",
        "card_number",
        "payment",
        "payment_method",
        "password",
        "api_key",
        "token",
        "authorization",
    }
)

_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct"
    r"|Way|Place|Pl|Terrace|Ter|Parkway|Pkwy|Highway|Hwy|Circle|Cir)"
)

# Order matters: more specific patterns run first so e.g. a card number is not
# partially consumed by the phone pattern.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "PHONE",
        re.compile(
            r"(?<![\w$])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"
        ),
    ),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (
        "ADDRESS",
        re.compile(
            rf"\b\d{{1,6}}\s+(?:[A-Z][\w'-]*\.?\s+){{1,4}}{_STREET_SUFFIX}\b\.?"
            r"(?:,?\s*(?:Apt|Unit|Suite|#)\s*\w+)?"
        ),
    ),
    (
        "NAME",
        re.compile(
            r"(?i:\b(?:my name is|call me|i am called)\s+)"
            r"([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?)"
        ),
    ),
]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def scrub(text: str) -> tuple[str, Counter[str]]:
    """Redacts PII in ``text``. Returns the clean text and counts per PII type."""
    counts: Counter[str] = Counter()
    if not text:
        return text, counts

    for label, pattern in _PATTERNS:

        def _sub(m: re.Match[str], label: str = label) -> str:
            if label == "CARD" and not _luhn_ok(re.sub(r"\D", "", m.group(0))):
                return m.group(0)  # long number that isn't a card (an ID, etc.)
            counts[label] += 1
            if label == "NAME":  # keep the lead-in ("my name is"), mask the name
                return m.group(0)[: m.start(1) - m.start(0)] + f"[{label}]"
            return f"[{label}]"

        text = pattern.sub(_sub, text)
    return text, counts


def redact_text(text: str) -> str:
    """Redacts PII in a string."""
    return scrub(text)[0]


def redact_value(value: Any, _depth: int = 0) -> Any:
    """Recursively redacts strings in dicts/lists/tuples and masks sensitive keys."""
    if _depth > 20:
        return MASK
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            k: MASK
            if isinstance(k, str) and k.lower() in SENSITIVE_KEYS
            else redact_value(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v, _depth + 1) for v in value)
    return value


# ---------------------------------------------------------------------------
# google-genai / ADK objects
# ---------------------------------------------------------------------------
def redact_part(part: types.Part) -> types.Part:
    update: dict[str, Any] = {}
    if part.text:
        update["text"] = redact_text(part.text)
    if part.function_call and part.function_call.args:
        update["function_call"] = part.function_call.model_copy(
            update={"args": redact_value(part.function_call.args)}
        )
    if part.function_response and part.function_response.response:
        update["function_response"] = part.function_response.model_copy(
            update={"response": redact_value(part.function_response.response)}
        )
    return part.model_copy(update=update) if update else part


def redact_content(content: types.Content) -> types.Content:
    return content.model_copy(
        update={"parts": [redact_part(p) for p in content.parts or []]}
    )


def redact_llm_request(req: LlmRequest) -> LlmRequest:
    """Returns a redacted copy of an LLM request (the original is untouched)."""
    config = req.config
    if config is not None and config.system_instruction is not None:
        si = config.system_instruction
        if isinstance(si, str):
            si = redact_text(si)
        elif isinstance(si, types.Content):
            si = redact_content(si)
        config = config.model_copy(update={"system_instruction": si})
    return LlmRequest(
        model=req.model,
        contents=[redact_content(c) for c in req.contents],
        config=config,
    )


def content_text(content: types.Content | None, limit: int = 1000) -> str:
    """Redacted plain-text view of a Content (for span attributes)."""
    if not content or not content.parts:
        return ""
    chunks = []
    for p in content.parts:
        if p.text and not p.thought:
            chunks.append(p.text)
        elif p.function_call:
            chunks.append(f"<call {p.function_call.name}>")
        elif p.function_response:
            chunks.append(f"<result {p.function_response.name}>")
    return redact_text(" ".join(chunks))[:limit]


# ---------------------------------------------------------------------------
# Sink adapters
# ---------------------------------------------------------------------------
def bq_content_formatter(raw_content: Any, event_type: str) -> Any:
    """``BigQueryLoggerConfig.content_formatter``: redact every event payload."""
    if isinstance(raw_content, LlmRequest):
        return redact_llm_request(raw_content)
    if isinstance(raw_content, types.Content):
        return redact_content(raw_content)
    if isinstance(raw_content, types.Part):
        return redact_part(raw_content)
    return redact_value(raw_content)


# Span attributes that may carry prompt/response/tool content.
_CONTENT_ATTR_PREFIXES = ("gcp.vertex.agent.", "gen_ai.", "pantrypal.")


class RedactingSpanProcessor(SpanProcessor):
    """Scrubs PII from content-bearing span attributes as each span ends.

    Runs in the SDK's ``_on_ending`` hook, which fires for every registered
    processor *before* any ``on_end``/export, so ADK's own attributes (e.g.
    ``gcp.vertex.agent.llm_request`` when content capture is enabled) are
    redacted no matter which exporter is configured or in what order.
    """

    def _on_ending(self, span: Any) -> None:
        attrs = getattr(span, "_attributes", None)
        if not attrs:
            return
        try:
            for key, value in list(attrs.items()):
                if isinstance(value, str) and key.startswith(_CONTENT_ATTR_PREFIXES):
                    clean = redact_text(value)
                    if clean != value:
                        attrs[key] = clean
        except Exception:  # never break tracing
            logging.getLogger(__name__).debug("span redaction failed", exc_info=True)


_trace_providers: set[int] = set()


def install_trace_redaction() -> None:
    """Adds :class:`RedactingSpanProcessor` to the global tracer provider (idempotent).

    Safe to call repeatedly: it is called at import time and again on each run,
    because servers often replace the tracer provider after the app is imported.
    """
    provider = trace.get_tracer_provider()
    add = getattr(provider, "add_span_processor", None)
    if add is None or id(provider) in _trace_providers:
        return
    add(RedactingSpanProcessor())
    _trace_providers.add(id(provider))


_installed = False


def install_log_redaction() -> None:
    """Scrubs PII from every log record created in this process (idempotent).

    Wrapping the ``LogRecord`` factory catches log lines from any logger,
    including third-party libraries, before any handler or exporter sees them.
    """
    global _installed
    if _installed:
        return
    previous: Callable[..., logging.LogRecord] = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        try:
            message = record.getMessage()
        except Exception:
            return record
        clean = redact_text(message)
        if clean != message:
            record.msg, record.args = clean, None
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
