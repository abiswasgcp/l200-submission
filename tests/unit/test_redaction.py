"""Unit tests for the PII redaction pipeline (no LLM calls)."""

import json
import logging

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.observability import log_event
from app.redaction import (
    MASK,
    bq_content_formatter,
    install_log_redaction,
    redact_text,
    redact_value,
    scrub,
)


def test_detects_common_pii():
    text = (
        "I'm Jo, my name is Jane Doe, email jane.doe@example.com, call "
        "+1 (415) 555-0134, deliver to 742 Evergreen Terrace, card "
        "4111 1111 1111 1111, ssn 123-45-6789"
    )
    clean, counts = scrub(text)
    for secret in ("Jane Doe", "jane.doe@example.com", "555-0134", "Evergreen", "4111"):
        assert secret not in clean
    assert "123-45-6789" not in clean
    assert "my name is [NAME]" in clean
    assert {"EMAIL", "PHONE", "ADDRESS", "CARD", "SSN", "NAME"} <= set(counts)


def test_keeps_domain_data():
    text = (
        "Plan 5 dinners for 2 under $80, nothing over 30 minutes. "
        "Order PP-E5496AC2 total $7.90, recipe veg-fried-rice, peanut-free. "
        "This is Thai food."
    )
    assert redact_text(text) == text


def test_long_non_card_number_is_kept():
    # 16 digits that fail the Luhn check (an ID, not a card)
    assert redact_text("ref 1234567812345678") == "ref 1234567812345678"


def test_redact_value_masks_sensitive_keys_and_nested_text():
    value = {
        "delivery_window": "Saturday morning",
        "address": "anything at all",
        "notes": ["ring 415-555-0134"],
        "recipe": {"name": "Shakshuka"},
    }
    out = redact_value(value)
    assert out["delivery_window"] == "Saturday morning"
    assert out["address"] == MASK
    assert out["notes"] == ["ring [PHONE]"]
    assert out["recipe"] == {"name": "Shakshuka"}


def test_bq_formatter_redacts_llm_request_without_mutating_original():
    req = LlmRequest(
        model="m",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="email me at a@b.co")],
            )
        ],
        config=types.GenerateContentConfig(system_instruction="call 415-555-0134"),
    )
    out = bq_content_formatter(req, "LLM_REQUEST")
    assert out.contents[0].parts[0].text == "email me at [EMAIL]"
    assert out.config.system_instruction == "call [PHONE]"
    assert req.contents[0].parts[0].text == "email me at a@b.co"


def test_bq_formatter_redacts_tool_payload_dicts():
    out = bq_content_formatter(
        {"tool": "update_pantry", "args": {"items": ["a@b.co"]}}, "TOOL_STARTING"
    )
    assert out == {"tool": "update_pantry", "args": {"items": ["[EMAIL]"]}}


def test_log_event_and_plain_logs_are_redacted(caplog):
    install_log_redaction()
    with caplog.at_level(logging.INFO):
        log_event("tool_intent", args={"note": "a@b.co", "phone": "x"})
        logging.getLogger("third.party").info("user said %s", "a@b.co")
    first = json.loads(caplog.records[0].getMessage())
    assert first["args"] == {"note": "[EMAIL]", "phone": MASK}
    assert caplog.records[1].getMessage() == "user said [EMAIL]"


def test_span_processor_redacts_content_attributes_before_export():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from app.redaction import RedactingSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # Exporter registered FIRST: redaction must still apply (runs in _on_ending).
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(RedactingSpanProcessor())
    with provider.get_tracer("t").start_as_current_span("call_llm") as span:
        span.set_attribute("gcp.vertex.agent.llm_request", '{"text": "a@b.co"}')
        span.set_attribute("other.attr", "a@b.co")  # non-content attrs untouched
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["gcp.vertex.agent.llm_request"] == '{"text": "[EMAIL]"}'
    assert attrs["other.attr"] == "a@b.co"
