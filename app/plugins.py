"""Runner-wide observability plugin.

A ``BasePlugin`` attached to the ``App`` wraps every agent, model call and tool
in the workflow without touching their code. ``ToolAuditPlugin``:

* logs a ``tool_intent`` line **before** each tool executes (what is about to
  run, why, and with which redacted arguments), then a ``tool_result`` line
  with the same ``call_id`` after it finishes, so every action can be traced
  from intent to outcome;
* tags the active OpenTelemetry span with tool status/latency and with
  **redacted** prompt, reply and tool-argument previews, so traces are
  debuggable without exposing PII.

All values pass through :mod:`app.redaction` first.
"""

from __future__ import annotations

import json
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from app.observability import log_event, set_span_attr
from app.redaction import content_text, install_trace_redaction, redact_value

_PREVIEW_CHARS = 1000


def _preview(value: Any) -> str:
    return json.dumps(redact_value(value), default=str)[:_PREVIEW_CHARS]


class ToolAuditPlugin(BasePlugin):
    """Intent + outcome audit logs and redacted span attributes."""

    def __init__(self, name: str = "tool_audit_plugin") -> None:
        super().__init__(name=name)
        self._started: dict[str, float] = {}

    @staticmethod
    def _key(tool_context: ToolContext) -> str:
        return tool_context.function_call_id or str(id(tool_context))

    # -- tools ---------------------------------------------------------------
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        call_id = self._key(tool_context)
        self._started[call_id] = time.perf_counter()
        purpose = (tool.description or "").strip().splitlines()[0:1]
        # Pre-execution intent: logged before the tool (or any guard) runs.
        log_event(
            "tool_intent",
            call_id=call_id,
            tool=tool.name,
            agent=tool_context.agent_name,
            session_id=tool_context.session.id,
            purpose=purpose[0] if purpose else "",
            args=tool_args,
        )
        set_span_attr("tool.args", _preview(tool_args))
        return None

    def _result(self, record: dict[str, Any]) -> None:
        log_event("tool_result", **record)
        set_span_attr("tool.status", record.get("status", "unknown"))
        if record.get("latency_ms") is not None:
            set_span_attr("tool.latency_ms", record["latency_ms"])

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        call_id = self._key(tool_context)
        started = self._started.pop(call_id, None)
        is_dict = isinstance(result, dict)
        status = result.get("status", "success") if is_dict else "success"
        self._result(
            {
                "call_id": call_id,
                "tool": tool.name,
                "agent": tool_context.agent_name,
                "session_id": tool_context.session.id,
                "status": status,
                "error_message": result.get("error_message") if is_dict else None,
                "result_keys": sorted(result) if is_dict else None,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)
                if started
                else None,
            }
        )
        set_span_attr("tool.result", _preview(result))
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict | None:
        call_id = self._key(tool_context)
        self._started.pop(call_id, None)
        self._result(
            {
                "call_id": call_id,
                "tool": tool.name,
                "agent": tool_context.agent_name,
                "session_id": tool_context.session.id,
                "status": "exception",
                "error_message": f"{type(error).__name__}: {error}",
            }
        )
        return None  # let ReflectAndRetryToolPlugin / ADK handle the error

    # -- models: redacted content on the call_llm span ------------------------
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        last = llm_request.contents[-1] if llm_request.contents else None
        set_span_attr("llm.input", content_text(last, _PREVIEW_CHARS))
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if not llm_response.partial:
            set_span_attr(
                "llm.output", content_text(llm_response.content, _PREVIEW_CHARS)
            )
        return None

    # -- invocation ------------------------------------------------------------
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        # The server may install its tracer provider after the app is imported.
        install_trace_redaction()
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        log_event(
            "invocation_complete",
            invocation_id=invocation_context.invocation_id,
            session_id=invocation_context.session.id,
            event_count=len(invocation_context.session.events),
        )
