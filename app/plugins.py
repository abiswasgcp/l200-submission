"""Runner-wide observability plugin.

A ``BasePlugin`` attached to the ``App`` wraps every agent and tool in the
workflow without touching their code. ``ToolAuditPlugin`` emits one structured
JSON log line per tool call (Cloud Logging parses these into ``jsonPayload``)
and tags the active OpenTelemetry span so tool outcomes are searchable in
Cloud Trace.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from opentelemetry import trace

logger = logging.getLogger("pantrypal.audit")


class ToolAuditPlugin(BasePlugin):
    """Structured audit log + span attributes for every tool call."""

    def __init__(self, name: str = "tool_audit_plugin") -> None:
        super().__init__(name=name)
        self._started: dict[str, float] = {}

    @staticmethod
    def _key(tool_context: ToolContext) -> str:
        return tool_context.function_call_id or str(id(tool_context))

    def _emit(self, record: dict[str, Any]) -> None:
        logger.info(json.dumps({"event": "tool_call", **record}, default=str))
        span = trace.get_current_span()
        span.set_attribute("pantrypal.tool.status", record.get("status", "unknown"))
        if "latency_ms" in record:
            span.set_attribute("pantrypal.tool.latency_ms", record["latency_ms"])

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        self._started[self._key(tool_context)] = time.perf_counter()
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        started = self._started.pop(self._key(tool_context), None)
        status = (
            result.get("status", "success") if isinstance(result, dict) else "success"
        )
        self._emit(
            {
                "tool": tool.name,
                "agent": tool_context.agent_name,
                "session_id": tool_context.session.id,
                "status": status,
                "error_message": result.get("error_message")
                if isinstance(result, dict)
                else None,
                "arg_keys": sorted(tool_args),  # keys only: never log user content
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)
                if started
                else None,
            }
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict | None:
        self._started.pop(self._key(tool_context), None)
        self._emit(
            {
                "tool": tool.name,
                "agent": tool_context.agent_name,
                "session_id": tool_context.session.id,
                "status": "exception",
                "error_message": f"{type(error).__name__}: {error}",
            }
        )
        return None  # let ReflectAndRetryToolPlugin / ADK handle the error

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "invocation_complete",
                    "invocation_id": invocation_context.invocation_id,
                    "session_id": invocation_context.session.id,
                    "event_count": len(invocation_context.session.events),
                }
            )
        )
