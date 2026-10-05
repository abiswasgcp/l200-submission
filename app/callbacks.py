"""Agent callbacks: memory persistence, guardrail fallbacks and the budget guard."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app import catalog
from app.schemas import Intent
from app.tools import LAST_LIST_KEY, LAST_PLAN_KEY, get_profile

logger = logging.getLogger(__name__)

GUARDRAIL_FALLBACK_MESSAGE = (
    "Sorry, I can't help with that. I'm PantryPal: I can remember your dietary "
    "profile and pantry, plan meals, build a shopping list, and place a grocery "
    "order once you approve it."
)


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------
async def add_session_to_memory_callback(callback_context: CallbackContext) -> None:
    """Sends the session to the memory service (Memory Bank in the cloud).

    Memory Bank extracts durable facts ("loves Thai food", "kids won't eat
    spicy") that ``preload_memory`` recalls at the start of later sessions.
    """
    try:
        await callback_context.add_session_to_memory()
    except ValueError as e:  # raised when no memory service is configured
        logger.warning("Memory extraction skipped: %s", e)
    return None


# ---------------------------------------------------------------------------
# Guardrail fallbacks (blocked or empty model responses)
# ---------------------------------------------------------------------------
def _is_blocked_or_empty(llm_response: LlmResponse) -> bool:
    if llm_response.partial:
        return False
    if llm_response.error_code:
        return True
    parts = llm_response.content.parts if llm_response.content else None
    return not parts or not any(p.text or p.function_call for p in parts)


def classifier_guardrail_fallback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """A blocked classification routes to 'unrelated' instead of crashing the graph."""
    if not _is_blocked_or_empty(llm_response):
        return None
    logger.warning(
        "Classifier response blocked/empty (error_code=%s)", llm_response.error_code
    )
    fallback = Intent(route="unrelated", request_summary="Blocked by safety filters.")
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=fallback.model_dump_json())]
        )
    )


def reply_guardrail_fallback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """A blocked user-facing reply becomes a polite fallback instead of silence."""
    if not _is_blocked_or_empty(llm_response):
        return None
    logger.warning(
        "Reply blocked/empty in %s (error_code=%s)",
        callback_context.agent_name,
        llm_response.error_code,
    )
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=GUARDRAIL_FALLBACK_MESSAGE)]
        )
    )


# ---------------------------------------------------------------------------
# Budget guard for checkout
# ---------------------------------------------------------------------------
def order_budget_guard(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Blocks orders whose total exceeds the pro-rated budget by more than 20%.

    Returning a dict skips the tool and hands that dict back to the model as the
    tool result.
    """
    if tool.name != "place_grocery_order":
        return None
    shopping_list = tool_context.state.get(LAST_LIST_KEY) or {}
    plan = tool_context.state.get(LAST_PLAN_KEY) or {}
    total = float(shopping_list.get("total_cost") or 0)
    days = max(1, len(plan.get("meals") or []))
    profile = get_profile(tool_context.state)
    limit = round(
        catalog.plan_budget(profile.weekly_budget, days)
        * catalog.ORDER_BUDGET_TOLERANCE,
        2,
    )
    if total > limit:
        logger.warning("Order blocked by budget guard: total=%s limit=%s", total, limit)
        return {
            "status": "error",
            "error_message": (
                f"Order total ${total:.2f} exceeds the budget limit ${limit:.2f}. "
                "Ask the user to re-plan with cheaper meals or raise their budget."
            ),
        }
    return None
