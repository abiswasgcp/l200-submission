"""Model routing: pick the cheapest model that can do each job, escalate on failure.

Strategy
--------
1. **Static, per-role tiers** (``ROLE_TIERS``): simple jobs (intent
   classification, formatting, summarisation) run on a lite model; tool-using
   agents run on the standard model.
2. **Dynamic escalation**: the meal planner starts on the standard model. When
   the deterministic validator rejects its plan and routes ``revise``, the
   retry is escalated to the deep (Pro) model, which is better at satisfying
   many constraints at once. If the deep model errors (e.g. preview capacity),
   the call falls back to the standard model so the user still gets an answer.

Every tier can be overridden with an env var (``MODEL_FAST``, ``MODEL_STANDARD``,
``MODEL_DEEP``) without code changes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from opentelemetry import trace

logger = logging.getLogger("pantrypal.routing")

Tier = Literal["fast", "standard", "deep"]

MODELS: dict[Tier, str] = {
    "fast": os.environ.get("MODEL_FAST", "gemini-3.5-flash-lite"),
    "standard": os.environ.get("MODEL_STANDARD", "gemini-3.8-flash"),
    "deep": os.environ.get("MODEL_DEEP", "gemini-3.1-pro-preview"),
}

# Which tier each node uses by default (the routing table).
ROLE_TIERS: dict[str, Tier] = {
    "intake_classifier": "fast",  # 5-way structured classification
    "presenter": "fast",  # formats JSON that code already computed
    "summarizer": "fast",  # events compaction summaries
    "profile_agent": "standard",  # tool calling + merging profile fields
    "checkout_agent": "standard",  # single HITL tool call
    "meal_planner": "standard",  # escalates to "deep" on revision
}


def gemini(tier: Tier) -> Gemini:
    """Builds a Gemini model for a tier, with retries on transient errors."""
    return Gemini(model=MODELS[tier], retry_options=types.HttpRetryOptions(attempts=3))


def model_for(role: str) -> Gemini:
    return gemini(ROLE_TIERS[role])


def _record(agent: str, tier: Tier, reason: str) -> None:
    span = trace.get_current_span()
    span.set_attribute("pantrypal.model_tier", tier)
    span.set_attribute("pantrypal.model", MODELS[tier])
    logger.info(
        json.dumps(
            {
                "event": "model_route",
                "agent": agent,
                "tier": tier,
                "model": MODELS[tier],
                "reason": reason,
            }
        )
    )


# ---------------------------------------------------------------------------
# Dynamic escalation for the planner
# ---------------------------------------------------------------------------
def escalate_planner_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """before_model_callback: use the deep model once the validator asked for a revision."""
    revisions = int(callback_context.state.get("plan_revisions") or 0)
    if revisions >= 1:
        llm_request.model = MODELS["deep"]
        _record(callback_context.agent_name, "deep", f"validator revision #{revisions}")
    else:
        _record(callback_context.agent_name, "standard", "first attempt")
    return None  # continue with the (possibly re-routed) request


async def fallback_from_deep_model(
    callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
) -> LlmResponse | None:
    """on_model_error_callback: if the deep model fails, retry once on standard."""
    if llm_request.model != MODELS["deep"]:
        return None  # not an escalated call; let ADK handle the error
    logger.warning(
        json.dumps(
            {
                "event": "model_fallback",
                "agent": callback_context.agent_name,
                "from": MODELS["deep"],
                "to": MODELS["standard"],
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        )
    )
    trace.get_current_span().set_attribute("pantrypal.model_fallback", True)
    llm_request.model = MODELS["standard"]
    final: LlmResponse | None = None
    async for response in gemini("standard").generate_content_async(llm_request):
        if not response.partial:
            final = response
    return final
