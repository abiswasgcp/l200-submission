"""Unit tests for model routing (no LLM calls)."""

import asyncio
from types import SimpleNamespace

from google.adk.models.llm_request import LlmRequest

from app.models import (
    MODELS,
    ROLE_TIERS,
    escalate_planner_model,
    fallback_from_deep_model,
    model_for,
)


def _ctx(state: dict) -> SimpleNamespace:
    return SimpleNamespace(state=state, agent_name="meal_planner")


def test_role_tiers_use_cheap_models_for_simple_jobs():
    assert ROLE_TIERS["intake_classifier"] == "fast"
    assert ROLE_TIERS["presenter"] == "fast"
    assert ROLE_TIERS["meal_planner"] == "standard"
    assert len(set(MODELS.values())) == 3  # three distinct models


def test_model_for_returns_tier_model():
    assert model_for("intake_classifier").model == MODELS["fast"]
    assert model_for("profile_agent").model == MODELS["standard"]


def test_planner_first_attempt_stays_on_standard():
    req = LlmRequest(model=MODELS["standard"])
    assert escalate_planner_model(_ctx({}), req) is None
    assert req.model == MODELS["standard"]


def test_planner_escalates_to_deep_after_revision():
    req = LlmRequest(model=MODELS["standard"])
    assert escalate_planner_model(_ctx({"plan_revisions": 1}), req) is None
    assert req.model == MODELS["deep"]


def test_fallback_ignores_non_deep_errors():
    req = LlmRequest(model=MODELS["standard"])
    result = asyncio.run(
        fallback_from_deep_model(
            callback_context=_ctx({}), llm_request=req, error=RuntimeError("x")
        )
    )
    assert result is None
    assert req.model == MODELS["standard"]
