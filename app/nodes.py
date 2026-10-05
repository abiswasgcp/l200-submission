"""Deterministic workflow nodes (no LLM): routing, validation, list building."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.genai import types
from opentelemetry import trace

from app import catalog
from app.tools import (
    LAST_LIST_KEY,
    LAST_PLAN_KEY,
    PANTRY_KEY,
    PROFILE_KEY,
    get_profile,
)

logger = logging.getLogger(__name__)

MAX_PLAN_REVISIONS = 2
VALID_ROUTES = ("profile_or_pantry", "meal_plan", "order", "help", "unrelated")

HELP_MESSAGE = (
    "Hi, I'm **PantryPal** 🥕, your meal-plan and grocery concierge. I can:\n"
    "1. **Remember your household**: diet, allergies, dislikes, servings and budget "
    '(e.g. *"We\'re vegetarian, allergic to peanuts, cooking for 2 on $80/week"*).\n'
    '2. **Track your pantry** (e.g. *"I have rice, eggs and spinach"*).\n'
    "3. **Plan meals** that respect your allergies and budget "
    '(e.g. *"Plan 5 dinners, nothing over 30 minutes"*).\n'
    "4. **Build a shopping list** that skips what you already have.\n"
    "5. **Place the grocery order**, but only after you approve it."
)

DECLINE_MESSAGE = (
    "Thanks for asking! I'm PantryPal, so I can only help with meal planning: "
    "your dietary profile, pantry, meal plans, shopping lists and grocery orders. "
    "Want me to plan some meals?"
)


def _span_attr(key: str, value: Any) -> None:
    """Adds a custom attribute to the current OpenTelemetry span (Cloud Trace)."""
    trace.get_current_span().set_attribute(f"pantrypal.{key}", value)


def _say(text: str):
    """Yields a user-visible message plus the node output."""
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)])
    )
    yield Event(output=text)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_intent(ctx: Context, node_input: Any = None):
    """Routes on the classifier's intent and resets per-request planning state.

    ``node_input`` is typed ``Any`` because a blocked classifier can yield None;
    anything unexpected is treated as 'unrelated'.
    """
    intent = node_input if isinstance(node_input, dict) else {}
    route = intent.get("route")
    if route not in VALID_ROUTES:
        route = "unrelated"
    _span_attr("route", route)
    logger.info("route_intent route=%s", route)
    state: dict[str, Any] = {}

    # Safety net: allergies/diet stated in ANY message (e.g. "I'm allergic to
    # peanuts, plan 3 dinners") are merged into the stored profile in code, so
    # the deterministic validator enforces them even if the turn never reaches
    # profile_agent. Allergies are only ever added here, never removed.
    mentioned = [
        a
        for a in (
            catalog.normalize_allergen(x)
            for x in intent.get("allergies_mentioned") or []
        )
        if a
    ]
    diet = intent.get("diet_mentioned")
    if route != "unrelated" and (mentioned or diet in catalog.DIET_RANK):
        profile = get_profile(ctx.state)
        profile.allergies = sorted(set(profile.allergies) | set(mentioned))
        if diet in catalog.DIET_RANK:
            profile.diet = diet
        state[PROFILE_KEY] = profile.model_dump()

    if route == "meal_plan":
        state |= {
            "plan_request": {
                "days": intent.get("days"),
                "max_prep_minutes": intent.get("max_prep_minutes"),
                "request_summary": intent.get("request_summary", ""),
            },
            "plan_revisions": 0,
            "plan_feedback": "",
        }
    # Forward the user's original message so downstream agents answer it.
    yield Event(output=ctx.user_content, route=route, state=state)


def help_node(node_input: Any = None):
    """Static capabilities message for greetings / 'what can you do'."""
    yield from _say(HELP_MESSAGE)


def decline(node_input: Any = None):
    """Politely declines out-of-scope requests."""
    yield from _say(DECLINE_MESSAGE)


# ---------------------------------------------------------------------------
# Plan validation loop
# ---------------------------------------------------------------------------
def validate_plan(ctx: Context, node_input: Any = None):
    """Checks the planner's output against hard rules (allergens, diet, budget).

    * Violations and revisions left: route 'revise' back to the planner with
      precise feedback (the loop).
    * Otherwise route 'ok'. Unsafe meals (allergen, diet or unknown recipe) are
      REMOVED, so they can never reach the user even if the model keeps
      proposing them.
    """
    plan = node_input if isinstance(node_input, dict) else {}
    meals = [m for m in plan.get("meals") or [] if isinstance(m, dict)]
    request = ctx.state.get("plan_request") or {}
    revisions = int(ctx.state.get("plan_revisions") or 0)
    profile = get_profile(ctx.state)

    by_day = {
        int(m.get("day", i + 1)): str(m.get("recipe_id", ""))
        for i, m in enumerate(meals)
    }
    violations, unsafe = catalog.check_plan(
        by_day,
        profile,
        requested_days=request.get("days"),
        max_prep_minutes=request.get("max_prep_minutes"),
    )
    if not meals:
        violations.append("The planner returned no meals.")

    # Budget check uses the real shopping-list total (after pantry savings).
    safe_ids = [rid for day, rid in sorted(by_day.items()) if day not in unsafe]
    if safe_ids:
        shopping = catalog.build_shopping_list(
            safe_ids, profile.servings, ctx.state.get(PANTRY_KEY) or []
        )
        budget = catalog.plan_budget(profile.weekly_budget, len(safe_ids))
        if shopping.total_cost > budget:
            violations.append(
                f"Estimated groceries ${shopping.total_cost:.2f} exceed the "
                f"pro-rated budget ${budget:.2f}; choose cheaper recipes "
                "(use max_cost_per_serving)."
            )

    _span_attr("plan_revisions", revisions)
    _span_attr("plan_violations", len(violations))

    if violations and revisions < MAX_PLAN_REVISIONS:
        feedback = (
            "Fix these problems and return the full corrected plan:\n- "
            + "\n- ".join(violations)
        )
        logger.info("validate_plan revise #%d: %s", revisions + 1, violations)
        yield Event(
            output=feedback,
            route="revise",
            state={"plan_revisions": revisions + 1, "plan_feedback": feedback},
        )
        return

    safe_meals = [m for m in meals if int(m.get("day", 0)) not in unsafe]
    warnings = list(violations)
    if unsafe:
        warnings.append(
            f"Removed {len(unsafe)} meal(s) that conflicted with allergies or diet."
        )
    _span_attr("plan_valid", not violations)
    yield Event(
        output={
            "meals": safe_meals,
            "rationale": plan.get("rationale", ""),
            "warnings": warnings,
        },
        route="ok",
        state={"plan_feedback": ""},
    )


# ---------------------------------------------------------------------------
# Shopping list
# ---------------------------------------------------------------------------
def build_shopping_list(ctx: Context, node_input: Any = None):
    """Turns the validated plan into a priced shopping list minus pantry stock."""
    plan = node_input if isinstance(node_input, dict) else {}
    meals = plan.get("meals") or []
    profile = get_profile(ctx.state)
    recipes = catalog.load_catalog()
    ids = [m["recipe_id"] for m in meals if m.get("recipe_id") in recipes]
    shopping = catalog.build_shopping_list(
        ids, profile.servings, ctx.state.get(PANTRY_KEY) or []
    )
    plan_rows = []
    for m in sorted(meals, key=lambda x: x.get("day", 0)):
        r = recipes.get(m.get("recipe_id"))
        if r:
            plan_rows.append(
                {
                    "day": m.get("day"),
                    "recipe_id": r["id"],
                    "name": r["name"],
                    "prep_minutes": r["prep_minutes"],
                    "cost_per_serving": r["cost_per_serving"],
                    "kcal_per_serving": r["kcal"],
                    "allergens": r["allergens"],
                }
            )
    budget = catalog.plan_budget(profile.weekly_budget, max(1, len(plan_rows)))
    result = {
        "plan": plan_rows,
        "rationale": plan.get("rationale", ""),
        "warnings": plan.get("warnings", []),
        "servings": profile.servings,
        "shopping_list": shopping.model_dump(),
        "budget": {"limit": budget, "within_budget": shopping.total_cost <= budget},
        "profile_is_default": not ctx.state.get("user:dietary_profile"),
    }
    _span_attr("shopping_items", len(shopping.items))
    yield Event(
        output=result,
        state={
            LAST_PLAN_KEY: {"meals": plan_rows, "rationale": result["rationale"]},
            LAST_LIST_KEY: shopping.model_dump(),
        },
    )
