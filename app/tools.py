"""Agent tools.

Design rules followed by every tool (see README "Tool & Interface Design"):

* verb_noun names and type-hinted parameters, so the generated function
  declarations are precise;
* docstrings say *when* to call the tool, not just what it does;
* every tool returns ``{"status": "success" | "error", ...}`` with an actionable
  ``error_message`` instead of raising, so the model can recover;
* outputs are compact (summaries, not the whole catalog);
* hard rules live in ``app.catalog`` and run in code, never in the prompt.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import ValidationError

from app import catalog
from app.schemas import DietaryProfile

# Session-state keys. The ``user:`` prefix scopes a key to the user across all
# of their sessions (persisted by the session service), which is what makes the
# profile and pantry survive a new conversation.
PROFILE_KEY = "user:dietary_profile"
PANTRY_KEY = "user:pantry"
LAST_PLAN_KEY = "user:last_plan"
LAST_LIST_KEY = "user:last_shopping_list"
ORDERS_KEY = "user:orders"


def get_profile(state: Any) -> DietaryProfile:
    """Reads the stored profile from state, falling back to defaults."""
    raw = state.get(PROFILE_KEY) or {}
    try:
        return DietaryProfile(**raw)
    except (ValidationError, TypeError):
        return DietaryProfile()


# ---------------------------------------------------------------------------
# Catalog tools
# ---------------------------------------------------------------------------
def search_recipes(
    tool_context: ToolContext,
    diet: str | None = None,
    exclude_allergens: list[str] | None = None,
    exclude_ingredients: list[str] | None = None,
    max_prep_minutes: int | None = None,
    max_cost_per_serving: float | None = None,
    cuisine: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Searches the recipe catalog. Use this to find candidate recipes for a plan.

    The household's stored diet, allergies and dislikes are ALWAYS applied on top
    of the arguments you pass, so results are safe for the household.

    Args:
        diet: One of 'omnivore', 'pescatarian', 'vegetarian', 'vegan'. Defaults
            to the stored profile's diet.
        exclude_allergens: Extra allergens to exclude (e.g. ['gluten']).
        exclude_ingredients: Extra ingredient words to exclude (e.g. ['mushroom']).
        max_prep_minutes: Only return recipes that take at most this long.
        max_cost_per_serving: Only return recipes at or under this USD cost.
        cuisine: Optional cuisine filter, e.g. 'italian', 'mexican', 'thai'.
        limit: Maximum number of results (1-30).

    Returns:
        {"status": "success", "count": int, "recipes": [recipe summaries]} sorted
        by cost then prep time, or {"status": "error", "error_message": str}.
    """
    profile = get_profile(tool_context.state)
    effective_diet = (diet or profile.diet).strip().lower()
    if effective_diet not in catalog.DIET_RANK:
        return {
            "status": "error",
            "error_message": (
                f"Unknown diet '{diet}'. Use one of {sorted(catalog.DIET_RANK)}."
            ),
        }
    # A request can only make the diet stricter than the stored profile.
    if catalog.DIET_RANK[effective_diet] > catalog.DIET_RANK[profile.diet]:
        effective_diet = profile.diet
    results = catalog.search_catalog(
        diet=effective_diet,
        exclude_allergens=[*profile.allergies, *(exclude_allergens or [])],
        exclude_ingredients=[*profile.dislikes, *(exclude_ingredients or [])],
        max_prep_minutes=max_prep_minutes,
        max_cost_per_serving=max_cost_per_serving,
        cuisine=cuisine,
        limit=limit,
    )
    if not results:
        return {
            "status": "error",
            "error_message": (
                "No recipes match these filters. Relax optional filters "
                "(cuisine, prep time, cost) but never the allergies or diet."
            ),
        }
    return {"status": "success", "count": len(results), "recipes": results}


def get_recipe_details(recipe_id: str) -> dict[str, Any]:
    """Gets ingredients, nutrition and cost for ONE recipe by its id.

    Use this when the user asks what is in a dish, or to double-check a recipe
    before recommending it.

    Args:
        recipe_id: Catalog id such as 'r05' (from search_recipes results).

    Returns:
        {"status": "success", "recipe": {...}} or
        {"status": "error", "error_message": str}.
    """
    recipe = catalog.load_catalog().get(recipe_id.strip().lower())
    if recipe is None:
        return {
            "status": "error",
            "error_message": f"No recipe with id '{recipe_id}'. Call search_recipes first.",
        }
    details = catalog.recipe_summary(recipe)
    details["base_servings"] = recipe["servings"]
    details["ingredients"] = [
        {k: i[k] for k in ("name", "qty", "unit")} for i in recipe["ingredients"]
    ]
    return {"status": "success", "recipe": details}


# ---------------------------------------------------------------------------
# Profile & pantry tools (session state, user scope)
# ---------------------------------------------------------------------------
def set_dietary_profile(
    tool_context: ToolContext,
    diet: str | None = None,
    allergies: list[str] | None = None,
    dislikes: list[str] | None = None,
    servings: int | None = None,
    weekly_budget: float | None = None,
) -> dict[str, Any]:
    """Saves or updates the household dietary profile. Only pass fields the user stated.

    Fields you omit keep their stored value. Allergies and dislikes passed here
    REPLACE the stored lists, so include previously stored items the user
    still has.

    Args:
        diet: 'omnivore', 'pescatarian', 'vegetarian' or 'vegan'.
        allergies: Allergens, e.g. ['peanuts', 'shellfish'].
        dislikes: Ingredients to avoid by preference, e.g. ['mushrooms'].
        servings: People to cook for per meal (1-12).
        weekly_budget: Weekly grocery budget in USD.

    Returns:
        {"status": "success", "profile": {...}, "warnings": [...]} or
        {"status": "error", "error_message": str}.
    """
    current = get_profile(tool_context.state).model_dump()
    warnings: list[str] = []
    if diet is not None:
        current["diet"] = diet.strip().lower()
    if allergies is not None:
        normalized = []
        for a in allergies:
            key = catalog.normalize_allergen(a)
            if key:
                normalized.append(key)
            else:
                warnings.append(
                    f"'{a}' is not a tracked allergen; ask the user to add it "
                    "as a dislike instead so it is still excluded."
                )
        current["allergies"] = sorted(set(normalized))
    if dislikes is not None:
        current["dislikes"] = sorted({d.strip().lower() for d in dislikes if d.strip()})
    if servings is not None:
        current["servings"] = servings
    if weekly_budget is not None:
        current["weekly_budget"] = weekly_budget
    try:
        profile = DietaryProfile(**current)
    except ValidationError as e:
        return {"status": "error", "error_message": f"Invalid profile: {e.errors()}"}
    tool_context.state[PROFILE_KEY] = profile.model_dump()
    return {"status": "success", "profile": profile.model_dump(), "warnings": warnings}


def get_dietary_profile(tool_context: ToolContext) -> dict[str, Any]:
    """Returns the stored household dietary profile (defaults if never set).

    Returns:
        {"status": "success", "profile": {...}, "is_default": bool}.
    """
    return {
        "status": "success",
        "profile": get_profile(tool_context.state).model_dump(),
        "is_default": not tool_context.state.get(PROFILE_KEY),
    }


def get_pantry(tool_context: ToolContext) -> dict[str, Any]:
    """Lists ingredients the household already has at home.

    Returns:
        {"status": "success", "items": [str], "count": int}.
    """
    items = list(tool_context.state.get(PANTRY_KEY) or [])
    return {"status": "success", "items": items, "count": len(items)}


def update_pantry(
    tool_context: ToolContext,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> dict[str, Any]:
    """Adds and/or removes pantry items. Use when the user says what they have or used up.

    Args:
        add: Ingredient names now in stock, e.g. ['rice', 'eggs'].
        remove: Ingredient names that ran out.

    Returns:
        {"status": "success", "items": [str], "added": [str], "removed": [str]}
        or {"status": "error", "error_message": str}.
    """
    if not add and not remove:
        return {
            "status": "error",
            "error_message": "Pass at least one item in 'add' or 'remove'.",
        }
    pantry = set(tool_context.state.get(PANTRY_KEY) or [])
    added = sorted({catalog.normalize_ingredient(a) for a in add or [] if a.strip()})
    removed = sorted(
        {catalog.normalize_ingredient(r) for r in remove or [] if r.strip()}
    )
    pantry |= set(added)
    pantry -= set(removed)
    tool_context.state[PANTRY_KEY] = sorted(pantry)
    return {
        "status": "success",
        "items": sorted(pantry),
        "added": added,
        "removed": removed,
    }


# ---------------------------------------------------------------------------
# Checkout (human-in-the-loop)
# ---------------------------------------------------------------------------
async def place_grocery_order(
    tool_context: ToolContext, delivery_window: str = "next available"
) -> dict[str, Any]:
    """Places the grocery order for the most recent shopping list (mock checkout).

    The user must explicitly approve this call; the runtime asks for
    confirmation before it executes. The items and total come from the saved
    shopping list, never from the model.

    Args:
        delivery_window: Preferred delivery window, e.g. 'Saturday morning'.

    Returns:
        {"status": "success", "order_id": str, "total_cost": float, ...} or
        {"status": "error", "error_message": str}.
    """
    shopping_list = tool_context.state.get(LAST_LIST_KEY)
    if not shopping_list or not shopping_list.get("items"):
        return {
            "status": "error",
            "error_message": "There is no shopping list yet. Ask the user to plan meals first.",
        }
    order_id = f"PP-{uuid.uuid4().hex[:8].upper()}"
    order = {
        "order_id": order_id,
        "placed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "delivery_window": delivery_window,
        "items": shopping_list["items"],
        "total_cost": shopping_list["total_cost"],
    }
    artifact_name = f"order-{order_id}.json"
    try:
        await tool_context.save_artifact(
            artifact_name,
            types.Part.from_bytes(
                data=json.dumps(order, indent=2).encode(), mime_type="application/json"
            ),
        )
    except ValueError:  # no artifact service configured (e.g. bare unit runs)
        artifact_name = None
    history = list(tool_context.state.get(ORDERS_KEY) or [])
    history.append(
        {
            k: order[k]
            for k in ("order_id", "placed_at", "total_cost", "delivery_window")
        }
    )
    tool_context.state[ORDERS_KEY] = history[-10:]
    return {
        "status": "success",
        "order_id": order_id,
        "item_count": len(order["items"]),
        "total_cost": order["total_cost"],
        "delivery_window": delivery_window,
        "receipt_artifact": artifact_name,
    }


# Wrapped so ADK pauses for explicit human approval before executing.
place_grocery_order_tool = FunctionTool(place_grocery_order, require_confirmation=True)
