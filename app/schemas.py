"""Typed data contracts shared by tools, workflow nodes and LLM agents.

Using Pydantic models (rather than free text or raw dicts) for every hand-off
keeps the graph deterministic: the planner's output is validated against
``MealPlan`` before any business rule runs on it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Diet = Literal["omnivore", "pescatarian", "vegetarian", "vegan"]

ALLERGENS: tuple[str, ...] = (
    "peanuts",
    "tree_nuts",
    "dairy",
    "eggs",
    "gluten",
    "soy",
    "fish",
    "shellfish",
    "sesame",
)


class Intent(BaseModel):
    """Output of the intake classifier: where to route this turn."""

    route: Literal["profile_or_pantry", "meal_plan", "order", "help", "unrelated"] = (
        Field(
            description=(
                "profile_or_pantry: user states/asks about diet, allergies, "
                "dislikes, household size, budget, or what's in their pantry. "
                "meal_plan: user wants a new meal plan, or to change/swap meals "
                "in the current plan. order: user wants to order/buy/checkout "
                "the groceries. help: greetings or 'what can you do'. "
                "unrelated: anything else."
            )
        )
    )
    days: int | None = Field(
        default=None, description="Number of days/meals requested, if stated (1-7)."
    )
    max_prep_minutes: int | None = Field(
        default=None, description="Maximum prep time per meal, if the user asked."
    )
    request_summary: str = Field(
        description="One sentence restating what the user wants, in their words."
    )
    allergies_mentioned: list[str] = Field(
        default_factory=list,
        description="Any food allergies the user states in THIS message, e.g. ['peanuts'].",
    )
    diet_mentioned: Diet | None = Field(
        default=None,
        description="Diet the user states in THIS message (vegan/vegetarian/...), if any.",
    )


class DietaryProfile(BaseModel):
    """Household dietary profile persisted in ``user:dietary_profile``."""

    diet: Diet = "omnivore"
    allergies: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    servings: int = Field(default=2, ge=1, le=12)
    weekly_budget: float = Field(default=100.0, gt=0)


class PlannedMeal(BaseModel):
    day: int = Field(description="Day number starting at 1.")
    recipe_id: str = Field(description="Catalog recipe id, e.g. 'r05'.")
    recipe_name: str


class MealPlan(BaseModel):
    """Structured output of the meal planner agent."""

    meals: list[PlannedMeal]
    rationale: str = Field(
        description="One or two sentences on why these recipes fit the household."
    )


class ShoppingItem(BaseModel):
    name: str
    qty: int | float
    unit: str
    aisle: str
    est_cost: float


class ShoppingList(BaseModel):
    items: list[ShoppingItem]
    skipped_from_pantry: list[str]
    total_cost: float
    pantry_savings: float
