"""Deterministic domain logic: recipe catalog, filtering, validation and lists.

Everything here is pure Python with no ADK or LLM dependency. That makes it
unit-testable, and it means the safety-critical rules (allergens, diet, budget)
never depend on what the model decides.
"""

from __future__ import annotations

import functools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.schemas import ALLERGENS, DietaryProfile, ShoppingItem, ShoppingList

_CATALOG_PATH = Path(__file__).parent / "data" / "recipes.json"

DIET_RANK = {"vegan": 0, "vegetarian": 1, "pescatarian": 2, "omnivore": 3}

_ALLERGEN_SYNONYMS = {
    "peanut": "peanuts",
    "peanuts": "peanuts",
    "groundnut": "peanuts",
    "nut": "tree_nuts",
    "nuts": "tree_nuts",
    "tree nut": "tree_nuts",
    "tree nuts": "tree_nuts",
    "tree_nuts": "tree_nuts",
    "cashew": "tree_nuts",
    "almond": "tree_nuts",
    "walnut": "tree_nuts",
    "pine nut": "tree_nuts",
    "dairy": "dairy",
    "milk": "dairy",
    "lactose": "dairy",
    "cheese": "dairy",
    "egg": "eggs",
    "eggs": "eggs",
    "gluten": "gluten",
    "wheat": "gluten",
    "soy": "soy",
    "soya": "soy",
    "fish": "fish",
    "shellfish": "shellfish",
    "shrimp": "shellfish",
    "prawn": "shellfish",
    "crab": "shellfish",
    "lobster": "shellfish",
    "sesame": "sesame",
}

# Budget headroom tolerated at checkout before the guardrail blocks an order.
ORDER_BUDGET_TOLERANCE = 1.2


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
@functools.cache
def load_catalog() -> dict[str, dict[str, Any]]:
    """Loads the recipe catalog keyed by id, adding a computed cost_per_serving."""
    data = json.loads(_CATALOG_PATH.read_text())
    catalog: dict[str, dict[str, Any]] = {}
    for recipe in data["recipes"]:
        total = sum(i["cost"] for i in recipe["ingredients"])
        recipe["cost_per_serving"] = round(total / recipe["servings"], 2)
        catalog[recipe["id"]] = recipe
    return catalog


@functools.cache
def known_ingredients() -> frozenset[str]:
    return frozenset(
        i["name"] for r in load_catalog().values() for i in r["ingredients"]
    )


def normalize_ingredient(name: str) -> str:
    """Lower-cases, trims, and snaps simple plurals to catalog ingredient names."""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    known = known_ingredients()
    if cleaned in known:
        return cleaned
    for candidate in (
        cleaned + "s",
        cleaned + "es",
        cleaned.removesuffix("s"),
        cleaned.removesuffix("es"),
    ):
        if candidate in known:
            return candidate
    return cleaned


def normalize_allergen(name: str) -> str | None:
    """Maps free-text allergen names to catalog allergen keys (None if unknown)."""
    key = re.sub(r"\s+", " ", name.strip().lower()).removesuffix(" allergy")
    if key in ALLERGENS:
        return key
    return _ALLERGEN_SYNONYMS.get(key) or _ALLERGEN_SYNONYMS.get(key.removesuffix("s"))


def _stem(word: str) -> str:
    """Crude plural stripping so 'tomatoes' matches 'tomato', 'mushrooms' 'mushroom'."""
    w = word.strip().lower()
    return w.removesuffix("es") if w.endswith("oes") else w.removesuffix("s")


def fits_diet(recipe_diet: str, user_diet: str) -> bool:
    return DIET_RANK[recipe_diet] <= DIET_RANK.get(user_diet, 3)


def recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    """Compact, LLM-friendly view of a recipe (no full ingredient list)."""
    return {
        "recipe_id": recipe["id"],
        "name": recipe["name"],
        "cuisine": recipe["cuisine"],
        "diet": recipe["diet"],
        "allergens": recipe["allergens"],
        "prep_minutes": recipe["prep_minutes"],
        "cost_per_serving": recipe["cost_per_serving"],
        "kcal_per_serving": recipe["kcal"],
        "protein_g_per_serving": recipe["protein_g"],
    }


def search_catalog(
    diet: str = "omnivore",
    exclude_allergens: list[str] | None = None,
    exclude_ingredients: list[str] | None = None,
    max_prep_minutes: int | None = None,
    max_cost_per_serving: float | None = None,
    cuisine: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Filters the catalog. All filters are hard constraints."""
    allergens = {
        a for a in (normalize_allergen(x) for x in exclude_allergens or []) if a
    }
    dislikes = [_stem(d) for d in exclude_ingredients or [] if d.strip()]
    results = []
    for recipe in load_catalog().values():
        if not fits_diet(recipe["diet"], diet):
            continue
        if allergens & set(recipe["allergens"]):
            continue
        if dislikes and any(
            d in i["name"] for d in dislikes for i in recipe["ingredients"]
        ):
            continue
        if max_prep_minutes and recipe["prep_minutes"] > max_prep_minutes:
            continue
        if max_cost_per_serving and recipe["cost_per_serving"] > max_cost_per_serving:
            continue
        if cuisine and recipe["cuisine"] != cuisine.strip().lower():
            continue
        results.append(recipe_summary(recipe))
    results.sort(key=lambda r: (r["cost_per_serving"], r["prep_minutes"]))
    return results[: max(1, min(limit, 30))]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def plan_budget(weekly_budget: float, days: int) -> float:
    """Pro-rates the weekly budget to the number of planned days."""
    return round(weekly_budget * min(days, 7) / 7, 2)


def check_plan(
    recipe_ids_by_day: dict[int, str],
    profile: DietaryProfile,
    requested_days: int | None = None,
    max_prep_minutes: int | None = None,
) -> tuple[list[str], set[int]]:
    """Validates a plan against hard rules.

    Returns:
        (violations, unsafe_days): human-readable violations, and the days whose
        recipe is unknown or contains an allergen (these must never be shown).
    """
    catalog = load_catalog()
    allergies = {a for a in (normalize_allergen(x) for x in profile.allergies) if a}
    dislikes = [_stem(d) for d in profile.dislikes if d.strip()]
    violations: list[str] = []
    unsafe: set[int] = set()

    if requested_days and len(recipe_ids_by_day) != requested_days:
        violations.append(
            f"Plan has {len(recipe_ids_by_day)} meals but {requested_days} were requested."
        )
    seen: set[str] = set()
    for day, rid in sorted(recipe_ids_by_day.items()):
        recipe = catalog.get(rid)
        if recipe is None:
            violations.append(f"Day {day}: recipe_id '{rid}' is not in the catalog.")
            unsafe.add(day)
            continue
        hit = allergies & set(recipe["allergens"])
        if hit:
            violations.append(
                f"Day {day}: {recipe['name']} contains allergen(s) {sorted(hit)}."
            )
            unsafe.add(day)
        if not fits_diet(recipe["diet"], profile.diet):
            violations.append(
                f"Day {day}: {recipe['name']} is {recipe['diet']}, not {profile.diet}."
            )
            unsafe.add(day)
        bad = [d for d in dislikes for i in recipe["ingredients"] if d in i["name"]]
        if bad:
            violations.append(
                f"Day {day}: {recipe['name']} uses disliked ingredient(s) {sorted(set(bad))}."
            )
        if max_prep_minutes and recipe["prep_minutes"] > max_prep_minutes:
            violations.append(
                f"Day {day}: {recipe['name']} takes {recipe['prep_minutes']} min "
                f"(limit {max_prep_minutes})."
            )
        if rid in seen:
            violations.append(f"Day {day}: {recipe['name']} is repeated.")
        seen.add(rid)
    return violations, unsafe


# ---------------------------------------------------------------------------
# Shopping list
# ---------------------------------------------------------------------------
def _tidy(qty: float) -> int | float:
    """2.0 -> 2, 1.333 -> 1.33 (friendlier shopping lists)."""
    rounded = round(qty, 2)
    return int(rounded) if rounded.is_integer() else rounded


def build_shopping_list(
    recipe_ids: list[str], servings: int, pantry: list[str]
) -> ShoppingList:
    """Aggregates ingredients for the plan, scaled to servings, minus pantry stock.

    Pantry tracking is presence-based (an item is either stocked or not), which
    keeps the demo simple and predictable.
    """
    catalog = load_catalog()
    stocked = {normalize_ingredient(p) for p in pantry}
    totals: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"qty": 0.0, "cost": 0.0, "aisle": ""}
    )
    for rid in recipe_ids:
        recipe = catalog[rid]
        factor = servings / recipe["servings"]
        for ing in recipe["ingredients"]:
            entry = totals[(ing["name"], ing["unit"])]
            entry["qty"] += ing["qty"] * factor
            entry["cost"] += ing["cost"] * factor
            entry["aisle"] = ing["aisle"]

    items: list[ShoppingItem] = []
    skipped: list[str] = []
    savings = 0.0
    for (name, unit), entry in sorted(
        totals.items(), key=lambda kv: (kv[1]["aisle"], kv[0][0])
    ):
        if name in stocked:
            skipped.append(name)
            savings += entry["cost"]
            continue
        items.append(
            ShoppingItem(
                name=name,
                qty=_tidy(entry["qty"]),
                unit=unit,
                aisle=entry["aisle"],
                est_cost=round(entry["cost"], 2),
            )
        )
    return ShoppingList(
        items=items,
        skipped_from_pantry=sorted(set(skipped)),
        total_cost=round(sum(i.est_cost for i in items), 2),
        pantry_savings=round(savings, 2),
    )
