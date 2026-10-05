"""Unit tests for deterministic domain logic and tools (no LLM calls)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import catalog, tools
from app.callbacks import order_budget_guard
from app.schemas import DietaryProfile


class FakeToolContext(SimpleNamespace):
    """Minimal stand-in for ToolContext: tools only use ``.state``."""

    def __init__(self, state: dict | None = None) -> None:
        super().__init__(state=state if state is not None else {})


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
def test_catalog_loads_with_cost_per_serving() -> None:
    recipes = catalog.load_catalog()
    assert len(recipes) >= 30
    for r in recipes.values():
        assert r["cost_per_serving"] > 0
        assert set(r["allergens"]) <= set(catalog.ALLERGENS)
        assert r["diet"] in catalog.DIET_RANK


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Peanut", "peanuts"),
        ("milk", "dairy"),
        ("shrimp", "shellfish"),
        ("tree nuts", "tree_nuts"),
        ("wheat", "gluten"),
        ("kryptonite", None),
    ],
)
def test_normalize_allergen(raw: str, expected: str | None) -> None:
    assert catalog.normalize_allergen(raw) == expected


def test_normalize_ingredient_snaps_plurals() -> None:
    assert catalog.normalize_ingredient(" Carrots ") == "carrot"
    assert catalog.normalize_ingredient("egg") == "eggs"
    assert catalog.normalize_ingredient("dragonfruit") == "dragonfruit"


def test_search_respects_diet_allergens_and_prep() -> None:
    results = catalog.search_catalog(
        diet="vegetarian", exclude_allergens=["peanuts"], max_prep_minutes=20
    )
    assert results
    rec = catalog.load_catalog()
    for r in results:
        assert "peanuts" not in r["allergens"]
        assert r["prep_minutes"] <= 20
        assert catalog.fits_diet(rec[r["recipe_id"]]["diet"], "vegetarian")


def test_search_excludes_disliked_ingredients() -> None:
    results = catalog.search_catalog(exclude_ingredients=["mushroom"])
    assert all(r["recipe_id"] != "r13" for r in results)


def test_check_plan_flags_allergen_and_diet_as_unsafe() -> None:
    profile = DietaryProfile(diet="vegetarian", allergies=["peanuts"])
    violations, unsafe = catalog.check_plan(
        {1: "r02", 2: "r07", 3: "r05"}, profile, requested_days=3
    )
    assert unsafe == {1, 2}  # peanut noodles + chicken stir-fry
    assert any("allergen" in v for v in violations)


def test_check_plan_detects_wrong_day_count_and_unknown_id() -> None:
    violations, unsafe = catalog.check_plan(
        {1: "r05", 2: "zzz"}, DietaryProfile(), requested_days=3
    )
    assert 2 in unsafe
    assert any("3 were requested" in v for v in violations)


def test_shopping_list_subtracts_pantry_and_scales() -> None:
    base = catalog.build_shopping_list(["r01"], servings=2, pantry=[])
    with_pantry = catalog.build_shopping_list(
        ["r01"], servings=2, pantry=["rice", "onions"]
    )
    names = {i.name for i in with_pantry.items}
    assert "rice" not in names and "onion" not in names
    assert set(with_pantry.skipped_from_pantry) == {"onion", "rice"}
    assert with_pantry.total_cost == pytest.approx(base.total_cost - 1.1)
    doubled = catalog.build_shopping_list(["r01"], servings=4, pantry=[])
    assert doubled.total_cost == pytest.approx(base.total_cost * 2)


def test_plan_budget_prorates() -> None:
    assert catalog.plan_budget(70, 5) == 50
    assert catalog.plan_budget(70, 10) == 70


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def test_set_profile_normalizes_and_merges() -> None:
    ctx = FakeToolContext()
    res = tools.set_dietary_profile(
        ctx, diet="Vegetarian", allergies=["Peanut", "glitter"]
    )
    assert res["status"] == "success"
    assert res["profile"]["allergies"] == ["peanuts"]
    assert res["warnings"]
    res2 = tools.set_dietary_profile(ctx, servings=4)
    assert res2["profile"]["diet"] == "vegetarian"
    assert res2["profile"]["allergies"] == ["peanuts"]
    assert ctx.state[tools.PROFILE_KEY]["servings"] == 4


def test_set_profile_rejects_invalid_values() -> None:
    res = tools.set_dietary_profile(FakeToolContext(), diet="carnivore")
    assert res["status"] == "error"


def test_search_recipes_always_applies_profile_allergies() -> None:
    ctx = FakeToolContext({tools.PROFILE_KEY: {"diet": "vegan", "allergies": ["soy"]}})
    res = tools.search_recipes(ctx, diet="omnivore", limit=30)  # can't loosen diet
    assert res["status"] == "success"
    for r in res["recipes"]:
        assert "soy" not in r["allergens"]
        assert r["diet"] == "vegan"


def test_search_recipes_unknown_diet_is_error() -> None:
    assert tools.search_recipes(FakeToolContext(), diet="keto")["status"] == "error"


def test_get_recipe_details() -> None:
    assert tools.get_recipe_details("R05")["recipe"]["name"] == "Black Bean Tacos"
    assert tools.get_recipe_details("nope")["status"] == "error"


def test_pantry_add_remove() -> None:
    ctx = FakeToolContext()
    assert tools.update_pantry(ctx)["status"] == "error"
    tools.update_pantry(ctx, add=["Eggs", "carrots", "rice"])
    res = tools.update_pantry(ctx, remove=["rice"])
    assert res["items"] == ["carrot", "eggs"]
    assert tools.get_pantry(ctx)["count"] == 2


@pytest.mark.asyncio
async def test_place_order_without_list_is_error() -> None:
    res = await tools.place_grocery_order(FakeToolContext())
    assert res["status"] == "error"


def test_order_tool_requires_confirmation() -> None:
    assert tools.place_grocery_order_tool._require_confirmation is True


def test_budget_guard_blocks_expensive_orders() -> None:
    tool = SimpleNamespace(name="place_grocery_order")
    state = {
        tools.PROFILE_KEY: {"weekly_budget": 70},
        tools.LAST_PLAN_KEY: {"meals": [{}] * 7},
        tools.LAST_LIST_KEY: {"total_cost": 100.0, "items": [{}]},
    }
    blocked = order_budget_guard(tool, {}, FakeToolContext(state))
    assert blocked and blocked["status"] == "error"
    state[tools.LAST_LIST_KEY]["total_cost"] = 60.0
    assert order_budget_guard(tool, {}, FakeToolContext(state)) is None
