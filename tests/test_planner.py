import os

os.environ["JARVIS_EVAL_MOCK"] = "1"

import pytest

from jarvis.cognition.planner import build_plan
from jarvis.cognition.router import route_intent


@pytest.mark.asyncio
async def test_compound_route():
    r = await route_intent(
        "Create event Lunch on 2026-05-22 12:00 for one hour and create document Lunch Notes and write Menu"
    )
    assert r["intent"] == "compound"


@pytest.mark.asyncio
async def test_compound_plan_min_steps():
    utterance = (
        "Create event Lunch on 2026-05-22 12:00 for one hour and "
        "create document Lunch Notes and write Menu"
    )
    r = await route_intent(utterance)
    plan = await build_plan(utterance, r["intent"], r.get("slots", {}))
    assert len(plan) >= 2
    tools = [s["tool"] for s in plan]
    assert "create_event" in tools
    assert "doc_create_markdown" in tools
