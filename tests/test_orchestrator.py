import os

os.environ["JARVIS_EVAL_MOCK"] = "1"

import pytest

from jarvis.cognition.planner import build_plan
from jarvis.cognition.router import route_intent


@pytest.mark.asyncio
async def test_route_calendar():
    r = await route_intent("Create a meeting tomorrow at 3pm")
    assert r["intent"] in ("calendar", "compound", "chat")


@pytest.mark.asyncio
async def test_plan_calendar():
    plan = await build_plan(
        "Create event Team Standup on 2026-05-22 10:00 for 30 minutes",
        "calendar",
        {},
    )
    assert any(s.get("tool") == "create_event" for s in plan)


@pytest.mark.asyncio
async def test_plan_mail_blocked_send():
    from jarvis.core.policy import check_policy

    ok, _ = check_policy("mail_send", None)
    assert not ok
