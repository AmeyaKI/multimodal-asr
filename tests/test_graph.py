import os

os.environ["JARVIS_EVAL_MOCK"] = "1"

import pytest

from jarvis.core.graph import run_graph


@pytest.mark.asyncio
async def test_graph_calendar():
    state = await run_graph(
        "Create event Team Standup on 2026-05-22 10:00 for 30 minutes",
        "test-session",
    )
    assert state.get("done")
    assert state.get("verification_ok") or state.get("response_text")
