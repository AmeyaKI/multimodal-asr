"""Orchestrator entry — delegates to LangGraph."""

from __future__ import annotations

from jarvis.config import get_settings
from jarvis.core.events import EventBus
from jarvis.core.graph import run_graph
from jarvis.core.state import AssistantState
from jarvis.memory.store import MemoryStore


async def run_orchestrator(
    transcript: str,
    session_id: str | None = None,
    bus: EventBus | None = None,
) -> AssistantState:
    from jarvis.core.events import get_bus

    bus = bus or get_bus()
    settings = get_settings()
    store = MemoryStore(settings.db_path)
    session_id = session_id or store.new_session()
    store.log_turn(session_id, "user", transcript)

    state = await run_graph(transcript, session_id, bus)

    if state.get("plan"):
        for i, step in enumerate(state["plan"]):
            if i < state.get("plan_index", 0):
                store.log_tool_call(
                    session_id,
                    step.get("tool", ""),
                    step.get("args", {}),
                    str(state.get("last_tool_result", "")),
                )

    store.log_turn(session_id, "assistant", state.get("response_text", ""))
    return state
