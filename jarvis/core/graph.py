"""LangGraph compiled orchestrator graph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from jarvis.core.state import AssistantState
from jarvis.cognition.planner import build_plan
from jarvis.cognition.router import route_intent
from jarvis.cognition.tool_registry import execute_tool
from jarvis.cognition.verifier import verify_step
from jarvis.core.events import Event, EventBus, EventType


async def node_route(state: AssistantState) -> AssistantState:
    route = await route_intent(state.get("transcript", ""))
    state["intent"] = route.get("intent", "chat")
    state["slots"] = route.get("slots", {})
    if route.get("needs_clarification"):
        state["response_text"] = route.get("clarification", "Can you clarify?")
        state["done"] = True
    return state


async def node_plan(state: AssistantState) -> AssistantState:
    if state.get("done"):
        return state
    if state.get("intent") == "chat":
        from langchain_core.messages import HumanMessage, SystemMessage
        from jarvis.cognition.llm import get_chat_model

        llm = get_chat_model()
        resp = await llm.ainvoke(
            [
                SystemMessage(content="Brief helpful reply."),
                HumanMessage(content=state.get("transcript", "")),
            ]
        )
        state["response_text"] = str(resp.content)
        state["done"] = True
        return state

    plan = await build_plan(
        state.get("transcript", ""),
        state.get("intent", ""),
        state.get("slots", {}),
    )
    state["plan"] = plan
    state["plan_index"] = 0
    if not plan:
        state["response_text"] = "I couldn't plan that action."
        state["done"] = True
    return state


async def node_execute(state: AssistantState, bus: EventBus | None = None) -> AssistantState:
    if state.get("done"):
        return state
    plan = state.get("plan", [])
    i = state.get("plan_index", 0)
    if i >= len(plan):
        state["response_text"] = "Done."
        state["verification_ok"] = True
        state["done"] = True
        return state

    step = plan[i]
    if bus:
        label = f"{step.get('domain', '')}: {step.get('tool', '')}"
        await bus.publish(Event(EventType.STEP_STARTED, {"label": label, "index": i}))

    if step.get("tool") == "mail_send":
        import secrets

        token = secrets.token_hex(8)
        state["confirm_token"] = token
        state["needs_confirm"] = True
        if bus:
            await bus.publish(
                Event(
                    EventType.NEED_CONFIRM,
                    {"action": "send_email", "token": token, "label": "Send email?"},
                )
            )
            await bus.publish(Event(EventType.STATE_CHANGED, {"state": "awaiting_confirm"}))
            ok = await bus.wait_confirm(token)
            if not ok:
                from jarvis.tools import mail_automation

                mail_automation.mail_close_without_sending()
                state["response_text"] = "Send cancelled."
                state["done"] = True
                return state

    result = execute_tool(
        step.get("tool", ""),
        step.get("args", {}),
        confirm_token=state.get("confirm_token"),
    )
    ok, msg = verify_step(step, result)
    if not ok:
        from jarvis.perception.screen import get_ax_tree_frontmost

        state["ax_context"] = get_ax_tree_frontmost()
        result = execute_tool(
            step.get("tool", ""),
            step.get("args", {}),
            confirm_token=state.get("confirm_token"),
        )
        ok, msg = verify_step(step, result)

    if not ok:
        state["error"] = msg
        state["response_text"] = f"Couldn't complete that: {msg}"
        state["done"] = True
        return state

    if step.get("tool") == "doc_create_markdown" and result.get("path"):
        for j in range(i + 1, len(plan)):
            if plan[j].get("tool") == "doc_append_text":
                plan[j]["args"]["path"] = result["path"]

    state["last_tool_result"] = result
    state["plan_index"] = i + 1
    if bus:
        await bus.publish(Event(EventType.STEP_DONE, {"label": step.get("tool", ""), "ok": True}))
    if state["plan_index"] >= len(plan):
        state["response_text"] = "Done."
        state["verification_ok"] = True
        state["done"] = True
    return state


def _should_continue(state: AssistantState) -> str:
    if state.get("done"):
        return "end"
    plan = state.get("plan", [])
    if state.get("plan_index", 0) >= len(plan):
        return "end"
    return "execute"


def build_graph(bus: EventBus | None = None):
    graph = StateGraph(AssistantState)

    async def execute_wrapper(s: AssistantState) -> AssistantState:
        return await node_execute(s, bus)

    graph.add_node("route", node_route)
    graph.add_node("plan", node_plan)
    graph.add_node("execute", execute_wrapper)
    graph.set_entry_point("route")
    graph.add_edge("route", "plan")
    graph.add_conditional_edges(
        "plan",
        lambda s: "end" if s.get("done") else "execute",
        {"execute": "execute", "end": END},
    )
    graph.add_conditional_edges(
        "execute",
        _should_continue,
        {"execute": "execute", "end": END},
    )
    return graph.compile()


async def run_graph(
    transcript: str,
    session_id: str,
    bus: EventBus | None = None,
) -> AssistantState:
    app = build_graph(bus)
    initial: AssistantState = {
        "transcript": transcript,
        "session_id": session_id,
        "plan_index": 0,
        "done": False,
    }
    if bus:
        await bus.publish(Event(EventType.STATE_CHANGED, {"state": "thinking"}))
    final = await app.ainvoke(initial)
    return final
