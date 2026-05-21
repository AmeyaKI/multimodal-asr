"""Typed session and task state for LangGraph."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class Intent(str, Enum):
    CHAT = "chat"
    CALENDAR = "calendar"
    MAIL = "mail"
    DOCS = "docs"
    SYSTEM = "system"
    COMPOUND = "compound"


class PlanStep(TypedDict, total=False):
    tool: str
    args: dict[str, Any]
    success_criteria: str
    domain: str


class AssistantState(TypedDict, total=False):
    """LangGraph state."""

    transcript: str
    intent: str
    slots: dict[str, Any]
    plan: list[PlanStep]
    plan_index: int
    last_tool_result: dict[str, Any]
    verification_ok: bool
    verification_message: str
    needs_confirm: bool
    confirm_token: str | None
    confirm_action: str | None
    response_text: str
    error: str | None
    session_id: str
    ax_context: str | None
    done: bool
