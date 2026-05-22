"""Intent router via local LLM."""

from __future__ import annotations

from typing import Any

from jarvis.cognition.llm import ainvoke_json
from jarvis.config import get_settings

ROUTER_SYSTEM = """You route macOS voice commands. Output JSON only:
{"intent": "chat|calendar|mail|docs|system|compound", "slots": {}, "needs_clarification": false, "clarification": ""}
intents:
- calendar: create/list events, schedule meetings
- mail: draft, email, send (compound with send needs mail)
- docs: create document, write file, notes
- system: open app, run shortcut
- compound: multiple domains in one request
- chat: general knowledge, no macOS action
Extract slots: title, start_iso, end_iso, to, subject, body, doc_title, doc_text, app_name.
Use ISO-8601 dates when possible (2026-05-22T15:00:00)."""


async def route_intent(transcript: str) -> dict[str, Any]:
    rule = _rule_route(transcript)
    if rule.get("intent") == "compound":
        return rule
    settings = get_settings()
    result = await ainvoke_json(
        f'User said: "{transcript}"',
        ROUTER_SYSTEM,
        model=settings.ollama_model,
    )
    if not result.get("intent"):
        return rule
    # Prefer rule-based compound if LLM missed it
    if rule.get("intent") == "compound":
        return rule
    return result


def _rule_route(transcript: str) -> dict[str, Any]:
    t = transcript.lower()
    slots: dict[str, Any] = {}
    has_cal = any(k in t for k in ("calendar", "event", "meeting", "schedule", "lunch"))
    has_mail = any(k in t for k in ("email", "mail", "draft"))
    has_doc = any(k in t for k in ("document", "doc", "notes", "write"))
    domains = sum([has_cal, has_mail, has_doc])
    if domains >= 2:
        return {"intent": "compound", "slots": slots, "needs_clarification": False}
    if has_cal:
        return {"intent": "calendar", "slots": slots, "needs_clarification": False}
    if "email" in t or "mail" in t or "draft" in t:
        intent = "mail"
        if "send" in t:
            slots["action"] = "send"
        return {"intent": intent, "slots": slots, "needs_clarification": False}
    if "document" in t or "doc" in t or "write" in t and "file" in t:
        return {"intent": "docs", "slots": slots, "needs_clarification": False}
    if "open" in t and "app" not in t:
        return {"intent": "system", "slots": slots, "needs_clarification": False}
    return {"intent": "chat", "slots": {}, "needs_clarification": False}
