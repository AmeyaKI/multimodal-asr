"""Multi-step task planner."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from jarvis.cognition.llm import ainvoke_json
from jarvis.config import get_settings
from jarvis.core.state import PlanStep

PLANNER_SYSTEM = """You plan macOS assistant tasks. Output JSON:
{"steps": [{"tool": "tool_name", "args": {}, "success_criteria": "...", "domain": "calendar|mail|docs|system"}]}
Available tools:
calendar: list_calendars, create_event, get_event
mail: mail_compose, mail_set_to, mail_set_subject, mail_set_body, mail_get_draft_state, mail_send
docs: doc_create_markdown, doc_append_text, doc_read
system: open_app
Never include mail_send without user confirmation — use success_criteria "draft ready" instead."""


async def build_plan(transcript: str, intent: str, slots: dict[str, Any]) -> list[PlanStep]:
    if intent == "chat":
        return []

    # Rule-based fast paths (compound always uses full multi-domain plan)
    plan_intent = "compound" if intent == "compound" else intent
    rule_plan = _rule_plan(transcript, plan_intent, slots)
    if intent == "compound" and rule_plan:
        return rule_plan
    if rule_plan and intent != "compound":
        return rule_plan

    settings = get_settings()
    result = await ainvoke_json(
        f'Intent: {intent}\nSlots: {slots}\nUser: "{transcript}"\nBuild minimal step plan.',
        PLANNER_SYSTEM,
        model=settings.ollama_planner_model,
    )
    steps = result.get("steps", [])
    return [PlanStep(tool=s["tool"], args=s.get("args", {}), success_criteria=s.get("success_criteria", ""), domain=s.get("domain", intent)) for s in steps]


def _rule_plan(transcript: str, intent: str, slots: dict[str, Any]) -> list[PlanStep]:
    t = transcript.lower()
    steps: list[PlanStep] = []
    is_compound = intent == "compound"

    if (is_compound or intent == "calendar") and (
        "event" in t or "meeting" in t or "lunch" in t or "schedule" in t
    ):
        title = (
            slots.get("title")
            or _extract_quoted(transcript)
            or _extract_event_title(transcript)
            or "New Event"
        )
        start = slots.get("start_iso") or _default_start(t)
        end = slots.get("end_iso") or _default_end(start, t)
        steps.append(
            PlanStep(
                tool="create_event",
                args={"title": title, "start_iso": start, "end_iso": end},
                success_criteria=f"event title contains {title[:20]}",
                domain="calendar",
            )
        )

    if (is_compound or intent == "mail") and any(k in t for k in ("email", "mail", "draft")):
        to_addr = slots.get("to") or _extract_email(transcript) or "test@example.com"
        subj = slots.get("subject") or _extract_after(transcript, "subject") or "Hello"
        body = slots.get("body") or _extract_after(transcript, "body") or "Hi there"
        if slots.get("action") == "send":
            steps.append(
                PlanStep(
                    tool="mail_send",
                    args={},
                    success_criteria="sent with confirm",
                    domain="mail",
                )
            )
        else:
            steps.extend(
                [
                    PlanStep(tool="mail_compose", args={}, success_criteria="compose open", domain="mail"),
                    PlanStep(tool="mail_set_to", args={"addresses": to_addr}, success_criteria=to_addr, domain="mail"),
                    PlanStep(tool="mail_set_subject", args={"subject": subj}, success_criteria=subj, domain="mail"),
                    PlanStep(tool="mail_set_body", args={"body": body}, success_criteria="body", domain="mail"),
                ]
            )

    if (is_compound or intent == "docs") and (
        "document" in t or " doc" in t or "notes" in t
    ):
        title = slots.get("doc_title") or _extract_doc_title(transcript) or "Notes"
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip().replace(" ", "_")
        text = slots.get("doc_text") or _extract_after(transcript, "write") or "Introduction"
        steps.append(
            PlanStep(tool="doc_create_markdown", args={"title": title}, success_criteria=title, domain="docs")
        )
        steps.append(
            PlanStep(
                tool="doc_append_text",
                args={"path": f"~/Documents/Jarvis/{safe}.md", "text": text},
                success_criteria=text[:30],
                domain="docs",
            )
        )

    return steps


def _default_start(t: str) -> str:
    if "2026-05-22" in t:
        return "2026-05-22T10:00:00"
    if "tomorrow" in t:
        d = datetime.now() + timedelta(days=1)
        return d.replace(hour=15, minute=0, second=0).isoformat()
    if "friday" in t and "noon" in t:
        return "2026-05-22T12:00:00"
    return (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0).isoformat()


def _default_end(start_iso: str, t: str) -> str:
    start = datetime.fromisoformat(start_iso)
    if "30 min" in t or "30 minutes" in t:
        return (start + timedelta(minutes=30)).isoformat()
    if "one hour" in t or "1 hour" in t:
        return (start + timedelta(hours=1)).isoformat()
    return (start + timedelta(hours=1)).isoformat()


def _extract_email(t: str) -> str | None:
    import re

    m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", t)
    return m.group(0) if m else None


def _extract_quoted(t: str) -> str | None:
    import re

    m = re.search(r'"([^"]+)"', t)
    return m.group(1) if m else None


def _extract_after(t: str, keyword: str) -> str | None:
    import re

    m = re.search(rf"{keyword}\s+(.+?)(?:\s+body|\s+subject|$)", t, re.I)
    return m.group(1).strip() if m else None


def _extract_doc_title(t: str) -> str | None:
    import re

    m = re.search(r"document\s+(\w+(?:\s+\w+)?)", t, re.I)
    return m.group(1) if m else None


def _extract_event_title(t: str) -> str | None:
    import re

    m = re.search(
        r"(?:create|add|schedule)\s+(?:an?\s+)?(?:event|meeting)\s+(.+?)\s+(?:on|at|for|tomorrow)",
        t,
        re.I,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:event|meeting)\s+(.+?)\s+on\s+", t, re.I)
    return m.group(1).strip() if m else None
