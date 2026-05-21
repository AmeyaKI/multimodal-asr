"""Post-condition verification for tool steps."""

from __future__ import annotations

from typing import Any

from jarvis.core.state import PlanStep


def verify_step(step: PlanStep, result: dict[str, Any]) -> tuple[bool, str]:
    if not result.get("ok", True) and result.get("error"):
        return False, result.get("error", "tool failed")

    criteria = (step.get("success_criteria") or "").lower()
    tool = step.get("tool", "")

    if tool == "create_event":
        title = result.get("title", "")
        if criteria and criteria.split()[-1] not in title.lower() and title.lower() not in criteria:
            # loose match
            if "event" not in criteria and title:
                return True, "event created"
        return bool(result.get("event_id")), "event verified"

    if tool.startswith("mail_"):
        if tool == "mail_send":
            if result.get("error") and "confirm" in str(result.get("error", "")).lower():
                return False, result["error"]
            return result.get("ok", False), "send status"
        if tool == "mail_set_subject":
            state = __import__("jarvis.tools.mail_automation", fromlist=["mail_get_draft_state"]).mail_get_draft_state()
            if state.get("ok") and criteria:
                return criteria.lower() in (state.get("subject") or "").lower(), "subject check"
        return result.get("ok", True), "mail step ok"

    if tool.startswith("doc_"):
        if tool == "doc_append_text" and criteria:
            path = result.get("path", step.get("args", {}).get("path", ""))
            from jarvis.tools import docs_automation

            read = docs_automation.doc_read(path)
            if read.get("ok"):
                return criteria.lower() in read.get("content", "").lower(), "doc content check"
        return result.get("ok", True), "doc step ok"

    return result.get("ok", True), "ok"
