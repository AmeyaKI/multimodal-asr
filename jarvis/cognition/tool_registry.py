"""Map tool names to callables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from jarvis.core.policy import check_policy
from jarvis.tools import calendar_eventkit, docs_automation, mail_automation, system

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_calendars": calendar_eventkit.list_calendars,
    "create_event": calendar_eventkit.create_event,
    "get_event": calendar_eventkit.get_event,
    "delete_event": calendar_eventkit.delete_event,
    "mail_compose": mail_automation.mail_compose,
    "mail_set_to": mail_automation.mail_set_to,
    "mail_set_subject": mail_automation.mail_set_subject,
    "mail_set_body": mail_automation.mail_set_body,
    "mail_get_draft_state": mail_automation.mail_get_draft_state,
    "mail_send": mail_automation.mail_send,
    "mail_close_without_sending": mail_automation.mail_close_without_sending,
    "doc_create_markdown": docs_automation.doc_create_markdown,
    "doc_append_text": docs_automation.doc_append_text,
    "doc_read": docs_automation.doc_read,
    "open_app": system.open_app,
    "get_frontmost_app": system.get_frontmost_app,
}


def execute_tool(name: str, args: dict[str, Any], confirm_token: str | None = None) -> dict[str, Any]:
    allowed, reason = check_policy(name, confirm_token)
    if not allowed:
        return {"ok": False, "error": reason}

    fn = TOOLS.get(name)
    if not fn:
        return {"ok": False, "error": f"unknown tool: {name}"}
    if name == "mail_send":
        args = dict(args)
        args.setdefault("confirm_token", confirm_token or "")
        args.setdefault("expected_token", confirm_token)
    if name == "doc_append_text" and "path" in args:
        p = str(args["path"]).replace("~/", "")
        args["path"] = str(Path.home() / p)
    try:
        out = fn(**args)
        if isinstance(out, dict) and "ok" not in out:
            out["ok"] = True
        return out
    except TypeError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
