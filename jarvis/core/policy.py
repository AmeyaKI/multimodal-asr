"""Safety policy tiers for tools."""

from __future__ import annotations

from enum import Enum

from jarvis.config import get_settings

class PolicyTier(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


TOOL_POLICY: dict[str, PolicyTier] = {
    "list_calendars": PolicyTier.READ,
    "get_event": PolicyTier.READ,
    "mail_get_draft_state": PolicyTier.READ,
    "doc_read": PolicyTier.READ,
    "get_frontmost_app": PolicyTier.READ,
    "create_event": PolicyTier.WRITE,
    "mail_compose": PolicyTier.WRITE,
    "mail_set_to": PolicyTier.WRITE,
    "mail_set_subject": PolicyTier.WRITE,
    "mail_set_body": PolicyTier.WRITE,
    "doc_create_markdown": PolicyTier.WRITE,
    "doc_append_text": PolicyTier.WRITE,
    "mail_send": PolicyTier.DESTRUCTIVE,
    "delete_event": PolicyTier.DESTRUCTIVE,
}


def check_policy(tool: str, confirm_token: str | None = None) -> tuple[bool, str]:
    tier = TOOL_POLICY.get(tool, PolicyTier.WRITE)
    settings = get_settings()
    if tier == PolicyTier.DESTRUCTIVE and tool == "mail_send":
        if settings.require_confirm_send and not confirm_token:
            return False, "destructive action blocked: confirmation required"
    return True, "ok"
