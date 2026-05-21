"""Visible Mail automation — compose, edit, send with confirmation."""

from __future__ import annotations

import time
from typing import Any

from jarvis.config import get_settings
from jarvis.tools.applescript import run_applescript

_mock_draft: dict[str, str] = {"to": "", "subject": "", "body": ""}
TYPE_DELAY_MS = 25


def _use_mock() -> bool:
    return get_settings().jarvis_eval_mock


def _type_visible(field_script: str, text: str) -> dict[str, Any]:
    """Set field with character delay for visibility."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    # Set field then type char by char for short bodies; long bodies set directly
    if len(text) <= 200:
        chars = "".join(
            f' delay {TYPE_DELAY_MS / 1000.0} '
            for _ in text
        )
        # AppleScript keystroke per char is slow; use set for long, keystroke chunk for short
        script = f'''
        {field_script}
        set theText to "{escaped}"
        repeat with c in characters of theText
            keystroke c
            delay {TYPE_DELAY_MS / 1000.0}
        end repeat
        '''
    else:
        script = f'{field_script}\nset value of field 1 to "{escaped}"'
    return run_applescript(script, timeout=120.0)


def mail_compose() -> dict[str, Any]:
    global _mock_draft
    if _use_mock():
        _mock_draft = {"to": "", "subject": "", "body": ""}
        return {"ok": True, "message": "mock compose"}

    script = '''
    tell application "Mail"
        activate
        set newMsg to make new outgoing message with properties {visible:true}
    end tell
    delay 0.5
    '''
    r = run_applescript(script)
    return {"ok": r["ok"], "error": r.get("stderr")}


def mail_set_to(addresses: str) -> dict[str, Any]:
    if _use_mock():
        _mock_draft["to"] = addresses
        return {"ok": True}
    script = f'''
    tell application "Mail"
        activate
        set theMsg to item 1 of (every outgoing message whose visible is true)
        tell theMsg
            make new to recipient at end of to recipients with properties {{address:"{addresses.replace('"', '\\"')}"}}
        end tell
    end tell
    '''
    return run_applescript(script)


def mail_set_subject(subject: str) -> dict[str, Any]:
    if _use_mock():
        _mock_draft["subject"] = subject
        return {"ok": True}
    esc = subject.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Mail"
        activate
        set theMsg to item 1 of (every outgoing message whose visible is true)
        set subject of theMsg to "{esc}"
    end tell
    '''
    return run_applescript(script)


def mail_set_body(body: str, visible_type: bool = True) -> dict[str, Any]:
    if _use_mock():
        _mock_draft["body"] = body
        return {"ok": True}
    esc = body.replace("\\", "\\\\").replace('"', '\\"')
    if visible_type and len(body) < 150:
        script = f'''
        tell application "Mail" to activate
        tell application "System Events"
            tell process "Mail"
                set frontmost to true
                delay 0.3
                keystroke "{esc}"
            end tell
        end tell
        '''
        # Fallback: direct content set
        fb = f'''
        tell application "Mail"
            set theMsg to item 1 of (every outgoing message whose visible is true)
            set content of theMsg to "{esc}"
        end tell
        '''
        r = run_applescript(script)
        if not r["ok"]:
            return run_applescript(fb)
        return r
    script = f'''
    tell application "Mail"
        set theMsg to item 1 of (every outgoing message whose visible is true)
        set content of theMsg to "{esc}"
    end tell
    '''
    return run_applescript(script)


def mail_get_draft_state() -> dict[str, Any]:
    if _use_mock():
        return {"ok": True, **_mock_draft}
    script = '''
    tell application "Mail"
        set theMsg to item 1 of (every outgoing message whose visible is true)
        set subj to subject of theMsg
        set cnt to content of theMsg
        return subj & "|||" & cnt
    end tell
    '''
    r = run_applescript(script)
    if r["ok"] and "|||" in r["stdout"]:
        subj, body = r["stdout"].split("|||", 1)
        return {"ok": True, "subject": subj.strip(), "body": body.strip()[:500]}
    return {"ok": False, "error": r.get("stderr", "no draft")}


def mail_send(confirm_token: str, expected_token: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    if settings.require_confirm_send:
        if not expected_token or confirm_token != expected_token:
            return {"ok": False, "error": "send blocked: confirmation required"}

    if _use_mock():
        return {"ok": True, "sent": True, "mock": True}

    script = '''
    tell application "Mail"
        set theMsg to item 1 of (every outgoing message whose visible is true)
        send theMsg
    end tell
    '''
    return run_applescript(script)


def mail_close_without_sending() -> dict[str, Any]:
    if _use_mock():
        return {"ok": True}
    script = '''
    tell application "Mail"
        try
            close (item 1 of (every outgoing message whose visible is true)) saving no
        end try
    end tell
    '''
    return run_applescript(script)
