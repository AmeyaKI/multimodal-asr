"""Bring native apps to the foreground and show actions visibly."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from jarvis.config import get_settings
from jarvis.tools.applescript import run_applescript


def _visible_enabled() -> bool:
    settings = get_settings()
    if settings.jarvis_eval_mock:
        return False
    return settings.show_actions_visually


def _pause() -> None:
    time.sleep(get_settings().step_pause_sec)


def activate_app(app_name: str) -> dict[str, Any]:
    if not _visible_enabled():
        return {"ok": True, "skipped": True}
    r = run_applescript(f'tell application "{app_name}" to activate')
    _pause()
    return r


def type_text_visible(app_process: str, text: str) -> dict[str, Any]:
    """Type text character-by-character in the frontmost window (needs Accessibility)."""
    if not _visible_enabled() or not text:
        return {"ok": True, "skipped": True}
    delay = get_settings().visible_typing_delay_ms / 1000.0
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "{app_process}" to activate
    delay 0.4
    tell application "System Events"
        tell process "{app_process}"
            set frontmost to true
            delay 0.2
            keystroke "{escaped}"
        end tell
    end tell
    '''
    # For long text, set in chunks with delay between (Mail body field)
    if len(text) > 120:
        return _set_mail_body_chunked(text)
    return run_applescript(script, timeout=120.0)


def _set_mail_body_chunked(body: str) -> dict[str, Any]:
    """Set Mail body in Mail.app with activate between (visible jump to Mail)."""
    esc = body.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Mail" to activate
    delay 0.5
    tell application "Mail"
        set theMsg to item 1 of (every outgoing message whose visible is true)
        set content of theMsg to ""
    end tell
    delay 0.3
    tell application "System Events"
        tell process "Mail"
            set frontmost to true
            keystroke "{esc}"
        end tell
    end tell
    '''
    return run_applescript(script, timeout=120.0)


def calendar_open_at_date(start_iso: str) -> dict[str, Any]:
    """Open Calendar.app and jump to the event date."""
    if not _visible_enabled():
        return {"ok": True, "skipped": True}
    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    script = f'''
    tell application "Calendar"
        activate
        delay 0.5
        set targetDate to (current date)
        set year of targetDate to {dt.year}
        set month of targetDate to {dt.month}
        set day of targetDate to {dt.day}
        set hours of targetDate to {dt.hour}
        set minutes of targetDate to {dt.minute}
        set seconds of targetDate to 0
        view calendar at targetDate
    end tell
    delay 0.5
    '''
    return run_applescript(script, timeout=30.0)


def calendar_show_event(title: str, start_iso: str) -> dict[str, Any]:
    """Open Calendar and highlight the event by title (after EventKit create)."""
    if not _visible_enabled():
        return {"ok": True, "skipped": True}
    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    esc_title = title.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Calendar"
        activate
        delay 0.4
        set targetDate to (current date)
        set year of targetDate to {dt.year}
        set month of targetDate to {dt.month}
        set day of targetDate to {dt.day}
        set hours of targetDate to {dt.hour}
        set minutes of targetDate to {dt.minute}
        set seconds of targetDate to 0
        view calendar at targetDate
        delay 0.5
        repeat with cal in calendars
            try
                set matches to (every event of cal whose summary is "{esc_title}")
                if (count of matches) > 0 then
                    show item 1 of matches
                    exit repeat
                end if
            end try
        end repeat
    end tell
    delay 0.3
    '''
    return run_applescript(script, timeout=45.0)


def mail_show_compose_window() -> dict[str, Any]:
    """Ensure Mail is front with compose window visible."""
    if not _visible_enabled():
        return {"ok": True, "skipped": True}
    script = '''
    tell application "Mail"
        activate
        delay 0.5
    end tell
    '''
    return run_applescript(script)


def doc_show_and_type(path: str, text: str, append: bool = False) -> dict[str, Any]:
    """Open document in TextEdit and type new text visibly."""
    if not _visible_enabled():
        return {"ok": True, "skipped": True}
    esc_path = path.replace("\\", "\\\\").replace('"', '\\"')
    if append and text:
        type_text_visible("TextEdit", text)
        return {"ok": True}
    script = f'''
    tell application "TextEdit"
        activate
        open POSIX file "{esc_path}"
        delay 0.5
    end tell
    '''
    r = run_applescript(script)
    if r["ok"] and text:
        _pause()
        type_text_visible("TextEdit", text)
    return r


def reveal_for_tool(tool: str, result: dict[str, Any], args: dict[str, Any]) -> None:
    """After a successful tool call, show the result in the native app."""
    if not _visible_enabled() or not result.get("ok"):
        return

    if tool == "create_event":
        calendar_show_event(
            result.get("title") or args.get("title", ""),
            result.get("start_iso") or args.get("start_iso", ""),
        )
    elif tool == "mail_compose":
        mail_show_compose_window()
    elif tool in ("mail_set_to", "mail_set_subject", "mail_set_body"):
        mail_show_compose_window()
    elif tool == "doc_create_markdown" and result.get("path"):
        doc_show_and_type(result["path"], "", append=False)
    elif tool == "doc_append_text" and result.get("path"):
        doc_show_and_type(
            result.get("path") or args.get("path", ""),
            args.get("text", ""),
            append=True,
        )
    elif tool == "open_app" and args.get("name"):
        activate_app(args["name"])
