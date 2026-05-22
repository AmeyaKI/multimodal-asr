"""Calendar tools via PyObjC EventKit."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from jarvis.config import get_settings

_store = None
_mock_events: dict[str, dict[str, Any]] = {}


def _use_mock() -> bool:
    return get_settings().jarvis_eval_mock


def _host_app_hint() -> str:
    """Name of the process macOS ties to Calendar permission."""
    import os
    import sys

    for key in ("TERM_PROGRAM", "CURSOR_TRACE_ID"):
        if os.environ.get(key):
            if key == "TERM_PROGRAM":
                return os.environ["TERM_PROGRAM"]  # Cursor, vscode, Apple_Terminal
            return "Cursor"
    exe = sys.executable
    if "Cursor" in exe:
        return "Cursor"
    if "Code" in exe:
        return "Visual Studio Code"
    return "Terminal (or Cursor)"


def _request_calendar_access(store) -> bool:
    """Request EventKit access and wait for the async completion handler."""
    from EventKit import EKEntityTypeEvent

    done = threading.Event()
    result = {"granted": False, "error": None}

    def handler(granted_flag, error):
        result["granted"] = bool(granted_flag)
        result["error"] = error
        done.set()

    # Prefer full access API on recent macOS
    if hasattr(store, "requestFullAccessToEventsWithCompletion_"):
        store.requestFullAccessToEventsWithCompletion_(handler)
    else:
        store.requestAccessToEntityType_completion_(EKEntityTypeEvent, handler)

    done.wait(timeout=60.0)
    return result["granted"]


def _ensure_calendar_access() -> None:
    from EventKit import (
        EKAuthorizationStatusAuthorized,
        EKAuthorizationStatusFullAccess,
        EKAuthorizationStatusNotDetermined,
        EKAuthorizationStatusWriteOnly,
        EKEntityTypeEvent,
        EKEventStore,
    )

    status = EKEventStore.authorizationStatusForEntityType_(EKEntityTypeEvent)
    authorized = {
        EKAuthorizationStatusAuthorized,
        EKAuthorizationStatusFullAccess,
        EKAuthorizationStatusWriteOnly,
    }
    if status in authorized:
        return

    store = EKEventStore.alloc().init()
    if status == EKAuthorizationStatusNotDetermined:
        if _request_calendar_access(store):
            return
    else:
        # Denied or restricted — try request once more in case user just toggled Settings
        if _request_calendar_access(store):
            return

    app = _host_app_hint()
    raise PermissionError(
        f"Calendar access denied for the app running Python ({app}), not VS Code "
        f"unless you run the command from VS Code's terminal. "
        f"Enable System Settings → Privacy & Security → Calendars → {app}. "
        f"Then quit and reopen that app and retry."
    )


def _get_store():
    global _store
    if _use_mock():
        return None
    if _store is not None:
        return _store
    _ensure_calendar_access()
    from EventKit import EKEventStore

    _store = EKEventStore.alloc().init()
    return _store


def list_calendars() -> list[dict[str, str]]:
    if _use_mock():
        return [{"id": "mock", "title": "Mock Calendar"}]
    from EventKit import EKEntityTypeEvent

    store = _get_store()
    cals = store.calendarsForEntityType_(EKEntityTypeEvent)
    return [
        {"id": str(c.calendarIdentifier()), "title": str(c.title())}
        for c in (cals or [])
    ]


def create_event(
    title: str,
    start_iso: str,
    end_iso: str,
    calendar_id: str | None = None,
    notes: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    if _use_mock():
        import uuid

        eid = str(uuid.uuid4())
        _mock_events[eid] = {
            "ok": True,
            "event_id": eid,
            "title": title,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "url": None,
        }
        return _mock_events[eid]

    from EventKit import EKEntityTypeEvent, EKEvent
    from Foundation import NSDate

    def parse_iso(s: str) -> NSDate:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())

    from jarvis.tools.visibility import calendar_open_at_date

    calendar_open_at_date(start_iso)

    store = _get_store()
    event = EKEvent.eventWithEventStore_(store)
    event.setTitle_(title)
    event.setStartDate_(parse_iso(start_iso))
    event.setEndDate_(parse_iso(end_iso))
    if notes:
        event.setNotes_(notes)
    if location:
        event.setLocation_(location)

    calendar_set = False
    if calendar_id:
        for cal in store.calendarsForEntityType_(EKEntityTypeEvent) or []:
            if str(cal.calendarIdentifier()) == calendar_id:
                event.setCalendar_(cal)
                calendar_set = True
                break
    if not calendar_set:
        default = store.defaultCalendarForNewEvents()
        if default:
            event.setCalendar_(default)
            calendar_set = True
        else:
            # Fallback: first writable calendar (common when default is unset)
            for cal in store.calendarsForEntityType_(EKEntityTypeEvent) or []:
                if cal.allowsContentModifications():
                    event.setCalendar_(cal)
                    calendar_set = True
                    break
    if not calendar_set:
        return {
            "ok": False,
            "error": "No writable calendar found. Add a calendar in Calendar.app.",
        }

    # PyObjC: error out-param must be None or objc.NULL (not a Python list)
    ok, error = store.saveEvent_span_error_(event, 0, None)  # EKSpanThisEvent = 0
    if not ok:
        err_msg = str(error) if error else "save failed"
        return {"ok": False, "error": err_msg}

    eid = str(event.eventIdentifier())
    return {
        "ok": True,
        "event_id": eid,
        "title": title,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "url": None,
    }


def get_event(event_id: str) -> dict[str, Any]:
    if _use_mock():
        ev = _mock_events.get(event_id)
        if ev:
            return {"ok": True, **ev}
        return {"ok": False, "error": "not found"}

    from EventKit import EKEntityTypeEvent
    from Foundation import NSDate

    store = _get_store()
    for ev in store.calendarsForEntityType_(EKEntityTypeEvent) or []:
        for item in ev.eventsForDate_(NSDate.date()) or []:
            if str(item.eventIdentifier()) == event_id:
                start = item.startDate()
                return {
                    "ok": True,
                    "event_id": event_id,
                    "title": str(item.title()),
                    "start_iso": datetime.fromtimestamp(
                        start.timeIntervalSince1970()
                    ).isoformat(),
                }
    return {"ok": False, "error": "event not found"}


def delete_event(event_id: str) -> dict[str, Any]:
    if _use_mock():
        _mock_events.pop(event_id, None)
        return {"ok": True}
    return {"ok": False, "error": "delete not implemented for live EventKit in v0.1"}
