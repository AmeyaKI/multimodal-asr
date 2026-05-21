"""Calendar tools via PyObjC EventKit."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from jarvis.config import get_settings

_store = None
_mock_events: dict[str, dict[str, Any]] = {}


def _use_mock() -> bool:
    return get_settings().jarvis_eval_mock


def _get_store():
    global _store
    if _use_mock():
        return None
    if _store is not None:
        return _store
    from EventKit import EKEntityTypeEvent, EKEventStore

    store = EKEventStore.alloc().init()
    granted = [False]

    def handler(granted_flag, error):
        granted[0] = granted_flag

    store.requestAccessToEntityType_completion_(EKEntityTypeEvent, handler)
    if not granted[0]:
        raise PermissionError("Calendar access denied. Grant in System Settings.")
    _store = store
    return store


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

    store = _get_store()
    event = EKEvent.eventWithEventStore_(store)
    event.setTitle_(title)
    event.setStartDate_(parse_iso(start_iso))
    event.setEndDate_(parse_iso(end_iso))
    if notes:
        event.setNotes_(notes)
    if location:
        event.setLocation_(location)

    if calendar_id:
        for cal in store.calendarsForEntityType_(EKEntityTypeEvent) or []:
            if str(cal.calendarIdentifier()) == calendar_id:
                event.setCalendar_(cal)
                break
    else:
        default = store.defaultCalendarForNewEvents()
        if default:
            event.setCalendar_(default)

    err = [None]
    ok = store.saveEvent_span_error_(event, 0, err)  # EKSpanThisEvent = 0
    if not ok:
        return {"ok": False, "error": str(err[0]) if err[0] else "save failed"}

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
                    "start_iso": datetime.fromtimestamp(start.timeIntervalSince1970()).isoformat(),
                }
    return {"ok": False, "error": "event not found"}


def delete_event(event_id: str) -> dict[str, Any]:
    if _use_mock():
        _mock_events.pop(event_id, None)
        return {"ok": True}
    # Simplified: would need fetch by id
    return {"ok": False, "error": "delete not implemented for live EventKit in v0.1"}
