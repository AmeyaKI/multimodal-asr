"""Menu bar HUD — transcript, step, Confirm/Cancel."""

from __future__ import annotations

import asyncio

import rumps

from jarvis.core.events import Event, EventType, get_bus


class JarvisHUD(rumps.App):
    def __init__(self) -> None:
        super().__init__("Jarvis", icon=None, template=None, quit_button=None)
        self.bus = get_bus()
        self.status_item = rumps.MenuItem("Idle")
        self.transcript_item = rumps.MenuItem("...")
        self.step_item = rumps.MenuItem("")
        self.menu = [
            self.status_item,
            self.transcript_item,
            self.step_item,
            None,
            rumps.MenuItem("Confirm", callback=self.on_confirm),
            rumps.MenuItem("Cancel", callback=self.on_cancel),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._pending_token: str | None = None

    def on_confirm(self, _) -> None:
        if self._pending_token:
            self.bus.submit_confirm(self._pending_token)

    def on_cancel(self, _) -> None:
        self.bus.submit_cancel()
        self._pending_token = None
        self.status_item.title = "Cancelled"

    def quit_app(self, _) -> None:
        rumps.quit_application()

    def update_from_event(self, event: Event) -> None:
        if event.type == EventType.STATE_CHANGED:
            self.status_item.title = event.payload.get("state", "idle").capitalize()
        elif event.type == EventType.TRANSCRIPT_FINAL:
            t = event.payload.get("text", "")
            self.transcript_item.title = (t[:60] + "...") if len(t) > 60 else t
        elif event.type == EventType.STEP_STARTED:
            self.step_item.title = event.payload.get("label", "")
        elif event.type == EventType.NEED_CONFIRM:
            self._pending_token = event.payload.get("token")
            self.status_item.title = "Confirm send?"
            self.step_item.title = event.payload.get("label", "Confirm")


_hud_app: JarvisHUD | None = None


def _sync_listener(event: Event) -> None:
    global _hud_app
    if _hud_app:
        _hud_app.update_from_event(event)


def run_hud_thread() -> None:
    global _hud_app
    bus = get_bus()

    def listener(event: Event):
        if _hud_app:
            _hud_app.update_from_event(event)

    for et in EventType:
        bus.subscribe(et, listener)

    _hud_app = JarvisHUD()
    _hud_app.run()
