"""Async event bus for HUD, TTS, and orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


class EventType(str, Enum):
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    STEP_STARTED = "step_started"
    STEP_DONE = "step_done"
    NEED_CONFIRM = "need_confirm"
    CONFIRM_RECEIVED = "confirm_received"
    CANCEL_RECEIVED = "cancel_received"
    STATE_CHANGED = "state_changed"
    SPEAK = "speak"
    LISTENING = "listening"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


Listener = Callable[[Event], Coroutine[Any, Any, None] | None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[EventType, list[Listener]] = {}
        self._confirm_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.ui_state: str = "idle"
        self.last_transcript: str = ""
        self.current_step: str = ""

    def subscribe(self, event_type: EventType, listener: Listener) -> None:
        self._listeners.setdefault(event_type, []).append(listener)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _enqueue_confirm(self, value: str | None) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._confirm_queue.put_nowait, value)
        else:
            self._confirm_queue.put_nowait(value)

    async def publish(self, event: Event) -> None:
        if event.type == EventType.TRANSCRIPT_FINAL:
            self.last_transcript = event.payload.get("text", "")
        if event.type == EventType.STEP_STARTED:
            self.current_step = event.payload.get("label", "")
        if event.type == EventType.STATE_CHANGED:
            self.ui_state = event.payload.get("state", self.ui_state)

        for listener in self._listeners.get(event.type, []):
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

    async def wait_confirm(self, token: str, timeout: float = 120.0) -> bool:
        """Block until HUD confirm or cancel."""
        try:
            received = await asyncio.wait_for(self._confirm_queue.get(), timeout=timeout)
            return received == token
        except asyncio.TimeoutError:
            return False

    def submit_confirm(self, token: str) -> None:
        self._enqueue_confirm(token)

    def submit_cancel(self) -> None:
        self._enqueue_confirm(None)


# Global bus for single-process app
_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
