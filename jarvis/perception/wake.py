"""Wake: hold Option hotkey to listen."""

from __future__ import annotations

import asyncio
from typing import Callable

from jarvis.config import get_settings


class WakeListener:
    """Blocks until hotkey pressed (hold to talk)."""

    def __init__(self, on_activate: Callable[[], None] | None = None) -> None:
        self._on_activate = on_activate
        self._active = False
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        settings = get_settings()
        key_name = settings.wake_hotkey.lower()

        key_map = {
            "option": keyboard.Key.alt,
            "alt": keyboard.Key.alt,
            "ctrl": keyboard.Key.ctrl,
            "control": keyboard.Key.ctrl,
        }
        target = key_map.get(key_name, keyboard.Key.alt)

        def on_press(key):
            if key == target or getattr(key, "name", None) == key_name:
                if not self._active:
                    self._active = True
                    if self._on_activate:
                        self._on_activate()

        def on_release(key):
            if key == target or getattr(key, "name", None) == key_name:
                self._active = False

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    async def wait_for_activation(self) -> None:
        """Wait until hotkey pressed."""
        while not self._active:
            await asyncio.sleep(0.05)

    @property
    def is_active(self) -> bool:
        return self._active
