"""Wake: hold Option hotkey to listen."""

from __future__ import annotations

import asyncio
from typing import Callable

from jarvis.config import get_settings

# macOS / pynput report Option as alt, alt_l, or alt_r
_ALT_KEY_NAMES = frozenset({"alt", "alt_l", "alt_r", "option"})


def _key_matches(key, target, key_name: str) -> bool:
    if key == target:
        return True
    name = getattr(key, "name", None)
    if name in _ALT_KEY_NAMES and key_name in ("option", "alt"):
        return True
    if name == key_name:
        return True
    return False


class WakeListener:
    """Blocks until hotkey pressed; recording stops when hotkey is released."""

    def __init__(
        self,
        on_press: Callable[[], None] | None = None,
        on_release: Callable[[], None] | None = None,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
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
            if not _key_matches(key, target, key_name):
                return
            if not self._active:
                self._active = True
                if self._on_press:
                    self._on_press()

        def on_release(key):
            if not _key_matches(key, target, key_name):
                return
            if self._active:
                self._active = False
                if self._on_release:
                    self._on_release()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._active = False

    async def wait_for_press(self) -> None:
        """Wait until hotkey is pressed."""
        while not self._active:
            await asyncio.sleep(0.02)

    @property
    def is_held(self) -> bool:
        return self._active

    # Back-compat alias
    is_active = is_held

    async def wait_for_activation(self) -> None:
        await self.wait_for_press()
