"""Duplex TTS with barge-in support."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from jarvis.core.events import Event, EventBus, EventType
from jarvis.voice.tts import speak

_ACKS = ("On it.", "Done.", "Sure.", "Got it.", "Couldn't do that.")


class DuplexSpeaker:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._proc: subprocess.Popen | None = None
        self._lock = asyncio.Lock()

    async def speak(self, text: str, allow_barge_in: bool = True) -> dict[str, Any]:
        async with self._lock:
            await self.bus.publish(Event(EventType.SPEAK, {"text": text}))
            if text in _ACKS or len(text) < 40:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: speak(text))
            return await asyncio.get_event_loop().run_in_executor(None, lambda: speak(text))

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        subprocess.run(["killall", "say"], capture_output=True)

    async def speak_ack(self, phrase: str = "On it.") -> None:
        await self.speak(phrase)
