"""Full assistant runtime: HUD + perception + orchestrator."""

from __future__ import annotations

import asyncio
import threading

from jarvis.config import get_settings
from jarvis.core.events import Event, EventType, get_bus
from jarvis.core.orchestrator import run_orchestrator
from jarvis.memory.store import MemoryStore
from jarvis.perception.pipeline import PerceptionPipeline
from jarvis.voice.duplex import DuplexSpeaker


class AssistantRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bus = get_bus()
        self.store = MemoryStore(self.settings.db_path)
        self.session_id = self.store.new_session()
        self.pipeline = PerceptionPipeline(self.bus)
        self.speaker = DuplexSpeaker(self.bus)
        self._hud_thread: threading.Thread | None = None

    async def _on_transcript(self, text: str) -> None:
        self.speaker.stop()  # barge-in
        await self.speaker.speak_ack("On it.")
        result = await run_orchestrator(text, self.session_id, self.bus)
        reply = result.get("response_text") or result.get("error") or "Done."
        await self.speaker.speak(reply)

    async def start(self) -> None:
        if self.settings.hud_enabled:
            self._start_hud()

        async def listen():
            await self.pipeline.run_loop(self._on_transcript)

        # Subscribe TTS to speak events
        async def on_speak(ev: Event):
            if ev.type == EventType.SPEAK:
                pass  # handled by duplex directly

        self.bus.subscribe(EventType.SPEAK, on_speak)

        print("Jarvis running. Hold Option to speak. Ctrl+C to quit.")
        try:
            await listen()
        except KeyboardInterrupt:
            self.pipeline.stop()

    def _start_hud(self) -> None:
        from jarvis.ui.menubar import run_hud_thread

        self._hud_thread = threading.Thread(target=run_hud_thread, daemon=True)
        self._hud_thread.start()
