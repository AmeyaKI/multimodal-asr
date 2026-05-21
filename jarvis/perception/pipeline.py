"""Audio → VAD → ASR perception pipeline."""

from __future__ import annotations

import asyncio

from jarvis.config import get_settings
from jarvis.core.events import Event, EventBus, EventType, get_bus
from jarvis.perception.asr_offline import transcribe_parakeet
from jarvis.perception.asr_stream import transcribe
from jarvis.perception.vad import VADStream
from jarvis.perception.wake import WakeListener


class PerceptionPipeline:
    def __init__(self, bus: EventBus | None = None) -> None:
        self.bus = bus or get_bus()
        self.vad = VADStream()
        self.wake = WakeListener(on_activate=self._on_wake)
        self._running = False
        self._barge_in = asyncio.Event()

    def _on_wake(self) -> None:
        self._barge_in.set()

    def request_barge_in(self) -> None:
        self._barge_in.set()

    async def run_loop(self, on_final_transcript) -> None:
        """Hold hotkey → capture → transcribe → callback."""
        self.wake.start()
        self._running = True
        settings = get_settings()

        while self._running:
            await self.bus.publish(
                Event(EventType.STATE_CHANGED, {"state": "idle"})
            )
            await self.wake.wait_for_activation()
            await self.bus.publish(
                Event(EventType.LISTENING, {"state": "listening"})
            )
            await self.bus.publish(
                Event(EventType.STATE_CHANGED, {"state": "listening"})
            )

            self.vad.set_speech_callback(lambda: self.request_barge_in())
            audio = await self.vad.capture_utterance()
            if audio is None:
                continue

            await self.bus.publish(
                Event(EventType.TRANSCRIPT_PARTIAL, {"text": "..."})
            )
            if settings.asr_backend == "parakeet":
                text = transcribe_parakeet(audio)
                result = {"text": text}
            else:
                result = transcribe(audio)

            text = result.get("text", "").strip()
            if not text:
                continue

            await self.bus.publish(
                Event(EventType.TRANSCRIPT_FINAL, {"text": text})
            )
            await on_final_transcript(text)

    def stop(self) -> None:
        self._running = False
        self.wake.stop()
