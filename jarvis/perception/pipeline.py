"""Audio → hold-to-talk → ASR perception pipeline."""

from __future__ import annotations

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
        self.wake = WakeListener()
        self._running = False

    async def run_loop(self, on_final_transcript) -> None:
        """Hold hotkey → capture until release → transcribe → callback."""
        settings = get_settings()
        hotkey = settings.wake_hotkey.capitalize()
        self.wake.start()
        self._running = True

        while self._running:
            await self.bus.publish(
                Event(EventType.STATE_CHANGED, {"state": "idle"})
            )
            await self.wake.wait_for_press()

            await self.bus.publish(Event(EventType.LISTENING, {}))
            await self.bus.publish(
                Event(EventType.STATE_CHANGED, {"state": "listening"})
            )
            print(f"\nListening… (release {hotkey} when done)", flush=True)

            audio = await self.vad.capture_while_held(lambda: self.wake.is_held)

            await self.bus.publish(
                Event(EventType.STATE_CHANGED, {"state": "idle"})
            )

            if audio is None:
                continue

            print("Transcribing…", flush=True)
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
                print("(no speech detected)\n", flush=True)
                continue

            print(f"You: {text}", flush=True)
            await self.bus.publish(
                Event(EventType.TRANSCRIPT_FINAL, {"text": text})
            )
            await on_final_transcript(text)

    def stop(self) -> None:
        self._running = False
        self.wake.stop()
