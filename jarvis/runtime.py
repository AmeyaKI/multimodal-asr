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

    async def _on_transcript(self, text: str) -> None:
        self.speaker.stop()  # barge-in
        print("Running command…", flush=True)
        await self.speaker.speak_ack("On it.")
        result = await run_orchestrator(text, self.session_id, self.bus)
        reply = result.get("response_text") or result.get("error") or "Done."
        print(f"Jarvis: {reply}\n", flush=True)
        await self.speaker.speak(reply)

    async def start(self) -> None:
        async def listen():
            await self.pipeline.run_loop(self._on_transcript)

        # Subscribe TTS to speak events
        async def on_speak(ev: Event):
            if ev.type == EventType.SPEAK:
                pass  # handled by duplex directly

        self.bus.subscribe(EventType.SPEAK, on_speak)

        hotkey = self.settings.wake_hotkey.capitalize()
        print(f"Jarvis running. Hold {hotkey} to speak, release to stop. Ctrl+C to quit.")
        print(
            "If the hotkey does nothing, grant Accessibility + Input Monitoring "
            "to Terminal (or your IDE) in System Settings → Privacy & Security.",
            flush=True,
        )
        try:
            await listen()
        except KeyboardInterrupt:
            self.pipeline.stop()


def run_with_hud() -> None:
    """Run AppKit HUD on the main thread; asyncio assistant in a background thread."""
    from jarvis.ui.menubar import run_hud

    worker_error: list[BaseException] = []
    loop_ready = threading.Event()

    def asyncio_worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        get_bus().set_loop(loop)
        loop_ready.set()
        try:
            loop.run_until_complete(AssistantRuntime().start())
        except BaseException as exc:
            worker_error.append(exc)
        finally:
            loop.close()

    worker = threading.Thread(target=asyncio_worker, daemon=True, name="jarvis-asyncio")
    worker.start()
    if not loop_ready.wait(timeout=10.0):
        raise RuntimeError("Assistant event loop did not start")
    try:
        run_hud()
    except KeyboardInterrupt:
        pass
    finally:
        worker.join(timeout=2.0)
    if worker_error:
        raise worker_error[0]


def run_assistant() -> None:
    """CLI entry for `jarvis run`: HUD on main thread when enabled, else asyncio only."""
    settings = get_settings()
    if settings.hud_enabled and threading.current_thread() is threading.main_thread():
        try:
            run_with_hud()
        except KeyboardInterrupt:
            pass
        return
    asyncio.run(AssistantRuntime().start())
