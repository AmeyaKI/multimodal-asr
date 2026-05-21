"""Silero VAD — continuous utterance capture."""

from __future__ import annotations

from collections.abc import AsyncIterator
from queue import Queue
from typing import Callable

import numpy as np
import sounddevice as sd
import torch

from jarvis.config import get_settings
from jarvis.perception.audio_io import FRAME_SIZE, SAMPLE_RATE


class VADStream:
    def __init__(self, vad_model=None) -> None:
        settings = get_settings()
        if vad_model is None:
            from silero_vad import load_silero_vad

            vad_model = load_silero_vad()
        self.vad_model = vad_model
        self.threshold = settings.vad_threshold
        self.max_silence = settings.max_silence_sec
        self.frame_duration = FRAME_SIZE / SAMPLE_RATE
        self._queue: Queue = Queue()
        self._recorded: list[np.ndarray] = []
        self._on_speech: Callable[[], None] | None = None

    def set_speech_callback(self, cb: Callable[[], None]) -> None:
        self._on_speech = cb

    def _callback(self, indata, frames, time, status) -> None:
        frame = indata.copy().squeeze()
        if len(frame) >= 160:
            self._queue.put(frame)

    async def capture_utterance(self) -> np.ndarray | None:
        """Block until silence ends utterance; return float32 audio."""
        import asyncio

        max_silent = int(self.max_silence / self.frame_duration)
        silence_count = 0
        self._recorded = []
        loop = asyncio.get_event_loop()

        def run_capture():
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=FRAME_SIZE,
                dtype="float32",
                callback=self._callback,
            ):
                while True:
                    frame = self._queue.get()
                    tensor = torch.from_numpy(frame).float()
                    with torch.no_grad():
                        prob = self.vad_model(tensor, SAMPLE_RATE).item()
                    if prob > self.threshold:
                        self._recorded.append(frame)
                        silence_count = 0
                        if self._on_speech and len(self._recorded) == 1:
                            self._on_speech()
                    else:
                        if self._recorded:
                            silence_count += 1
                        if silence_count >= max_silent and self._recorded:
                            break

        await loop.run_in_executor(None, run_capture)
        if not self._recorded:
            return None
        return np.concatenate(self._recorded).astype(np.float32)

    async def stream_frames(self) -> AsyncIterator[np.ndarray]:
        """Yield frames while stream is open (for barge-in)."""
        import asyncio

        loop = asyncio.get_event_loop()
        while True:
            try:
                frame = await loop.run_in_executor(None, lambda: self._queue.get(timeout=0.1))
                yield frame
            except Exception:
                await asyncio.sleep(0.05)
