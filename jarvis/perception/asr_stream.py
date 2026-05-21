"""Streaming ASR via faster-whisper."""

from __future__ import annotations

from typing import Any

import numpy as np

from jarvis.config import get_settings

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel

    settings = get_settings()
    device = "cpu"
    compute = "int8"
    try:
        import torch

        if torch.backends.mps.is_available():
            device = "cpu"  # faster-whisper MPS limited; use cpu
    except Exception:
        pass
    _model = WhisperModel(settings.asr_model, device=device, compute_type=compute)
    return _model


def transcribe(audio: np.ndarray) -> dict[str, Any]:
    """Full utterance transcription."""
    if audio is None or len(audio) < 1600:
        return {"text": "", "partial": False}
    model = _get_model()
    segments, _ = model.transcribe(audio, language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "partial": False}


def transcribe_partial(audio: np.ndarray) -> dict[str, Any]:
    """Partial transcript for live HUD (same as final for batch whisper)."""
    return transcribe(audio)
