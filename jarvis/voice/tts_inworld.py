"""Optional Inworld TTS (from legacy src/tts.py)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import requests
import sounddevice as sd
import soundfile as sf


def speak_inworld(text: str, voice: str = "Ashley") -> dict[str, Any]:
    key = os.getenv("INWORLD_API_KEY")
    if not key:
        return {"ok": False, "error": "INWORLD_API_KEY not set"}
    url = "https://api.inworld.ai/tts/v1/voice"
    headers = {"Authorization": f"Basic {key}", "Content-Type": "application/json"}
    payload = {"text": text, "voiceId": voice, "modelId": "inworld-tts-1"}
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    audio_bytes = base64.b64decode(response.json()["audioContent"])
    audio_buffer = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")
    sd.play(audio_data, sample_rate)
    sd.wait()
    return {"ok": True}
