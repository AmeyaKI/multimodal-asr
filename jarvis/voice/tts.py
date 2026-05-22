"""TTS: Inworld default, optional macOS say()."""

from __future__ import annotations

import subprocess
from typing import Any

from jarvis.config import get_settings


def speak(text: str, block: bool = True) -> dict[str, Any]:
    settings = get_settings()
    if not text.strip():
        return {"ok": True}
    if settings.tts_backend == "inworld":
        try:
            from jarvis.voice.tts_inworld import speak_inworld

            return speak_inworld(text, api_key=settings.inworld_api_key)
        except Exception as e:
            return {"ok": False, "error": str(e)}
    try:
        subprocess.run(["say", text], check=True, timeout=60)
        return {"ok": True, "text": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}
