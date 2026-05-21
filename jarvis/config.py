"""Application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama (local-first)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_planner_model: str = "llama3.2:3b"
    jarvis_llm_fallback: Literal["none", "gemini"] = "none"
    gemini_api_key: str | None = None

    # Voice
    asr_backend: Literal["whisper", "parakeet"] = "whisper"
    asr_model: str = "base.en"
    vad_threshold: float = 0.5
    max_silence_sec: float = 2.0
    wake_hotkey: str = "option"  # hold to listen
    tts_backend: Literal["say", "inworld"] = "say"

    # Safety
    require_confirm_send: bool = True
    hud_enabled: bool = True

    # Paths (project-local data dir; override with JARVIS_DATA_DIR)
    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / ".jarvis-data"
    )
    docs_dir: Path = Field(
        default_factory=lambda: Path.home() / "Documents" / "Jarvis"
    )

    # Eval
    jarvis_eval_mock: bool = False

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "jarvis.db"


def get_settings() -> Settings:
    return Settings()
