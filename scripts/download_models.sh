#!/usr/bin/env bash
# Pull Ollama models and optional Whisper assets.
set -euo pipefail

echo "Pulling Ollama models..."
ollama pull llama3.2:3b || echo "Warning: ollama pull failed — install Ollama first"

echo "Whisper models download on first run (faster-whisper base.en)."
echo "Done."
