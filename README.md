# Jarvis — macOS Voice Assistant

Local-first voice assistant for macOS: hold **Option** to speak, execute calendar/mail/document tasks with visible automation and a menu bar HUD.

## Architecture

- **Perception:** Silero VAD + faster-whisper ASR
- **Cognition:** Ollama (local) router/planner + LangGraph-style orchestrator
- **Action:** EventKit calendar, Mail automation, Markdown docs
- **UI:** Menu bar HUD (transcript, steps, Confirm/Cancel)
- **MCP:** `jarvis mcp` exposes tools over stdio

## Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
# Or: pip install -r requirements.txt

# Install Ollama and pull model
brew install ollama
ollama pull llama3.2:3b

# Permissions
chmod +x scripts/setup_macos_permissions.sh
./scripts/setup_macos_permissions.sh

# Health check
python -m jarvis.main health
# Data stored in ./.jarvis-data (override: JARVIS_DATA_DIR)
```

## Usage

```bash
# Voice assistant (HUD + hold Option to talk)
python -m jarvis.main run

# Single text command
python -m jarvis.main text "Create event Team Sync on 2026-05-22 10:00 for 30 minutes"

# MCP server
python -m jarvis.main mcp

# Eval (mock mode)
JARVIS_EVAL_MOCK=1 python -m jarvis.eval.run_eval
```

## Environment


| Variable               | Default                  | Description                               |
| ---------------------- | ------------------------ | ----------------------------------------- |
| `OLLAMA_BASE_URL`      | `http://127.0.0.1:11434` | Ollama API                                |
| `OLLAMA_MODEL`         | `llama3.2:3b`            | Router model                              |
| `JARVIS_EVAL_MOCK`     | `0`                      | Mock tools for tests                      |
| `JARVIS_LLM_FALLBACK`  | `none`                   | Set `gemini` + `GEMINI_API_KEY` for cloud |
| `REQUIRE_CONFIRM_SEND` | `true`                   | Block email send without HUD confirm      |
| `SHOW_ACTIONS_VISUALLY` | `true`                  | Open Calendar/Mail/TextEdit in foreground |
| `VISIBLE_TYPING_DELAY_MS` | `35`                  | Keystroke delay for visible typing        |
| `STEP_PAUSE_SEC` | `0.5`                          | Pause between automation steps            |


## Virtual environment

All dependencies install into `.venv/` (gitignored). Locked versions: `requirements-lock.txt`.

```bash
source .venv/bin/activate
python -m jarvis.main run      # voice + HUD
python -m jarvis.main text "Create event Team Standup on 2026-05-22 10:00 for 30 minutes"
# Quote the full phrase so the shell passes it as one argument
JARVIS_EVAL_MOCK=1 python -m jarvis.eval.run_eval
```

## Legacy code

The previous `src/` prototype has been removed; implementation lives in `jarvis/`.