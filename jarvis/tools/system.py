"""System tools: open apps, shortcuts, frontmost app."""

from __future__ import annotations

from typing import Any

from jarvis.tools.applescript import run_applescript


def open_app(name: str) -> dict[str, Any]:
    script = f'tell application "{name}" to activate'
    return run_applescript(script)


def get_frontmost_app() -> dict[str, Any]:
    script = '''
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    return frontApp
    '''
    r = run_applescript(script)
    if r["ok"]:
        return {"ok": True, "app": r["stdout"]}
    return {"ok": False, "error": r["stderr"]}


def run_shortcut(name: str, input_text: str = "") -> dict[str, Any]:
    """Run a macOS Shortcut by name."""
    escaped = input_text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Shortcuts Events"
        run the shortcut named "{name}" with input "{escaped}"
    end tell
    '''
    return run_applescript(script, timeout=60.0)
