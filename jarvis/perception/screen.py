"""Screen capture and Accessibility snapshot."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def capture_screen(path: str | None = None) -> dict[str, Any]:
    out = path or "/tmp/jarvis_screen.png"
    r = subprocess.run(["screencapture", "-x", out], capture_output=True, text=True)
    return {"ok": r.returncode == 0, "path": out}


def get_ax_tree_frontmost() -> str:
    """Lightweight AX snapshot via AppleScript."""
    script = '''
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        try
            set winName to name of window 1 of process frontApp
        on error
            set winName to ""
        end try
    end tell
    return frontApp & " | " & winName
    '''
    from jarvis.tools.applescript import run_applescript

    r = run_applescript(script)
    if r["ok"]:
        return r["stdout"]
    return ""
