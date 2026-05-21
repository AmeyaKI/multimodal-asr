"""Thin AppleScript execution fallback."""

from __future__ import annotations

import subprocess
from typing import Any


def run_applescript(script: str, timeout: float = 30.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return {"ok": True, "stdout": (result.stdout or "").strip(), "stderr": ""}
        return {"ok": False, "stdout": "", "stderr": (result.stderr or "Unknown error").strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)}
