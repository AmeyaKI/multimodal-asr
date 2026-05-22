"""CLI argument parsing."""

import os
import subprocess
import sys

os.environ["JARVIS_EVAL_MOCK"] = "1"


def test_text_command_parses_quoted_utterance():
    root = os.path.dirname(os.path.dirname(__file__))
    venv_py = os.path.join(root, ".venv", "bin", "python")
    py = venv_py if os.path.isfile(venv_py) else sys.executable
    proc = subprocess.run(
        [
            py,
            "-m",
            "jarvis.main",
            "text",
            "Create event Test on 2026-05-25 10:00 for 30 minutes",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "JARVIS_EVAL_MOCK": "1"},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Done" in proc.stdout or "done" in proc.stdout.lower()
