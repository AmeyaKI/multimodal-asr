"""Document creation and writing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from jarvis.config import get_settings
from jarvis.tools.applescript import run_applescript


def doc_create_markdown(title: str, subdir: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    base = settings.docs_dir
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
    path = base / f"{safe}.md"
    path.write_text(f"# {title}\n\n", encoding="utf-8")
    run_applescript(f'tell application "TextEdit" to open POSIX file "{path}"')
    return {"ok": True, "path": str(path), "title": title}


def doc_append_text(path: str, text: str, visible: bool = True) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    content = p.read_text(encoding="utf-8")
    p.write_text(content + text + "\n", encoding="utf-8")
    if visible:
        run_applescript(f'''
        tell application "TextEdit"
            activate
            open POSIX file "{p}"
        end tell
        ''')
    return {"ok": True, "path": str(p), "appended_len": len(text)}


def doc_read(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"ok": False, "error": "not found"}
    return {"ok": True, "path": str(p), "content": p.read_text(encoding="utf-8")[:4000]}


def doc_revert_backup(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    bak = p.with_suffix(p.suffix + ".bak")
    if bak.exists():
        shutil.copy(bak, p)
        return {"ok": True}
    return {"ok": False, "error": "no backup"}


def doc_backup(path: str) -> None:
    p = Path(path).expanduser()
    if p.exists():
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
