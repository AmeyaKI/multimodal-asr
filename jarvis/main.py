"""CLI entrypoint for Jarvis assistant."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from jarvis.config import get_settings


def cli() -> None:
    parser = argparse.ArgumentParser(prog="jarvis", description="Jarvis macOS voice assistant")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("health", help="Check Ollama and permissions")
    sub.add_parser("run", help="Start voice assistant (HUD + pipeline)")
    sub.add_parser("text", help="Process a single text command")

    p_text = sub.add_parser("text-cmd", help="Run one text command")
    p_text.add_argument("utterance", nargs="+", help="Command text")

    args = parser.parse_args()
    if args.command == "health":
        sys.exit(run_health())
    if args.command == "run":
        asyncio.run(run_assistant())
    elif args.command == "text" or args.command == "text-cmd":
        utterance = " ".join(getattr(args, "utterance", []) or [])
        if not utterance and args.command == "text":
            utterance = input("Command: ").strip()
        asyncio.run(run_text_command(utterance))
    mcp_parser = sub.add_parser("mcp", help="Start MCP server (stdio)")
    mcp_parser.set_defaults(command="mcp")

    if args.command == "mcp":
        from jarvis.mcp.server import serve

        serve()
    else:
        parser.print_help()
        sys.exit(0)


def run_health() -> int:
    """Health check: Ollama + data dir."""
    settings = get_settings()
    ok = True
    print("Jarvis health check")
    print("-" * 40)

    # Data dir
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Data dir: {settings.data_dir}")

    # Ollama
    try:
        import httpx

        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            print(f"[OK] Ollama reachable at {settings.ollama_base_url}")
            if models:
                print(f"     Models: {', '.join(models[:5])}")
            else:
                print("[WARN] No models pulled. Run: ollama pull llama3.2:3b")
        else:
            print(f"[FAIL] Ollama returned {r.status_code}")
            ok = False
    except Exception as e:
        print(f"[FAIL] Ollama not reachable: {e}")
        print("       Install: https://ollama.com — then: ollama pull llama3.2:3b")
        ok = False

    # Permissions reminder
    print("[INFO] macOS permissions required:")
    print("       - Microphone (voice input)")
    print("       - Automation: Mail, Calendar, Notes, System Events")
    print("       - Accessibility (UI automation fallback)")
    print("       Run: scripts/setup_macos_permissions.sh")

    # SQLite
    from jarvis.memory.store import MemoryStore

    store = MemoryStore(settings.db_path)
    sid = store.new_session()
    print(f"[OK] SQLite memory at {settings.db_path} (session {sid[:8]}...)")

    print("-" * 40)
    if ok:
        print("HEALTH OK")
        return 0
    print("HEALTH FAILED (Ollama optional if not using voice/LLM yet)")
    return 1 if os.getenv("JARVIS_STRICT_HEALTH") else 0


async def run_text_command(utterance: str) -> None:
    from jarvis.core.events import get_bus
    from jarvis.core.orchestrator import run_orchestrator

    settings = get_settings()
    from jarvis.memory.store import MemoryStore

    store = MemoryStore(settings.db_path)
    session_id = store.new_session()
    bus = get_bus()
    result = await run_orchestrator(utterance, session_id=session_id, bus=bus)
    print(result.get("response_text", result.get("error", result)))


async def run_assistant() -> None:
    from jarvis.runtime import AssistantRuntime

    runtime = AssistantRuntime()
    await runtime.start()


if __name__ == "__main__":
    cli()
