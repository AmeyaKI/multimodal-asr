"""MCP client adapter for LangGraph (stdio subprocess)."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


class MCPClient:
    """Minimal stdio MCP client for tool calls."""

    def __init__(self, command: list[str] | None = None) -> None:
        self.command = command or [
            sys.executable,
            "-m",
            "jarvis.mcp.server",
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # In-process fallback for same process
        from jarvis.cognition.tool_registry import execute_tool

        return execute_tool(name, arguments)
