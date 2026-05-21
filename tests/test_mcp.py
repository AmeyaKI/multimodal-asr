"""MCP tool registry mirrors server tools."""

from jarvis.cognition.tool_registry import TOOLS


def test_mcp_tools_registered():
    expected = {
        "create_event",
        "mail_compose",
        "mail_set_subject",
        "doc_create_markdown",
        "open_app",
    }
    assert expected.issubset(set(TOOLS.keys()))
