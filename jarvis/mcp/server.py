"""FastMCP server exposing all Jarvis tools."""

from __future__ import annotations

from typing import Any


def serve() -> None:
    from mcp.server.fastmcp import FastMCP

    from jarvis.tools import calendar_eventkit, docs_automation, mail_automation, system

    mcp = FastMCP("jarvis-macos")

    @mcp.tool()
    def list_calendars() -> list[dict[str, str]]:
        """List available calendars."""
        return calendar_eventkit.list_calendars()

    @mcp.tool()
    def create_event(
        title: str,
        start_iso: str,
        end_iso: str,
        calendar_id: str | None = None,
        notes: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Create a calendar event (ISO-8601 dates)."""
        return calendar_eventkit.create_event(
            title, start_iso, end_iso, calendar_id, notes, location
        )

    @mcp.tool()
    def get_event(event_id: str) -> dict[str, Any]:
        """Get calendar event by id."""
        return calendar_eventkit.get_event(event_id)

    @mcp.tool()
    def mail_compose() -> dict[str, Any]:
        """Open a new visible Mail message."""
        return mail_automation.mail_compose()

    @mcp.tool()
    def mail_set_to(addresses: str) -> dict[str, Any]:
        """Set Mail To recipients."""
        return mail_automation.mail_set_to(addresses)

    @mcp.tool()
    def mail_set_subject(subject: str) -> dict[str, Any]:
        """Set Mail subject."""
        return mail_automation.mail_set_subject(subject)

    @mcp.tool()
    def mail_set_body(body: str) -> dict[str, Any]:
        """Set Mail body content."""
        return mail_automation.mail_set_body(body)

    @mcp.tool()
    def mail_get_draft_state() -> dict[str, Any]:
        """Read current Mail draft fields."""
        return mail_automation.mail_get_draft_state()

    @mcp.tool()
    def mail_send(confirm_token: str) -> dict[str, Any]:
        """Send Mail (requires confirm_token from HUD)."""
        return mail_automation.mail_send(confirm_token, expected_token=confirm_token)

    @mcp.tool()
    def doc_create_markdown(title: str) -> dict[str, Any]:
        """Create markdown doc in ~/Documents/Jarvis/."""
        return docs_automation.doc_create_markdown(title)

    @mcp.tool()
    def doc_append_text(path: str, text: str) -> dict[str, Any]:
        """Append text to a document."""
        return docs_automation.doc_append_text(path, text)

    @mcp.tool()
    def doc_read(path: str) -> dict[str, Any]:
        """Read document content."""
        return docs_automation.doc_read(path)

    @mcp.tool()
    def open_app(name: str) -> dict[str, Any]:
        """Activate a macOS application."""
        return system.open_app(name)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    serve()
