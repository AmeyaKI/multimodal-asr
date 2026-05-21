"""Tool unit tests (mock mode)."""

import os

os.environ["JARVIS_EVAL_MOCK"] = "1"

from jarvis.tools import calendar_eventkit, docs_automation, mail_automation


def test_calendar_create():
    r = calendar_eventkit.create_event(
        "Test Event",
        "2026-05-22T14:00:00",
        "2026-05-22T15:00:00",
    )
    assert r.get("event_id")
    g = calendar_eventkit.get_event(r["event_id"])
    assert g.get("ok") or g.get("title")


def test_mail_draft():
    mail_automation.mail_compose()
    mail_automation.mail_set_to("test@example.com")
    mail_automation.mail_set_subject("Hello")
    mail_automation.mail_set_body("Hi there")
    state = mail_automation.mail_get_draft_state()
    assert state.get("subject") == "Hello"


def test_mail_send_blocked():
    r = mail_automation.mail_send("wrong", expected_token="right")
    assert not r.get("ok") or r.get("error")


def test_doc_create():
    r = docs_automation.doc_create_markdown("Test Doc")
    assert r.get("ok")
    path = r["path"]
    docs_automation.doc_append_text(path, "Hello world")
    read = docs_automation.doc_read(path)
    assert "Hello world" in read.get("content", "")
