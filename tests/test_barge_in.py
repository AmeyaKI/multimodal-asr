"""Barge-in: DuplexSpeaker stops TTS on request."""

from jarvis.core.events import EventBus
from jarvis.voice.duplex import DuplexSpeaker


def test_duplex_stop():
    bus = EventBus()
    speaker = DuplexSpeaker(bus)
    speaker.stop()  # should not raise
