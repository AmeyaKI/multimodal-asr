"""Wake hotkey matching."""

from jarvis.perception.wake import _key_matches


class _FakeKey:
    def __init__(self, name: str | None = None):
        self.name = name


def test_option_matches_alt_names():
    from pynput import keyboard

    target = keyboard.Key.alt
    assert _key_matches(keyboard.Key.alt_l, target, "option")
    assert _key_matches(_FakeKey("alt_r"), target, "option")
    assert not _key_matches(keyboard.Key.space, target, "option")
