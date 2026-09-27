"""paste_text with a fake clipboard and keyboard: nothing is really typed."""
import contextlib

import pytest

from whisperclaude import inserter


@pytest.fixture
def fake(monkeypatch):
    state = {"clipboard": "old clipboard äöü", "at_ctrl_v": None, "keys": []}

    class FakeKeyboard:
        def pressed(self, key):
            state["keys"].append(("hold", key))
            return contextlib.nullcontext()

        def tap(self, key):
            state["keys"].append(("tap", key))
            state["at_ctrl_v"] = state["clipboard"]

    monkeypatch.setattr(inserter.pyperclip, "paste", lambda: state["clipboard"])
    monkeypatch.setattr(inserter.pyperclip, "copy", lambda t: state.update(clipboard=t))
    monkeypatch.setattr(inserter, "Controller", FakeKeyboard)
    monkeypatch.setattr(inserter, "_modifiers_down", lambda: False)
    monkeypatch.setattr(inserter.time, "sleep", lambda s: None)
    return state


def test_pastes_text_and_restores_clipboard(fake):
    inserter.paste_text("Grüße, neuer Text")
    assert fake["at_ctrl_v"] == "Grüße, neuer Text"
    assert fake["keys"] == [("hold", inserter.Key.ctrl), ("tap", "v")]
    assert fake["clipboard"] == "old clipboard äöü"


def test_empty_text_does_nothing(fake):
    inserter.paste_text("")
    assert fake["keys"] == []
    assert fake["clipboard"] == "old clipboard äöü"


def test_empty_old_clipboard_is_not_restored(fake):
    """Non-text clipboard (e.g. an image) reads as ''; we leave our text instead of ''."""
    fake["clipboard"] = ""
    inserter.paste_text("neu")
    assert fake["clipboard"] == "neu"
