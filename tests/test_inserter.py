"""paste_text with a fake clipboard and keyboard (nothing is really typed), plus one round trip
through the real Windows clipboard (it is saved and put back)."""
import contextlib

import pytest

from whisperclaude import inserter

TEXT = 13  # CF_UNICODETEXT


def as_text(saved):
    data = dict(saved).get(TEXT)
    return None if data is None else data.decode("utf-16-le").rstrip("\0")


@pytest.fixture
def fake(monkeypatch):
    state = {"clipboard": [(TEXT, "old clipboard äöü\0".encode("utf-16-le")), (8, b"image")],
             "at_ctrl_v": None, "keys": []}

    class FakeKeyboard:
        def pressed(self, key):
            state["keys"].append(("hold", key))
            return contextlib.nullcontext()

        def tap(self, key):
            state["keys"].append(("tap", key))
            state["at_ctrl_v"] = as_text(state["clipboard"])

    monkeypatch.setattr(inserter, "save_clipboard", lambda: list(state["clipboard"]))
    monkeypatch.setattr(inserter, "restore_clipboard",
                        lambda saved: state.update(clipboard=list(saved)))
    monkeypatch.setattr(inserter, "set_clipboard_text", lambda t: state.update(
        clipboard=[(TEXT, (t + "\0").encode("utf-16-le"))]))
    monkeypatch.setattr(inserter, "clipboard_text", lambda: as_text(state["clipboard"]))
    monkeypatch.setattr(inserter, "Controller", FakeKeyboard)
    monkeypatch.setattr(inserter, "_modifiers_down", lambda: False)
    monkeypatch.setattr(inserter.time, "sleep", lambda s: None)
    return state


def test_pastes_text_and_restores_every_format(fake):
    before = list(fake["clipboard"])
    inserter.paste_text("Grüße, neuer Text")
    assert fake["at_ctrl_v"] == "Grüße, neuer Text"
    assert fake["keys"] == [("hold", inserter.Key.ctrl), ("tap", "v")]
    assert fake["clipboard"] == before  # text and image


def test_empty_text_does_nothing(fake):
    inserter.paste_text("")
    assert fake["keys"] == []


def test_empty_clipboard_is_empty_again_afterwards(fake):
    """The bug: an empty (or image-only) clipboard used to keep the dictation."""
    fake["clipboard"] = []
    inserter.paste_text("neu")
    assert fake["at_ctrl_v"] == "neu" and fake["clipboard"] == []


def test_line_breaks_are_pasted_as_windows_crlf(fake):
    inserter.paste_text("one\n\n- two\r\nthree")
    assert fake["at_ctrl_v"] == "one\r\n\r\n- two\r\nthree"
    assert inserter.to_clipboard_text("plain") == "plain"


def test_something_copied_during_the_paste_is_not_overwritten(fake, monkeypatch):
    monkeypatch.setattr(inserter.time, "sleep", lambda s: s == 0.3 and fake.update(
        clipboard=[(TEXT, "just copied\0".encode("utf-16-le"))]))
    inserter.paste_text("neu")
    assert as_text(fake["clipboard"]) == "just copied"


def test_failed_clipboard_restore_is_not_a_paste_error(fake, monkeypatch):
    def locked(saved):
        raise inserter.ClipboardError("clipboard locked")

    monkeypatch.setattr(inserter, "restore_clipboard", locked)
    inserter.paste_text("neu")  # must not raise: the text was pasted
    assert fake["at_ctrl_v"] == "neu"


def test_real_clipboard_round_trip():
    users = inserter.save_clipboard()
    custom = inserter._RegisterClipboardFormatW("WhisperClaude test format")
    try:
        for content in ([], [(TEXT, "äöü\0".encode("utf-16-le")), (custom, b"\x01\x02")]):
            inserter._write(content)
            saved = inserter.save_clipboard()
            inserter.set_clipboard_text("Diktat")
            assert inserter.clipboard_text() == "Diktat"
            inserter.restore_clipboard(saved)
            assert inserter.clipboard_text() == as_text(content)
            assert {f for f, _ in content} <= {f for f, _ in inserter.save_clipboard()}
    finally:
        inserter.restore_clipboard(users)
