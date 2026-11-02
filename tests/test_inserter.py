"""paste_text with a fake clipboard and keyboard (nothing is really typed), plus one round trip
through the real Windows clipboard (it is saved and put back)."""
import contextlib

import pytest

from whisperclaude import inserter

TEXT = 13 # CF_UNICODETEXT


def as_text(saved):
    data = dict(saved).get(TEXT)
    return None if data is None else data.decode("utf-16-le").rstrip("\0")


@pytest.fixture
def fake(monkeypatch):
    state = {"clipboard": [(TEXT, "old clipboard äöü\0".encode("utf-16-le")), (8, b"image")],
             "at_ctrl_v": None, "keys": [], "sequence": 0, "restores": 0}

    def put(content):
        """Any change of the clipboard, like Windows: it counts the changes."""
        state.update(clipboard=list(content), sequence=state["sequence"] + 1)

    def set_text(text):
        put([(TEXT, (text + "\0").encode("utf-16-le"))])

    class FakeKeyboard:
        def pressed(self, key):
            state["keys"].append(("hold", key))
            return contextlib.nullcontext()

        def tap(self, key):
            state["keys"].append(("tap", key))
            state["at_ctrl_v"] = as_text(state["clipboard"])

    def restore(saved):
        state["restores"] += 1
        put(saved)

    state["put"], state["set_text"] = put, set_text
    monkeypatch.setattr(inserter, "save_clipboard", lambda: list(state["clipboard"]))
    monkeypatch.setattr(inserter, "restore_clipboard", restore)
    monkeypatch.setattr(inserter, "set_clipboard_text", set_text)
    monkeypatch.setattr(inserter, "clipboard_text", lambda: as_text(state["clipboard"]))
    monkeypatch.setattr(inserter, "_GetClipboardSequenceNumber", lambda: state["sequence"])
    monkeypatch.setattr(inserter, "Controller", FakeKeyboard)
    monkeypatch.setattr(inserter, "_modifiers_down", lambda: False)
    monkeypatch.setattr(inserter.time, "sleep", lambda s: None)
    return state


def test_pastes_text_and_restores_every_format(fake):
    before = list(fake["clipboard"])
    inserter.paste_text("Grüße, neuer Text")
    assert fake["at_ctrl_v"] == "Grüße, neuer Text"
    assert fake["keys"] == [("hold", inserter.Key.ctrl), ("tap", "v")]
    assert fake["clipboard"] == before # text and image


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
    monkeypatch.setattr(inserter.time, "sleep", lambda s: s == 0.3 and fake["set_text"]("just copied"))
    inserter.paste_text("neu")
    assert as_text(fake["clipboard"]) == "just copied"


def test_text_reset_by_another_app_is_still_restored(fake, monkeypatch):
    """A clipboard manager that writes our text again changes the counter, not the text."""
    monkeypatch.setattr(inserter.time, "sleep",
                        lambda s: s == 0.3 and fake["set_text"](fake["at_ctrl_v"]))
    before = list(fake["clipboard"])
    inserter.paste_text("neu")
    assert fake["clipboard"] == before


def test_restore_does_not_need_to_read_the_clipboard(fake, monkeypatch):
    """Nobody changed it, so the comparison (which opens the clipboard) is not needed."""
    def busy():
        raise inserter.ClipboardError("busy")

    monkeypatch.setattr(inserter, "clipboard_text", busy)
    before = list(fake["clipboard"])
    inserter.paste_text("neu")
    assert fake["clipboard"] == before


def test_clipboard_is_restored_when_the_paste_fails(fake, monkeypatch):
    def broken(self, key):
        raise RuntimeError("keyboard hook gone")

    monkeypatch.setattr(inserter.Controller, "tap", broken)
    before = list(fake["clipboard"])
    with pytest.raises(RuntimeError):
        inserter.paste_text("neu")
    assert fake["clipboard"] == before


def test_clipboard_is_restored_when_putting_the_text_in_fails_halfway(fake, monkeypatch):
    def emptied_then_failed(text):
        fake["put"]([])
        raise inserter.ClipboardError("out of memory")

    monkeypatch.setattr(inserter, "set_clipboard_text", emptied_then_failed)
    before = list(fake["clipboard"])
    with pytest.raises(inserter.ClipboardError):
        inserter.paste_text("neu")
    assert fake["clipboard"] == before and fake["keys"] == []


def test_nothing_is_pasted_if_the_old_clipboard_cannot_be_saved(fake, monkeypatch):
    def busy():
        raise inserter.ClipboardError("busy")

    monkeypatch.setattr(inserter, "save_clipboard", busy)
    before = list(fake["clipboard"])
    with pytest.raises(inserter.ClipboardError):
        inserter.paste_text("neu")
    assert fake["clipboard"] == before and fake["keys"] == [] and fake["restores"] == 0


@pytest.fixture
def selection(fake, monkeypatch):
    """capture_selection with an app that copies "selected" into the clipboard."""
    def copy(timeout):
        fake["set_text"]("selected")
        return True

    monkeypatch.setattr(inserter, "_copy_selection", copy)
    monkeypatch.setattr(inserter, "_read_copied", lambda: (as_text(fake["clipboard"]), b""))
    return fake


def test_selection_is_read_and_the_clipboard_comes_back(selection):
    before = list(selection["clipboard"])
    assert inserter.capture_selection() == "selected"
    assert selection["clipboard"] == before


def test_clipboard_comes_back_when_the_selection_cannot_be_read(selection, monkeypatch):
    def busy():
        raise inserter.ClipboardError("busy")

    monkeypatch.setattr(inserter, "_read_copied", busy)
    before = list(selection["clipboard"])
    with pytest.raises(inserter.ClipboardError):
        inserter.capture_selection()
    assert selection["clipboard"] == before


def test_nothing_selected_leaves_the_clipboard_alone(fake, monkeypatch):
    monkeypatch.setattr(inserter, "_copy_selection", lambda timeout: False)
    before, sequence = list(fake["clipboard"]), fake["sequence"]
    assert inserter.capture_selection() is None
    assert fake["clipboard"] == before and fake["sequence"] == sequence and fake["restores"] == 0


def test_no_selection_is_copied_if_the_old_clipboard_cannot_be_saved(fake, monkeypatch):
    def busy():
        raise inserter.ClipboardError("busy")

    monkeypatch.setattr(inserter, "save_clipboard", busy)
    monkeypatch.setattr(inserter, "_copy_selection", lambda timeout: pytest.fail("copied"))
    with pytest.raises(inserter.ClipboardError):
        inserter.capture_selection()


def test_failed_clipboard_restore_is_not_a_paste_error(fake, monkeypatch):
    def locked(saved):
        raise inserter.ClipboardError("clipboard locked")

    monkeypatch.setattr(inserter, "restore_clipboard", locked)
    inserter.paste_text("neu") # must not raise: the text was pasted
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


def test_real_clipboard_is_restored_unless_the_user_copied_something():
    users = inserter.save_clipboard()
    old = [(TEXT, "alt\0".encode("utf-16-le"))]
    try:
        inserter._write(old)
        inserter.set_clipboard_text("Diktat")
        sequence = inserter._GetClipboardSequenceNumber()
        inserter._restore_unless_copied(old, "Diktat", sequence) # nobody touched it
        assert inserter.clipboard_text() == "alt"
        inserter.set_clipboard_text("Diktat")
        sequence = inserter._GetClipboardSequenceNumber()
        inserter.set_clipboard_text("neu kopiert") # the user copied during the paste
        inserter._restore_unless_copied(old, "Diktat", sequence)
        assert inserter.clipboard_text() == "neu kopiert"
        inserter.set_clipboard_text("Diktat")
        inserter._restore_unless_copied(old, "Diktat", None) # putting our text in had failed
        assert inserter.clipboard_text() == "alt"
    finally:
        inserter.restore_clipboard(users)


def test_dictation_goes_into_win_v_history_but_never_into_the_cloud():
    users = inserter.save_clipboard()
    no_history = inserter._RegisterClipboardFormatW("CanIncludeInClipboardHistory")
    no_cloud = inserter._RegisterClipboardFormatW("CanUploadToCloudClipboard")
    try:
        inserter.set_clipboard_text("Diktat")
        formats = {f for f, _ in inserter.save_clipboard()}
        assert no_cloud in formats and no_history not in formats
        inserter.restore_clipboard([(TEXT, "alt\0".encode("utf-16-le"))])
        formats = {f for f, _ in inserter.save_clipboard()}
        assert {no_cloud, no_history} <= formats # the old content isn't a new history entry
    finally:
        inserter.restore_clipboard(users)
