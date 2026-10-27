"""Tray menu "Recent dictations" (built, not shown: no icon thread is started)."""
from pathlib import Path

from whisperclaude.history import History
from whisperclaude.tray import Tray


def make_tray(tmp_path, keep=5):
    copied = []
    history = History(tmp_path, keep)
    tray = Tray(lambda: None, lambda: None, Path("log.txt"), Path("config.toml"),
                history=history, on_copy=copied.append)
    return tray, history, copied


def texts(items):
    return [str(i.text) for i in items]


def test_empty_history(tmp_path):
    tray, _, _ = make_tray(tmp_path)
    assert texts(tray._recent_items()) == ["No dictations yet"]


def test_click_copies_that_dictation_and_clear_empties_it(tmp_path):
    tray, history, copied = make_tray(tmp_path)
    history.add("erstes Diktat")
    history.add("zweites Diktat")
    items = tray._recent_items()
    labels = texts(items)
    assert labels[0] == "Click to copy"
    assert labels[1].startswith("zweites Diktat\t") and labels[2].startswith("erstes Diktat\t")
    assert items[1].default and not items[2].default  # the newest is shown in bold
    items[2](tray.icon)  # click "erstes Diktat"
    assert copied == ["erstes Diktat"]
    texts_after = tray._recent_items()
    texts_after[-1](tray.icon)  # "Clear history"
    assert history.entries() == []


def test_menu_entry_hidden_when_history_is_off(tmp_path):
    tray, _, _ = make_tray(tmp_path, keep=0)
    recent = next(i for i in tray.icon.menu.items if str(i.text) == "Recent dictations")
    assert not recent.visible


def test_menu_is_never_rebuilt_while_it_is_open(monkeypatch, tmp_path):
    """The suspected cause of a click on a dictation quitting the app: the open menu was
    rebuilt underneath (from another thread, or by pystray after the click)."""
    import threading

    import pystray._win32 as win

    from whisperclaude.tray import WM_NOTIFY

    builds, seen_while_open = [], []

    def menu_open(icon, wparam, lparam):  # stands in for pystray showing the menu
        other = threading.Thread(target=tray.refresh)  # e.g. a dictation finished meanwhile
        other.start()
        other.join()
        icon.update_menu()  # pystray rebuilds after every click
        seen_while_open.append(len(builds))

    monkeypatch.setattr(win.Icon, "_on_notify", menu_open)
    monkeypatch.setattr(win.Icon, "_update_menu", lambda icon: builds.append(1))
    tray, _, _ = make_tray(tmp_path)
    tray.icon._message_handlers[WM_NOTIFY](0, 0x205)  # right-click
    assert seen_while_open == [0] and len(builds) == 1  # deferred, then rebuilt once
    tray.refresh()
    assert len(builds) == 2  # menu closed: rebuilds happen right away again


# ---------- animated icon ----------

def test_icons_are_made_in_memory():
    import ctypes

    from whisperclaude.tray import hicon_from_image
    from whisperclaude.tray_icons import STATES, STYLES, render

    for style in STYLES:
        for state in STATES:
            handle = hicon_from_image(render(style, state, 0.3, 0.5).resize((32, 32)))
            assert handle
            ctypes.windll.user32.DestroyIcon(ctypes.c_void_p(handle))


def test_frames_follow_state_voice_and_paste():
    from whisperclaude import tray as t

    class FakeIcon:
        _hwnd = None  # never shown: nothing is pushed

    anim = t._Animator(FakeIcon(), "bars", level_fn=lambda: 1.0)
    try:
        now = anim._since
        assert anim._frame_key(now)[0] == "loading"
        anim.set("recording")
        assert anim._frame_key(anim._since + 0.05) == ("recording", 0, 9)  # loud voice
        anim.set("idle")
        assert anim._frame_key(anim._since) == ("idle", 0, 0)  # idle: one still frame
        anim.flash_pasted()
        assert anim._frame_key(anim._flash_since + 0.1)[0] == "pasted"
        assert anim._frame_key(anim._flash_since + t.PASTED_FLASH_S + 0.1)[0] == "idle"
    finally:
        anim.stop()


def test_classic_style_has_no_animation(tmp_path):
    tray, _, _ = make_tray(tmp_path)  # make_tray uses the default style: classic
    assert tray._animator is None


def test_quit_asks_first(tmp_path):
    for answer, quits in ((False, 0), (True, 1)):
        quit_calls = []
        tray = Tray(lambda: quit_calls.append(1), lambda: None, Path("log"), Path("cfg"),
                    confirm=lambda answer=answer: answer)
        quit_item = next(i for i in tray.icon.menu.items if str(i.text) == "Quit WhisperClaude")
        quit_item(tray.icon)
        assert len(quit_calls) == quits
