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
    assert labels[0] == "Click one to copy it"
    assert labels[1].endswith("zweites Diktat") and labels[2].endswith("erstes Diktat")
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
