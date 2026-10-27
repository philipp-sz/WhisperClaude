"""System tray icon (pystray, own thread): colored by state; menu with status, recent
dictations (click one to copy it), config, log, restart and quit."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from whisperclaude.app import State
from whisperclaude.history import History

log = logging.getLogger(__name__)
try:  # pystray's own tray-icon callback message (right/left click), not Windows' WM_NOTIFY
    from pystray._util.win32 import WM_NOTIFY
except ImportError:  # not on Windows
    WM_NOTIFY = -1

COLORS = {State.LOADING: "#9aa0a6", State.IDLE: "#4caf50", State.RECORDING: "#ff4d4f",
          State.TRANSCRIBING: "#f5a623"}


def _image(color: str) -> Image.Image:
    """Colored circle with a simple white microphone."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((2, 2, 62, 62), fill=color)
    d.rounded_rectangle((25, 12, 39, 38), radius=7, fill="white")        # capsule
    d.arc((18, 22, 46, 46), start=0, end=180, fill="white", width=3)     # holder
    d.line((32, 46, 32, 52), fill="white", width=3)                      # stand
    d.line((24, 52, 40, 52), fill="white", width=3)                      # base
    return img


class Tray:
    def __init__(self, on_quit: Callable[[], None], on_restart: Callable[[], None],
                 log_path: Path, config_path: Path, hotkey_label: str = "Left Ctrl",
                 history: History | None = None,
                 on_copy: Callable[[str], None] = lambda text: None) -> None:
        self._state = State.LOADING
        self.engine = ""  # "iGPU" / "CPU", set once the model is loaded
        self._labels = {State.LOADING: "Loading model…", State.IDLE: f"Ready (tap {hotkey_label})",
                        State.RECORDING: "Recording", State.TRANSCRIBING: "Transcribing"}
        self._images = {s: _image(c) for s, c in COLORS.items()}
        self._history, self._on_copy = history, on_copy
        self._menu_lock = threading.Lock()
        self._menu_open = False
        self._menu_stale = False
        on_quit, on_restart = self._logged("Quit", on_quit), self._logged("Restart", on_restart)
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: f"Status: {self._status()}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Recent dictations", pystray.Menu(self._recent_items),
                             visible=history is not None and history.keep > 0),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open config", lambda icon, item: os.startfile(config_path)),
            pystray.MenuItem("Restart (apply config)", lambda icon, item: on_restart()),
            pystray.MenuItem("Open log", lambda icon, item: os.startfile(log_path)),
            pystray.MenuItem("Quit", lambda icon, item: on_quit()),
        )
        self.icon = pystray.Icon("WhisperClaude", self._images[State.LOADING],
                                 f"WhisperClaude – {self._labels[State.LOADING]}", menu)
        self._guard_menu()

    @staticmethod
    def _logged(name: str, action: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            log.info("tray menu: %s", name)  # a quit is always traceable in the log
            action()
        return run

    def _guard_menu(self) -> None:
        """Never rebuild the menu while it is open, and never from two threads at once.

        pystray rebuilds the Win32 menu in whichever thread calls update_menu() (here: the
        worker after each dictation, the Tk thread on state changes, pystray itself after a
        click), even while that menu is on screen. Windows doesn't guard an open menu being
        destroyed, and freed menu handles get reused, so the menu shown and the action list
        pystray dispatches clicks to could disagree. Suspected after the app quit itself when
        a dictation was clicked in "Recent dictations" (log: no copy, then "quitting").
        """
        handlers = getattr(self.icon, "_message_handlers", None)
        if not isinstance(handlers, dict) or WM_NOTIFY not in handlers:
            return  # not pystray's Win32 backend
        show_menu = handlers[WM_NOTIFY]

        def on_notify(wparam: int, lparam: int) -> int:
            with self._menu_lock:
                self._menu_open = True
            try:
                return show_menu(wparam, lparam)  # blocks while the menu is open
            finally:
                with self._menu_lock:
                    self._menu_open = False
                if self._menu_stale:
                    self.refresh()

        handlers[WM_NOTIFY] = on_notify
        self.icon.update_menu = self.refresh  # pystray's own rebuild after a click, too

    def _recent_items(self) -> list[pystray.MenuItem]:
        """Submenu, rebuilt on update_menu(): one item per dictation, newest first."""
        entries = self._history.entries() if self._history else []
        if not entries:
            return [pystray.MenuItem("No dictations yet", None, enabled=False)]
        items = [pystray.MenuItem("Click one to copy it", None, enabled=False)]
        items += [pystray.MenuItem(e.label(), self._copy_action(e.text)) for e in entries]
        return items + [pystray.Menu.SEPARATOR,
                        pystray.MenuItem("Clear history", lambda icon, item: self._clear_history())]

    def _copy_action(self, text: str) -> Callable[[object, object], None]:
        # exactly (icon, item): pystray rejects actions with more parameters, defaults included
        def copy(icon: object, item: object) -> None:
            log.info("tray menu: copy a dictation")
            self._on_copy(text)
        return copy

    def _clear_history(self) -> None:
        if self._history:
            self._history.clear()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the menu (state or history changed); deferred while the menu is open."""
        with self._menu_lock:
            if self._menu_open:
                self._menu_stale = True
                return
            self._menu_stale = False
            type(self.icon).update_menu(self.icon)

    def _status(self) -> str:
        label = self._labels[self._state]
        return f"{label} · {self.engine}" if self.engine else label

    def start(self) -> None:
        self.icon.run_detached()

    def set_state(self, state: State, msg: str = "") -> None:
        self._state = state
        self.icon.icon = self._images[state]
        self.icon.title = f"WhisperClaude – {self._status()}"
        self.refresh()

    def stop(self) -> None:
        self.icon.stop()
