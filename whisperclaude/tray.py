"""System tray icon (pystray, own thread): shows the state (animated, see tray_icons.py); menu
with status, recent dictations (click one to copy it), settings, log, restart and quit."""
from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from whisperclaude.app import State
from whisperclaude.history import History
from whisperclaude.tray_icons import ANIMATED, CYCLE_S, render

log = logging.getLogger(__name__)
try:  # pystray's own tray-icon callback message (right/left click), not Windows' WM_NOTIFY
    from pystray._util.win32 import NIF_ICON, NIM_MODIFY, WM_NOTIFY
except ImportError:  # not on Windows
    NIF_ICON = NIM_MODIFY = WM_NOTIFY = -1

COLORS = {State.LOADING: "#9aa0a6", State.IDLE: "#4caf50", State.RECORDING: "#ff4d4f",
          State.TRANSCRIBING: "#f5a623"}
FPS = 10
PASTED_FLASH_S = 1.2
QUIT_QUESTION = ("Quit WhisperClaude?\n\nDictation stays off until you start it again "
                 "or log in the next time.")


def confirm_quit() -> bool:
    """Yes/No dialog; "No" is the default button, so a stray Enter or click doesn't quit."""
    flags = 0x4 | 0x20 | 0x100 | 0x10000 | 0x40000  # YESNO, QUESTION, DEFBUTTON2, FOREGROUND, TOPMOST
    return ctypes.windll.user32.MessageBoxW(None, QUIT_QUESTION, "WhisperClaude", flags) == 6


def _image(color: str) -> Image.Image:
    """The "classic" icon: colored circle with a simple white microphone."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((2, 2, 62, 62), fill=color)
    d.rounded_rectangle((25, 12, 39, 38), radius=7, fill="white")        # capsule
    d.arc((18, 22, 46, 46), start=0, end=180, fill="white", width=3)     # holder
    d.line((32, 46, 32, 52), fill="white", width=3)                      # stand
    d.line((24, 52, 40, 52), fill="white", width=3)                      # base
    return img


# ---------- in-memory icons (pystray writes every icon change to a temp .ico file) ----------

class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class _ICONINFO(ctypes.Structure):
    _fields_ = [("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD),
                ("yHotspot", wintypes.DWORD), ("hbmMask", wintypes.HBITMAP),
                ("hbmColor", wintypes.HBITMAP)]


# Own DLL instances with declared types: 64-bit handles overflow ctypes' default int arguments.
_gdi = ctypes.WinDLL("gdi32")
_user = ctypes.WinDLL("user32")
_gdi.CreateDIBSection.restype = wintypes.HBITMAP
_gdi.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
                                  ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
_gdi.CreateBitmap.restype = wintypes.HBITMAP
_gdi.CreateBitmap.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.UINT,
                              ctypes.c_void_p]
_gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_user.CreateIconIndirect.restype = wintypes.HICON
_user.CreateIconIndirect.argtypes = [ctypes.c_void_p]


def hicon_from_image(img: Image.Image) -> int:
    """A Windows icon handle (32-bit with alpha) straight from a PIL image, no file."""
    gdi, user = _gdi, _user
    img = img.convert("RGBA")
    w, h = img.size
    header = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    bits = ctypes.c_void_p()
    color = gdi.CreateDIBSection(None, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
    data = img.tobytes("raw", "BGRA")
    ctypes.memmove(bits, data, len(data))
    mask = gdi.CreateBitmap(w, h, 1, 1, bytes(((w + 15) // 16) * 2 * h))
    info = _ICONINFO(True, 0, 0, mask, color)
    handle = user.CreateIconIndirect(ctypes.byref(info))
    gdi.DeleteObject(color)
    gdi.DeleteObject(mask)
    return handle


class _Animator:
    """Shows the state as an animated tray icon (styles in tray_icons.py).

    Frames are built once as in-memory icons at the real tray size and cached; changing a
    frame is one Shell_NotifyIcon call. The thread only ticks (FPS) while something animates:
    idle, it sleeps until the next state change, so an idle app costs nothing.
    """

    def __init__(self, icon: pystray.Icon, style: str, level_fn: Callable[[], float]) -> None:
        self.icon, self.style, self.level_fn = icon, style, level_fn
        self.size = ctypes.windll.user32.GetSystemMetrics(49) or 16  # SM_CXSMICON, DPI-scaled
        self._cache: dict[tuple, int] = {}
        self._state, self._since = "loading", time.monotonic()
        self._flash_since = -1.0
        self._wake = threading.Event()
        self._stopped = False
        threading.Thread(target=self._run, daemon=True, name="tray-animation").start()

    def set(self, state: str) -> None:
        self._state, self._since, self._flash_since = state, time.monotonic(), -1.0
        self._wake.set()

    def flash_pasted(self) -> None:
        self._flash_since = time.monotonic()
        self._wake.set()

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()

    def _frame_key(self, now: float) -> tuple:
        state, t = self._state, now - self._since
        if state == "idle" and 0 <= now - self._flash_since < PASTED_FLASH_S:
            return ("pasted", min(int((now - self._flash_since) * FPS), 4), 0)
        if state not in ANIMATED:
            return (state, 0, 0)
        level = round(min(1.0, max(0.0, self.level_fn())) * 9) if state == "recording" else 0
        return (state, int(t * FPS) % int(CYCLE_S * FPS), level)

    def _run(self) -> None:
        shown, failed = None, False
        while not self._stopped:
            key = self._frame_key(time.monotonic())
            try:
                if key != shown and self._push(key):
                    shown = key
            except Exception:  # never let a drawing problem end the thread silently
                if not failed:
                    log.exception("tray animation frame failed")
                failed, shown = True, key
            moving = key[0] in ANIMATED
            self._wake.wait(1 / FPS if moving else None)
            self._wake.clear()

    def _push(self, key: tuple) -> bool:
        if not getattr(self.icon, "_hwnd", None):
            return False  # tray icon not shown yet
        handle = self._cache.get(key)
        if handle is None:
            state, step, level = key
            img = render(self.style, state, step / FPS, level / 9)
            handle = self._cache[key] = hicon_from_image(
                img.resize((self.size, self.size), Image.LANCZOS))
        self.icon._icon_handle = handle  # so pystray re-adds this one after an explorer restart
        self.icon._message(NIM_MODIFY, NIF_ICON, hIcon=handle)
        return True


# ---------- the tray icon and its menu ----------

class Tray:
    def __init__(self, on_quit: Callable[[], None], on_restart: Callable[[], None],
                 log_path: Path, config_path: Path, hotkey_label: str = "Left Ctrl",
                 history: History | None = None,
                 on_copy: Callable[[str], None] = lambda text: None,
                 style: str = "classic", level_fn: Callable[[], float] = lambda: 0.0,
                 confirm: Callable[[], bool] = confirm_quit) -> None:
        self._state = State.LOADING
        self.engine = ""  # "iGPU" / "CPU", set once the model is loaded
        self._labels = {State.LOADING: "Loading model…", State.IDLE: f"Ready (tap {hotkey_label})",
                        State.RECORDING: "Recording…", State.TRANSCRIBING: "Transcribing…"}
        self._images = {s: _image(c) for s, c in COLORS.items()}
        self._history, self._on_copy = history, on_copy
        self._menu_lock = threading.Lock()
        self._menu_open = False
        self._menu_stale = False
        on_quit, on_restart = self._logged("Quit", on_quit), self._logged("Restart", on_restart)
        ask_quit = on_quit

        def on_quit() -> None:  # runs on the tray thread, the menu is closed already
            if confirm():
                ask_quit()
            else:
                log.info("tray menu: Quit cancelled")
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: f"WhisperClaude · {self._status()}", None,
                             enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Recent dictations", pystray.Menu(self._recent_items),
                             visible=history is not None and history.keep > 0),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings (config.toml)",
                             lambda icon, item: os.startfile(config_path)),
            pystray.MenuItem("Restart to apply settings", lambda icon, item: on_restart()),
            pystray.MenuItem("Open log", lambda icon, item: os.startfile(log_path)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit WhisperClaude", lambda icon, item: on_quit()),
        )
        animated = style != "classic"
        first = render(style, "loading") if animated else self._images[State.LOADING]
        self.icon = pystray.Icon("WhisperClaude", first,
                                 f"WhisperClaude – {self._labels[State.LOADING]}", menu)
        self._animator = _Animator(self.icon, style, level_fn) if animated else None
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
        """Submenu, rebuilt on update_menu(): preview left, time right-aligned (the tab puts
        it in the menu's second column), newest first and in bold."""
        entries = self._history.entries() if self._history else []
        if not entries:
            return [pystray.MenuItem("No dictations yet", None, enabled=False)]
        items = [pystray.MenuItem("Click to copy", None, enabled=False)]
        items += [pystray.MenuItem(f"{e.preview()}\t{e.when()}", self._copy_action(e.text),
                                   default=(i == 0))
                  for i, e in enumerate(entries)]
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
        if self._animator is not None:
            self._animator.set(state.value)
        else:
            self.icon.icon = self._images[state]
        self.icon.title = f"WhisperClaude – {self._status()}"
        self.refresh()

    def flash_pasted(self) -> None:
        """A short check mark after a paste (animated styles only)."""
        if self._animator is not None:
            self._animator.flash_pasted()

    def stop(self) -> None:
        if self._animator is not None:
            self._animator.stop()
        self.icon.stop()
