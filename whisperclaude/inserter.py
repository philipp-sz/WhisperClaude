"""Insert text into the focused app: clipboard save -> set -> Ctrl+V -> restore."""
from __future__ import annotations

import ctypes
import logging
import time

import pyperclip
from pynput.keyboard import Controller, Key

log = logging.getLogger(__name__)

# Virtual-key codes: Shift, Ctrl, Alt, left/right Win
_MODIFIER_VKS = (0x10, 0x11, 0x12, 0x5B, 0x5C)


def _modifiers_down() -> bool:
    """True if the user is still physically holding a modifier key."""
    user32 = ctypes.windll.user32
    return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in _MODIFIER_VKS)


def wait_for_modifiers_released(timeout: float = 2.0) -> None:
    """Wait until Ctrl/Alt/Shift/Win are up, so our Ctrl+V isn't Ctrl+Alt+V.

    We wait instead of sending fake key-ups: a synthetic Alt-up can open an
    app's menu bar.
    """
    deadline = time.monotonic() + timeout
    while _modifiers_down():
        if time.monotonic() > deadline:
            log.warning("modifier keys still held after %.1f s, pasting anyway", timeout)
            return
        time.sleep(0.02)


def paste_text(text: str, restore_delay: float = 0.3) -> None:
    """Paste text into whatever has focus and restore the previous clipboard.

    Only text clipboard content is restored; images etc. are lost (v1 limitation).
    """
    if not text:
        return
    try:
        old = pyperclip.paste()
    except pyperclip.PyperclipException:
        old = None
    pyperclip.copy(text)

    wait_for_modifiers_released()
    time.sleep(0.05)
    kb = Controller()
    with kb.pressed(Key.ctrl):
        kb.tap("v")

    # The target app reads the clipboard asynchronously; restoring too early
    # would paste the old content instead.
    time.sleep(restore_delay)
    if old:
        pyperclip.copy(old)
