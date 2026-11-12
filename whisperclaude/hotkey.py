"""Global hotkeys: a short tap of Left Ctrl (alone) puts a TOGGLE event into a queue, a tap
of the translate key (Left Alt) a TRANSLATE event."""
from __future__ import annotations

import ctypes
import logging
import queue
import time
from collections.abc import Callable

from pynput import mouse
from pynput.keyboard import Key, KeyCode, Listener

from whisperclaude.app import CANCEL, TOGGLE, TRANSLATE

log = logging.getLogger(__name__)

DEFAULT_KEY = Key.ctrl_l
MAX_HOLD_S = 0.5

WM_KEYDOWN, WM_SYSKEYDOWN = 0x100, 0x104
LLKHF_EXTENDED = 0x01
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 0x1, 0x2
MASK_VK = 0xE8 # unassigned virtual key (AutoHotkey's "menu mask key" does the same)
MASKED_VKS = {0x12, 0xA4, 0xA5, 0x5B, 0x5C} # Alt, Left/Right Alt, Left/Right Win
OWN_EVENT = 0x57434C44 # dwExtraInfo of the key events we send ourselves ("WCLD")


class TapDetector:
    """Detects a tap: target key pressed and released within max_hold, nothing in between.

    "Nothing" = no other key, no mouse click or scroll, so Ctrl+C, Ctrl+click and
    Ctrl+scroll (zoom) don't count. Pure logic, no pynput thread, so it can be tested directly.
    """

    def __init__(
        self,
        target: Key | KeyCode = DEFAULT_KEY,
        max_hold: float = MAX_HOLD_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.target = target
        self.max_hold = max_hold
        self.clock = clock
        self._down_at: float | None = None
        self._interrupted = False

    def press(self, key: Key | KeyCode | None) -> None:
        if key == self.target:
            if self._down_at is None: # ignore auto-repeat while held
                self._down_at = self.clock()
                self._interrupted = False
        else:
            self.interrupt()

    def interrupt(self) -> None:
        """Something else happened while the target key is down: not a tap."""
        if self._down_at is not None:
            self._interrupted = True

    def release(self, key: Key | KeyCode | None) -> bool:
        """Return True if this release completes a tap."""
        if key != self.target or self._down_at is None:
            return False
        held = self.clock() - self._down_at
        self._down_at = None
        return not self._interrupted and held <= self.max_hold


GENERIC_MODIFIERS = {"ctrl", "alt"} # VK 0x11 / 0x12: the hook only sends the _l / _r variants


def parse_key(name: str) -> Key | KeyCode:
    """Config name -> pynput key: 'ctrl_l', 'ctrl_r', 'f9', 'pause', ... or a single character.

    The low-level hook reports the left/right variants only (Key.ctrl_l, Key.alt_l, ...), so the
    generic 'ctrl' and 'alt' would never match anything: rejected with a hint, not guessed
    (which side did the user mean?). 'alt_r' is what pynput reports as Key.alt_gr (same VK 0xA5).
    """
    if name in GENERIC_MODIFIERS:
        raise ValueError(f"{name!r} is ambiguous, say which one: "
                         f"'{name}_l' (left) or '{name}_r' (right)")
    if name == "alt_r":
        return Key.alt_gr
    if name in Key.__members__:
        return Key[name]
    if len(name) == 1:
        return KeyCode.from_char(name)
    raise ValueError(f"unknown key {name!r} (use e.g. 'ctrl_l', 'ctrl_r', 'f9', 'pause')")


def key_label(name: str) -> str:
    """Human-readable key name for the UI."""
    labels = {"ctrl_l": "Left Ctrl", "ctrl_r": "Right Ctrl", "alt_l": "Left Alt",
              "shift_r": "Right Shift", "caps_lock": "Caps Lock"}
    return labels.get(name, name.replace("_", " ").title())


def key_vk(key: Key | KeyCode) -> int:
    """Windows virtual-key code of a pynput key (0 if it has none, e.g. a plain character)."""
    code = key.value if isinstance(key, Key) else key
    return code.vk or 0


def _send_masked_release(vk: int, scan: int, flags: int) -> None:
    """Replay a swallowed key-up with a dummy key in front of it.

    A tap of Alt alone opens the focused app's menu bar (Office: key tips; VS Code, Explorer,
    Notepad: the menu), a tap of Win the Start menu. Windows does that on the key-up, and only
    if no other key came in between. So the hook swallows the real key-up and sends "dummy key
    down, dummy key up, key up" instead: the app sees a combination and does nothing.
    """
    extended = KEYEVENTF_EXTENDEDKEY if flags & LLKHF_EXTENDED else 0
    user32 = ctypes.windll.user32
    user32.keybd_event(MASK_VK, 0, 0, OWN_EVENT)
    user32.keybd_event(MASK_VK, 0, KEYEVENTF_KEYUP, OWN_EVENT)
    user32.keybd_event(vk, scan, KEYEVENTF_KEYUP | extended, OWN_EVENT)


class TranslateKeyFilter:
    """pynput's win32_event_filter: detects a tap of the translate key inside the hook.

    on_tap() is called on a tap. For Alt and Win the tap's key-up is swallowed (suppress()
    raises, which makes Windows drop the event) and replayed with a dummy key in front, see
    _send_masked_release. That replay is ignored here (marked with OWN_EVENT).
    """

    def __init__(self, vk: int, max_hold: float, on_tap: Callable[[], None],
                 suppress: Callable[[], None], clock: Callable[[], float] = time.monotonic,
                 send: Callable[[int, int, int], None] = _send_masked_release) -> None:
        self.tap = TapDetector(vk, max_hold, clock) # on virtual-key codes
        self.vk, self.on_tap, self.suppress, self.send = vk, on_tap, suppress, send

    def interrupt(self) -> None:
        self.tap.interrupt()

    def __call__(self, msg: int, data: ctypes.Structure) -> bool:
        if not self.vk or (data.dwExtraInfo or 0) == OWN_EVENT:
            return True
        try:
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                self.tap.press(data.vkCode)
                return True
            if not self.tap.release(data.vkCode):
                return True
            self.on_tap()
            if data.vkCode not in MASKED_VKS:
                return True
            self.send(data.vkCode, data.scanCode, data.flags)
        except Exception: # any other exception would stop pynput's listener
            log.exception("translate key filter failed")
            return True
        self.suppress() # raises: Windows drops the original key-up
        return False


class HotkeyListener:
    """Keyboard + mouse listeners (pynput threads) feeding the TapDetectors.

    The callbacks run inside Windows low-level hooks, which Windows silently removes if a
    callback is slow, so they only do trivial work (events.put never waits for the consumer).
    The translate key is detected in pynput's event filter, which runs inside the hook itself
    (the other callbacks run later, on pynput's thread): only there can its key-up be swallowed.
    """

    def __init__(self, events: queue.Queue, key: Key | KeyCode = DEFAULT_KEY,
                 max_hold: float = MAX_HOLD_S, cancel_key: Key | KeyCode | None = None,
                 translate_key: Key | KeyCode | None = None) -> None:
        self._events, self._key, self._max_hold = events, key, max_hold
        self._cancel_key = cancel_key
        self._translate_vk = key_vk(translate_key) if translate_key is not None else 0
        self._start()

    def _start(self) -> None:
        tap = TapDetector(self._key, self._max_hold)
        translate = TranslateKeyFilter(self._translate_vk, self._max_hold,
                                       lambda: self._events.put((TRANSLATE, None)),
                                       lambda: self._keyboard.suppress_event())

        # `injected`: Windows marks key events made by software. A precision-touchpad pinch reaches
        # apps as Ctrl + wheel with an injected Ctrl press and release; with the wheel part not
        # passing our mouse hook, that looked like a Ctrl tap and started recordings by itself
        # (seen with two fingers on the touchpad). Injected record-key events never form a tap;
        # injected other keys still interrupt one.
        def on_press(k: Key | KeyCode | None, injected: bool = False) -> None:
            if injected and k == self._key:
                return
            tap.press(k)
            if k is not None and k == self._cancel_key: # App ignores it unless recording
                self._events.put((CANCEL, None))

        def on_release(k: Key | KeyCode | None, injected: bool = False) -> None:
            if injected and k == self._key:
                return
            if tap.release(k):
                self._events.put((TOGGLE, None))

        def interrupt() -> None:
            tap.interrupt()
            translate.interrupt()

        self._keyboard = Listener(on_press=on_press, on_release=on_release,
                                  win32_event_filter=translate)
        self._mouse = mouse.Listener(
            on_click=lambda x, y, button, pressed: interrupt(),
            on_scroll=lambda x, y, dx, dy: interrupt(),
        )
        for listener in (self._keyboard, self._mouse):
            listener.daemon = True
            listener.start()

    def stop(self) -> None:
        self._keyboard.stop()
        self._mouse.stop()

    def restart(self) -> None:
        """Install fresh hooks (e.g. after a wake from sleep, in case Windows dropped the old)."""
        self.stop()
        self._start()


class GapDetector:
    """Detects that the machine was suspended: a periodic check that suddenly sees a long gap.

    Uses the wall clock, which keeps running during sleep (unlike monotonic clocks, which may
    not on Windows). A clock change by the user also triggers it, which is harmless here.
    """

    def __init__(self, threshold_s: float = 15.0, clock: Callable[[], float] = time.time) -> None:
        self.threshold_s = threshold_s
        self.clock = clock
        self._last = clock()

    def check(self) -> float:
        """Call periodically (every ~1 s). Returns the gap in seconds if it exceeded the
        threshold since the previous call, else 0."""
        now = self.clock()
        gap, self._last = now - self._last, now
        return gap if gap > self.threshold_s else 0.0


def start_hotkey_listener(events: queue.Queue, key: Key | KeyCode = DEFAULT_KEY,
                          max_hold: float = MAX_HOLD_S,
                          cancel_key: Key | KeyCode | None = None,
                          translate_key: Key | KeyCode | None = None) -> HotkeyListener:
    """Start listening. Call .stop() on the result to end it."""
    return HotkeyListener(events, key, max_hold, cancel_key, translate_key)
