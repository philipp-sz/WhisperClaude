"""Global hotkey: a short tap of Left Ctrl (alone) puts a TOGGLE event into a queue."""
from __future__ import annotations

import queue
import time
from typing import Callable

from pynput import mouse
from pynput.keyboard import Key, KeyCode, Listener

from whisperclaude.app import TOGGLE

DEFAULT_KEY = Key.ctrl_l
MAX_HOLD_S = 0.5


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
            if self._down_at is None:  # ignore auto-repeat while held
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


def parse_key(name: str) -> Key | KeyCode:
    """Config name -> pynput key: 'ctrl_l', 'ctrl_r', 'f9', 'pause', ... or a single character."""
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


class HotkeyListener:
    """Keyboard + mouse listeners (pynput threads) feeding one TapDetector.

    The callbacks run inside Windows low-level hooks, which Windows silently removes if a
    callback is slow, so they only do trivial work (events.put never waits for the consumer).
    """

    def __init__(self, events: queue.Queue, key: Key | KeyCode = DEFAULT_KEY,
                 max_hold: float = MAX_HOLD_S) -> None:
        self._events, self._key, self._max_hold = events, key, max_hold
        self._start()

    def _start(self) -> None:
        tap = TapDetector(self._key, self._max_hold)

        def on_release(k: Key | KeyCode | None) -> None:
            if tap.release(k):
                self._events.put((TOGGLE, None))

        self._keyboard = Listener(on_press=tap.press, on_release=on_release)
        self._mouse = mouse.Listener(
            on_click=lambda x, y, button, pressed: tap.interrupt(),
            on_scroll=lambda x, y, dx, dy: tap.interrupt(),
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
                          max_hold: float = MAX_HOLD_S) -> HotkeyListener:
    """Start listening. Call .stop() on the result to end it."""
    return HotkeyListener(events, key, max_hold)
