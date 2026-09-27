"""Global toggle hotkey: each press puts a TOGGLE event into a queue."""
from __future__ import annotations

import queue

from pynput.keyboard import GlobalHotKeys

from whisperclaude.app import TOGGLE

DEFAULT_HOTKEY = "<ctrl>+<alt>+<space>"


def start_hotkey_listener(events: queue.Queue, hotkey: str = DEFAULT_HOTKEY) -> GlobalHotKeys:
    """Start listening in pynput's own thread. Call .stop() on the result to end it."""
    listener = GlobalHotKeys({hotkey: lambda: events.put((TOGGLE, None))})
    listener.daemon = True
    listener.start()
    return listener
