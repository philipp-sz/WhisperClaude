"""System tray icon (pystray, own thread): colored by state, menu with status / open log / quit."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from whisperclaude.app import State

COLORS = {None: "#9aa0a6", State.IDLE: "#4caf50", State.RECORDING: "#ff4d4f",
          State.TRANSCRIBING: "#f5a623"}
LABELS = {None: "Loading model…", State.IDLE: "Ready (tap Left Ctrl)",
          State.RECORDING: "Recording", State.TRANSCRIBING: "Transcribing"}


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
    def __init__(self, on_quit: Callable[[], None], log_path: Path) -> None:
        self._state: State | None = None
        self._images = {s: _image(c) for s, c in COLORS.items()}
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: f"Status: {LABELS[self._state]}", None, enabled=False),
            pystray.MenuItem("Open log", lambda icon, item: os.startfile(log_path)),
            pystray.MenuItem("Quit", lambda icon, item: on_quit()),
        )
        self.icon = pystray.Icon("WhisperClaude", self._images[None],
                                 f"WhisperClaude – {LABELS[None]}", menu)

    def start(self) -> None:
        self.icon.run_detached()

    def set_state(self, state: State, msg: str = "") -> None:
        self._state = state
        self.icon.icon = self._images[state]
        self.icon.title = f"WhisperClaude – {LABELS[state]}"
        self.icon.update_menu()

    def stop(self) -> None:
        self.icon.stop()
