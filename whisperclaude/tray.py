"""System tray icon (pystray, own thread): colored by state, menu with status / open log / quit."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from whisperclaude.app import State

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
                 log_path: Path, config_path: Path, hotkey_label: str = "Left Ctrl") -> None:
        self._state = State.LOADING
        self.engine = ""  # "iGPU" / "CPU", set once the model is loaded
        self._labels = {State.LOADING: "Loading model…", State.IDLE: f"Ready (tap {hotkey_label})",
                        State.RECORDING: "Recording", State.TRANSCRIBING: "Transcribing"}
        self._images = {s: _image(c) for s, c in COLORS.items()}
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: f"Status: {self._status()}", None, enabled=False),
            pystray.MenuItem("Open config", lambda icon, item: os.startfile(config_path)),
            pystray.MenuItem("Restart (apply config)", lambda icon, item: on_restart()),
            pystray.MenuItem("Open log", lambda icon, item: os.startfile(log_path)),
            pystray.MenuItem("Quit", lambda icon, item: on_quit()),
        )
        self.icon = pystray.Icon("WhisperClaude", self._images[State.LOADING],
                                 f"WhisperClaude – {self._labels[State.LOADING]}", menu)

    def _status(self) -> str:
        label = self._labels[self._state]
        return f"{label} · {self.engine}" if self.engine else label

    def start(self) -> None:
        self.icon.run_detached()

    def set_state(self, state: State, msg: str = "") -> None:
        self._state = state
        self.icon.icon = self._images[state]
        self.icon.title = f"WhisperClaude – {self._status()}"
        self.icon.update_menu()

    def stop(self) -> None:
        self.icon.stop()
