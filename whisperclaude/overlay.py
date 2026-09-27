"""Floating status pill (Wispr-style): "● Recording 0:04" / "Transcribing…" / short messages.

The window never takes focus (WS_EX_NOACTIVATE) and is click-through (WS_EX_TRANSPARENT),
so the text box you dictate into keeps focus. Hidden = fully transparent, not withdrawn,
because re-showing a withdrawn window can activate it.
"""
from __future__ import annotations

import ctypes
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import font as tkfont

from whisperclaude.app import State

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
SPI_GETWORKAREA = 0x0030

KEY = "#ff00fe"  # transparent color key (corners outside the pill)
BG = "#202124"
FG = "#ffffff"
RED = "#ff4d4f"
AMBER = "#f5a623"
ALPHA = 0.92
MESSAGE_S = 2.5


def enable_dpi_awareness() -> None:
    """Crisp text on scaled displays. Must run before tk.Tk() is created."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except (AttributeError, OSError):
        pass


def _work_area() -> tuple[int, int, int, int]:
    """Primary monitor area without the taskbar: (left, top, right, bottom)."""
    rect = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    return rect.left, rect.top, rect.right, rect.bottom


class Overlay:
    """Uses the Tk root window itself as the pill. All methods must run on the Tk thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.withdraw()
        root.overrideredirect(True)
        root.configure(bg=KEY)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", KEY)
        root.attributes("-alpha", 0.0)
        self.scale = root.winfo_fpixels("1i") / 96
        self.font = tkfont.Font(family="Segoe UI", size=11)
        self.height = round(36 * self.scale)
        self.canvas = tk.Canvas(root, bg=KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        root.geometry("1x1+0+0")
        # The very first show activates the window (before our styles apply);
        # hand focus straight back to whatever had it.
        user32 = ctypes.windll.user32
        previous = user32.GetForegroundWindow()
        root.deiconify()
        root.update_idletasks()
        self._no_activate()
        root.update()
        if previous and user32.GetForegroundWindow() == user32.GetParent(root.winfo_id()):
            user32.SetForegroundWindow(previous)
        self._rec_started: float | None = None
        self._job: str | None = None

    def _no_activate(self) -> None:
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.root.winfo_id())
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE,
            style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT
            | WS_EX_LAYERED | WS_EX_TOPMOST,
        )

    # ---------- drawing ----------

    def _draw(self, text: str, dot: str | None = None) -> None:
        s = self.scale
        pad, dot_d, gap, h = round(16 * s), round(10 * s), round(8 * s), self.height
        w = pad * 2 + self.font.measure(text) + ((dot_d + gap) if dot else 0)
        c = self.canvas
        c.delete("all")
        c.create_oval(0, 0, h, h, fill=BG, outline="")
        c.create_oval(w - h, 0, w, h, fill=BG, outline="")
        c.create_rectangle(h // 2, 0, w - h // 2, h, fill=BG, outline="")
        x = pad
        if dot:
            c.create_oval(x, (h - dot_d) // 2, x + dot_d, (h + dot_d) // 2, fill=dot, outline="")
            x += dot_d + gap
        c.create_text(x, h // 2, text=text, fill=FG, font=self.font, anchor="w")

        left, _, right, bottom = _work_area()
        x0 = left + (right - left - w) // 2
        y0 = bottom - h - round(28 * s)
        self.root.geometry(f"{w}x{h}+{x0}+{y0}")
        self.root.attributes("-alpha", ALPHA)

    def _cancel_job(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None

    # ---------- public ----------

    def hide(self) -> None:
        self._cancel_job()
        self._rec_started = None
        self.root.attributes("-alpha", 0.0)

    def show_message(self, text: str, seconds: float | None = MESSAGE_S) -> None:
        """Show text; auto-hide after `seconds` (None = stay)."""
        self._cancel_job()
        self._rec_started = None
        self._draw(text)
        if seconds is not None:
            self._job = self.root.after(int(seconds * 1000), self.hide)

    def show_recording(self) -> None:
        self._cancel_job()
        self._rec_started = time.monotonic()
        self._tick()

    def _tick(self) -> None:
        if self._rec_started is None:
            return
        secs = int(time.monotonic() - self._rec_started)
        self._draw(f"Recording {secs // 60}:{secs % 60:02d}", dot=RED)
        self._job = self.root.after(250, self._tick)

    def show_transcribing(self) -> None:
        self._cancel_job()
        self._rec_started = None
        self._draw("Transcribing…", dot=AMBER)

    def set_state(self, state: State, msg: str = "") -> None:
        """App.on_state callback."""
        if state is State.RECORDING:
            self.show_recording()
        elif state is State.TRANSCRIBING:
            self.show_transcribing()
        elif msg:
            self.show_message(msg)
        else:
            self.hide()
