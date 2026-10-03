"""Floating status pill (Wispr-style) with animations.

- Recording: pulsing red dot, level bars that follow the voice, timer.
- Transcribing: amber wave running through the bars.
- Start: pill grows out of a circle (grey dot pulses while the model loads), turns green.
- Stop/restart: text fades, pill shrinks back into a circle and fades out.

The window never takes focus (WS_EX_NOACTIVATE) and is click-through (WS_EX_TRANSPARENT).
Hidden = fully transparent, not withdrawn (re-showing a withdrawn window can activate it).
Frames (~30 fps) only run while something animates, so an idle app costs no CPU.
"""
from __future__ import annotations

import ctypes
import math
import time
import tkinter as tk
from collections import deque
from ctypes import wintypes
from tkinter import font as tkfont
from typing import Callable

from whisperclaude.app import State

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
SPI_GETWORKAREA = 0x0030
HWND_TOPMOST = ctypes.c_void_p(-1)
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, SWP_SHOWWINDOW = 0x0001, 0x0002, 0x0010, 0x0040

KEY = "#ff00fe"  # transparent color key (corners outside the pill)
BG = "#202124"
FG = "#ffffff"
GREY = "#9aa0a6"
GREEN = "#4caf50"
RED = "#ff4d4f"
AMBER = "#f5a623"
ALPHA = 0.92
MESSAGE_S = 2.5
FRAME_MS = 33
FADE_MS = 150
N_BARS = 16


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


def level_to_unit(rms: float) -> float:
    """Mic RMS -> 0..1 bar height on a dB scale (-55 dB = silent, -20 dB = full).

    dB because loudness is perceived logarithmically; the range suits quiet laptop mics.
    """
    db = 20 * math.log10(max(rms, 1e-6))
    return min(1.0, max(0.0, (db + 55) / 35))


def _ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def _mix(c1: str, c2: str, p: float) -> str:
    """Blend two #rrggbb colors (Tk text has no alpha, so this is how text fades)."""
    p = min(1.0, max(0.0, p))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * p):02x}" for x, y in zip(a, b))


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
        s = self.scale = root.winfo_fpixels("1i") / 96
        self.font = tkfont.Font(family="Segoe UI", size=11)
        self.h = round(36 * s)
        self.pad, self.gap, self.dot_d = round(16 * s), round(10 * s), round(10 * s)
        self.bar_w, self.bar_gap = max(2, round(3 * s)), max(1, round(2 * s))
        self.bar_min, self.bar_max = max(2, round(4 * s)), round(20 * s)
        self.bars_w = N_BARS * (self.bar_w + self.bar_gap) - self.bar_gap
        self.canvas = tk.Canvas(root, bg=KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._bg = [self.canvas.create_oval(0, 0, 0, 0, fill=BG, outline=""),
                    self.canvas.create_oval(0, 0, 0, 0, fill=BG, outline=""),
                    self.canvas.create_rectangle(0, 0, 0, 0, fill=BG, outline="")]

        self.level_fn: Callable[[], float] = lambda: 0.0  # set to the recorder's level
        self._jobs: dict[str, str] = {}
        self._frame_fn: Callable[[float], bool | None] | None = None
        self._t0 = 0.0
        self._alpha = 0.0
        self._hwnd = 0  # real top-level window handle, set by _no_activate()
        self._w = self.h
        self._dot: int | None = None
        self._label: int | None = None
        self._start_target = float(self.h)

        # The very first show activates the window (before our styles apply);
        # hand focus straight back to whatever had it.
        root.geometry("1x1+0+0")
        user32 = ctypes.windll.user32
        previous = user32.GetForegroundWindow()
        root.deiconify()
        root.update_idletasks()
        self._no_activate()
        root.update()
        if previous and user32.GetForegroundWindow() == user32.GetParent(root.winfo_id()):
            user32.SetForegroundWindow(previous)

    def _raise_topmost(self) -> None:
        """Put the pill back on top of every other window, without activating it.

        Tk's "-topmost" is only applied once. After a long sleep (seen: 14 h) Windows had
        moved the window below other windows while it still carried the topmost flag: alive,
        visible, correct alpha and position, but not on screen. Re-asserting the topmost
        position every time the pill is shown fixes that.
        """
        ctypes.windll.user32.SetWindowPos(
            self._hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW)

    def _no_activate(self) -> None:
        user32 = ctypes.windll.user32
        hwnd = self._hwnd = user32.GetParent(self.root.winfo_id())
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE,
            style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT
            | WS_EX_LAYERED | WS_EX_TOPMOST,
        )

    # ---------- scheduling ----------

    def _after(self, name: str, ms: int, fn: Callable[[], None]) -> None:
        self._cancel(name)
        self._jobs[name] = self.root.after(ms, fn)

    def _cancel(self, *names: str) -> None:
        for name in names or tuple(self._jobs):
            job = self._jobs.pop(name, None)
            if job is not None:
                self.root.after_cancel(job)

    def _animate(self, fn: Callable[[float], bool | None]) -> None:
        """Run fn(t_seconds) every frame until it returns False."""
        self._frame_fn = fn
        self._t0 = time.monotonic()
        self._frame()

    def _frame(self) -> None:
        fn = self._frame_fn
        if fn is None:
            return
        if fn(time.monotonic() - self._t0) is False:
            self._frame_fn = None
            return
        self._after("frame", FRAME_MS, self._frame)

    def _fade(self, target: float, ms: int = FADE_MS,
              then: Callable[[], None] | None = None) -> None:
        self._cancel("fade")
        if target > 0:  # about to be shown
            self._raise_topmost()
        start, t0 = self._alpha, time.monotonic()

        def step() -> None:
            p = min(1.0, (time.monotonic() - t0) * 1000 / ms) if ms else 1.0
            self._alpha = start + (target - start) * p
            self.root.attributes("-alpha", self._alpha)
            if p < 1:
                self._after("fade", FRAME_MS, step)
            elif then:
                then()

        step()

    # ---------- drawing ----------

    def _set_width(self, w: float) -> None:
        """Resize the pill (and window), centered above the taskbar."""
        h, w = self.h, max(self.h, int(w))
        self._w = w
        left, _, right, bottom = _work_area()
        x0, y0 = left + (right - left - w) // 2, bottom - h - round(28 * self.scale)
        self.root.geometry(f"{w}x{h}+{x0}+{y0}")
        c = self.canvas
        c.coords(self._bg[0], 0, 0, h, h)
        c.coords(self._bg[1], w - h, 0, w, h)
        c.coords(self._bg[2], h // 2, 0, w - h // 2, h)

    def _clear(self) -> None:
        self._cancel("frame", "hide")
        self._frame_fn = None
        self.canvas.delete("content")
        self._dot = self._label = None

    def _text(self, x: float, text: str, color: str = FG) -> int:
        return self.canvas.create_text(x, self.h // 2, text=text, fill=color, font=self.font,
                                       anchor="w", tags="content")

    def _make_dot(self, color: str) -> int:
        return self.canvas.create_oval(0, 0, 0, 0, fill=color, outline="", tags="content")

    def _place_dot(self, item: int, radius: float) -> None:
        cx, cy = self.pad + self.dot_d / 2, self.h / 2
        self.canvas.coords(item, cx - radius, cy - radius, cx + radius, cy + radius)

    def _make_bars(self, color: str) -> list[int]:
        return [self.canvas.create_line(0, 0, 0, 0, fill=color, width=self.bar_w,
                                        capstyle=tk.ROUND, tags="content")
                for _ in range(N_BARS)]

    def _set_bar(self, item: int, i: int, x0: float, v: float) -> None:
        x = x0 + i * (self.bar_w + self.bar_gap) + self.bar_w / 2
        half = (self.bar_min + (self.bar_max - self.bar_min) * v) / 2 - self.bar_w / 2
        half = max(half, 0.5)  # round caps add bar_w/2 at each end
        self.canvas.coords(item, x, self.h / 2 - half, x, self.h / 2 + half)

    # ---------- states ----------

    def hide(self) -> None:
        self._cancel("hide")
        self._fade(0.0, then=self._clear)

    def show_message(self, text: str, seconds: float | None = MESSAGE_S) -> None:
        """Show text; auto-hide after `seconds` (None = stay)."""
        self._clear()
        self._text(self.pad, text)
        self._set_width(self.pad * 2 + self.font.measure(text))
        self._fade(ALPHA)
        if seconds is not None:
            self._after("hide", int(seconds * 1000), self.hide)

    def show_recording(self) -> None:
        self._clear()
        dot = self._make_dot(RED)
        bx = self.pad + self.dot_d + self.gap
        bars = self._make_bars(RED)
        tx = bx + self.bars_w + self.gap
        timer = self._text(tx, "0:00")
        self._set_width(tx + self.font.measure("00:00") + self.pad)
        history = deque([0.0] * N_BARS, maxlen=N_BARS)
        smooth = [0.0]

        def frame(t: float) -> None:
            target = level_to_unit(self.level_fn())
            # fast attack, slow release: bars jump up with the voice and settle gently
            smooth[0] += (target - smooth[0]) * (0.55 if target > smooth[0] else 0.18)
            history.append(smooth[0])
            for i, (item, v) in enumerate(zip(bars, history)):
                self._set_bar(item, i, bx, v)
            secs = int(t)
            self.canvas.itemconfig(timer, text=f"{secs // 60}:{secs % 60:02d}")
            self._place_dot(dot, self.dot_d / 2 * (0.8 + 0.2 * math.sin(t * 4)))

        self._animate(frame)
        self._fade(ALPHA)

    def show_transcribing(self) -> None:
        self._clear()
        bx = self.pad
        bars = self._make_bars(AMBER)
        label = "Transcribing…"
        tx = bx + self.bars_w + self.gap
        self._text(tx, label)
        self._set_width(tx + self.font.measure(label) + self.pad)

        def frame(t: float) -> None:
            for i, item in enumerate(bars):
                v = 0.5 + 0.5 * math.sin(t * 7 - i * 0.55)
                self._set_bar(item, i, bx, 0.1 + 0.6 * v * v)

        self._animate(frame)
        self._fade(ALPHA)

    def show_starting(self, text: str = "Loading model…") -> None:
        """Start animation: circle grows into the pill, grey dot pulses until show_ready()."""
        self._clear()
        self._dot = dot = self._make_dot(GREY)
        tx = self.pad + self.dot_d + self.gap
        self._label = label = self._text(tx, text, BG)
        self._start_target = tx + self.font.measure(text) + self.pad
        self._set_width(self.h)
        self._fade(ALPHA, ms=200)

        def frame(t: float) -> None:
            grow = _ease_out(min(1.0, t / 0.45))
            w = self.h + (self._start_target - self.h) * grow
            self._set_width(w if grow < 1 else self._w + (self._start_target - self._w) * 0.3)
            self.canvas.itemconfig(label, fill=_mix(BG, FG, (t - 0.3) / 0.25))
            self._place_dot(dot, self.dot_d / 2 * (0.55 + 0.45 * abs(math.sin(t * 3))))

        self._animate(frame)

    def set_starting_text(self, text: str) -> None:
        """Change the start-animation text (e.g. "Preparing GPU…"); the pill glides to fit."""
        if self._label is None:
            return
        self.canvas.itemconfig(self._label, text=text)
        self._start_target = self.pad + self.dot_d + self.gap + self.font.measure(text) + self.pad

    def show_ready(self, text: str, seconds: float = MESSAGE_S) -> None:
        """Second half of the start animation: dot turns green with a pop, pill resizes."""
        if self._dot is None or self._label is None:
            self.show_message(text, seconds)
            return
        dot, label = self._dot, self._label
        self.canvas.itemconfig(dot, fill=GREEN)
        self.canvas.itemconfig(label, text=text, fill=FG)
        w0 = self._w
        w1 = self.pad + self.dot_d + self.gap + self.font.measure(text) + self.pad

        def frame(t: float) -> bool | None:
            p = min(1.0, t / 0.3)
            self._set_width(w0 + (w1 - w0) * _ease_out(p))
            pop = 1 + 0.5 * math.sin(math.pi * p)  # grows to 1.5x and back
            self._place_dot(dot, self.dot_d / 2 * pop)
            return False if p >= 1 else None

        self._animate(frame)
        self._after("hide", int(seconds * 1000), self.hide)

    def show_goodbye(self, text: str, on_done: Callable[[], None]) -> None:
        """Stop animation: text fades, pill shrinks into a circle and fades out."""
        self._clear()
        dot = self._make_dot(GREY)
        tx = self.pad + self.dot_d + self.gap
        label = self._text(tx, text)
        w0 = tx + self.font.measure(text) + self.pad
        self._set_width(w0)
        self._place_dot(dot, self.dot_d / 2)
        self._fade(ALPHA, ms=100)

        def frame(t: float) -> bool | None:
            if t < 0.6:  # hold so the text can be read
                return None
            p = min(1.0, (t - 0.6) / 0.35)
            self.canvas.itemconfig(label, fill=_mix(FG, BG, p * 2))
            self._set_width(w0 + (self.h - w0) * p * p)
            self._place_dot(dot, self.dot_d / 2 * (1 - p))
            if p >= 1:
                self._fade(0.0, then=on_done)
                return False
            return None

        self._animate(frame)

    def set_state(self, state: State, msg: str = "") -> None:
        """App.on_state callback."""
        if state is State.LOADING:
            if msg:  # a tap while the model loads: the start pill says what's going on
                self.set_starting_text(msg)
        elif state is State.RECORDING:
            self.show_recording()
        elif state is State.TRANSCRIBING:
            self.show_transcribing()
        elif msg:
            self.show_message(msg)
        else:
            self.hide()
