"""Floating status pill (Wispr-style) with animations.

- Recording: level bars (or waves) that follow the voice, timer; classic: a pulsing red dot.
- Transcribing: amber wave running through the bars. Translating: the same in blue.
- Styles: "classic" (plain canvas bars), "gradient" (bars in a color gradient) and "wave"
  (layered translucent waves); the last two are rendered by pillart.
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

import numpy as np
from PIL import Image, ImageTk

from whisperclaude import pillart
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
BLUE = "#4c8bf5"
BUTTON, BUTTON_HOVER = "#3c4043", "#5f6368"
# The Copy button of the wave and gradient styles: outlined, filled on hover.
OUTLINED, OUTLINED_HOVER = ("#2a2b2f", "#4a4d53"), ("#3c4043", "#6b6f76")  # (fill, border)
PASTED_S = 5.0  # how long the pill offers "Copy" after a paste
KEPT_S = 8.0  # ... and after a dictation that was kept, not pasted
ALPHA = 0.92
MESSAGE_S = 2.5
FRAME_MS = 33
FADE_MS = 150
N_BARS = 16
N_WAVE = 36  # level history the waves show (~1.2 s)
STYLES = ("wave", "gradient", "classic")
CLASSIC_COLORS = {"recording": RED, "transcribing": AMBER, "translating": BLUE}
# Badges of the wave and gradient styles: kind -> (palette, symbol); "loading" is a spinner.
BADGES = {"ok": ("pasted", "check"), "error": ("error", "bang"), "info": ("info", "info"),
          "stopped": ("stopped", "stop")}


def message_kind(text: str) -> str:
    """Badge for a message: "error", "ok" (e.g. "Copied") or "info"."""
    low = text.lower()
    if low.startswith(("error", "could not", "config error")) or "failed" in low:
        return "error"
    if low.startswith("cancelled"):
        return "stopped"
    return "ok" if low.startswith(("copied", "pasted")) else "info"


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


class LevelMeter:
    """Mic RMS -> 0..1 wave or bar height, on a dB scale with automatic gain.

    dB because loudness is perceived logarithmically. A fixed scale (-55 dB = flat, -20 dB =
    full) left quiet laptop mics nearly flat. The meter follows the loudest recent speech
    (peak: jumps up, falls 5 dB/s) and moves the top of the scale 75 % of the way from -20 dB
    down to it: speech peaking at -45 dB fills ~70 % instead of ~30 %, at -52 dB ~50 % instead
    of ~10 %, and loud speech still looks louder than quiet speech. It also follows the
    background noise (falls at once, rises 0.3 dB/s); the scale starts 6 dB above it and spans
    at least 15 dB, so silence stays flat. Call once per frame.
    """

    LOUD_DB = -20.0  # full height without any gain
    FOLLOW = 0.75  # how far the top of the scale follows a quieter peak
    MIN_RANGE_DB = 15.0

    def __init__(self, fps: float = 1000 / FRAME_MS) -> None:
        self.peak = -45.0  # assume a quiet mic: a loud one corrects it with its first word
        self.noise = -60.0
        self._fall, self._rise = 5.0 / fps, 0.3 / fps  # dB per frame
        self.unit = 0.0  # the latest result (the tray icon reads it from its own thread)

    def __call__(self, rms: float) -> float:
        db = 20 * math.log10(max(rms, 1e-6))
        self.peak = max(db, self.peak - self._fall)
        self.noise = db if db < self.noise else self.noise + self._rise
        floor = min(-45.0, max(-60.0, self.noise + 6))
        top = self.LOUD_DB + self.FOLLOW * (min(self.peak, self.LOUD_DB) - self.LOUD_DB)
        top = max(top, floor + self.MIN_RANGE_DB)
        self.unit = min(1.0, max(0.0, (db - floor) / (top - floor)))
        return self.unit


def _ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def _ease_out_back(p: float) -> float:
    """Overshoots a little before it settles: a "pop"."""
    return 1 + 2.70158 * (p - 1) ** 3 + 1.70158 * (p - 1) ** 2


def _mix(c1: str, c2: str, p: float) -> str:
    """Blend two #rrggbb colors (Tk text has no alpha, so this is how text fades)."""
    p = min(1.0, max(0.0, p))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * p):02x}" for x, y in zip(a, b))


class Overlay:
    """Uses the Tk root window itself as the pill. All methods must run on the Tk thread."""

    def __init__(self, root: tk.Tk, style: str = "wave") -> None:
        self.root = root
        self.style = style
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
        self.art_w = round(96 * s) if style == "wave" else self.bars_w
        self.art_h = self.h - 2 * round(4 * s)
        self.badge_size = round(20 * s)
        self._badge: ImageTk.PhotoImage | None = None  # the start pill's spinner, then check
        self._images: list[ImageTk.PhotoImage] = []  # shown images; Tk keeps no reference
        self.canvas = tk.Canvas(root, bg=KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._bg = [self.canvas.create_oval(0, 0, 0, 0, fill=BG, outline=""),
                    self.canvas.create_oval(0, 0, 0, 0, fill=BG, outline=""),
                    self.canvas.create_rectangle(0, 0, 0, 0, fill=BG, outline="")]

        self.level_fn: Callable[[], float] = lambda: 0.0  # set to the recorder's level
        self.meter = LevelMeter()  # new one per recording; .unit is the shown level
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

    def _set_clickable(self, clickable: bool) -> None:
        """Normally the pill is click-through (WS_EX_TRANSPARENT). While it offers "Copy" it
        takes clicks; WS_EX_NOACTIVATE still keeps the focus in the user's app."""
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
        new = style & ~WS_EX_TRANSPARENT if clickable else style | WS_EX_TRANSPARENT
        if new != style:
            user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, new)

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
        self._dot = self._label = self._badge = None
        self._images = []
        for event in ("<Button-1>", "<Enter>", "<Leave>"):
            self.canvas.unbind(event)
        self.canvas.configure(cursor="")
        if self._hwnd:
            self._set_clickable(False)

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

    def _make_art(self, x: float) -> ImageTk.PhotoImage:
        """The image the gradient and wave styles are drawn into, at x, centered."""
        photo = ImageTk.PhotoImage("RGB", (self.art_w, self.art_h))
        self._images.append(photo)
        self.canvas.create_image(x, (self.h - self.art_h) // 2, image=photo, anchor="nw",
                                 tags="content")
        return photo

    def _draw_art(self, photo: ImageTk.PhotoImage, values: np.ndarray, t: float,
                  kind: str) -> None:
        palette = pillart.PALETTES[kind]
        if self.style == "wave":
            pixels = pillart.waves(self.art_w, self.art_h, values, t, palette, BG,
                                   line=max(1.0, 1.2 * self.scale))
        else:
            pixels = pillart.bars(self.art_w, self.art_h, values, self.bar_w, self.bar_gap,
                                  self.bar_min, self.bar_max, palette, BG)
        photo.paste(Image.fromarray(pixels))

    def _text_x(self) -> float:
        """Where the text starts in a pill with a dot (classic) or a badge in front."""
        if self.style == "classic":
            return self.pad + self.dot_d + self.gap
        return self.pad - round(3 * self.scale) + self.badge_size + round(8 * self.scale)

    def _make_badge(self) -> ImageTk.PhotoImage:
        size = self.badge_size
        photo = ImageTk.PhotoImage("RGB", (size, size))
        self._images.append(photo)
        self.canvas.create_image(self.pad - round(3 * self.scale), (self.h - size) // 2,
                                 image=photo, anchor="nw", tags="content")
        return photo

    def _paint_badge(self, photo: ImageTk.PhotoImage, kind: str, grow: float = 1.0,
                     draw: float = 1.0, t: float = 0.0) -> None:
        if kind == "loading":
            pixels = pillart.spinner(self.badge_size, t, pillart.PALETTES["loading"], BG, grow)
        else:
            palette, glyph = BADGES[kind]
            pixels = pillart.badge(self.badge_size, grow, draw, pillart.PALETTES[palette], BG,
                                   glyph)
        photo.paste(Image.fromarray(pixels))

    @staticmethod
    def _pop(t: float) -> tuple[float, float]:
        """(grow, draw) of a badge popping in: overshoots, then its symbol draws itself."""
        return _ease_out_back(min(1.0, t / 0.3)), min(1.0, max(0.0, (t - 0.15) / 0.3))

    def _pop_badge(self, photo: ImageTk.PhotoImage, kind: str) -> None:
        def frame(t: float) -> bool | None:
            self._paint_badge(photo, kind, *self._pop(t))
            return False if t >= 0.45 else None

        self._animate(frame)

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
        """Show text; auto-hide after `seconds` (None = stay). The colored styles put a badge
        in front: "!" for errors, a check for "Copied", "i" otherwise."""
        self._clear()
        if self.style == "classic":
            self._text(self.pad, text)
            self._set_width(self.pad * 2 + self.font.measure(text))
        else:
            badge = self._make_badge()
            self._text(self._text_x(), text)
            self._set_width(self._text_x() + self.font.measure(text) + self.pad)
            self._pop_badge(badge, message_kind(text))
        self._fade(ALPHA)
        if seconds is not None:
            self._after("hide", int(seconds * 1000), self.hide)

    def show_pasted(self, on_copy: Callable[[], None], seconds: float = PASTED_S,
                    label: str = "Pasted") -> None:
        """After a paste: "✓ Pasted   ⧉ Copy" for a few seconds. A click copies the text
        again, for when the paste went nowhere (no text field had focus). With another label
        (a kept dictation: "Cancelled", "Not pasted …") it offers the text the same way.
        Hovering keeps the pill open; clicking never takes the focus from the app below."""
        self._clear()
        s, h = self.scale, self.h
        c = self.canvas
        art = self.style != "classic"
        tx = self._text_x()
        kind = message_kind(label)
        if art:  # a gradient badge whose symbol draws itself
            badge = self._make_badge()
        else:
            dot = self._make_dot(GREEN if kind == "ok" else GREY)
            self._place_dot(dot, self.dot_d / 2)
        self._text(tx, label)
        # the "Copy" button: a smaller pill with a two-sheets icon
        bx0 = tx + self.font.measure(label) + 2 * self.gap
        inset, bpad, icon = round(6 * s), round(10 * s), round(11 * s)
        bh = h - 2 * inset
        lx = bx0 + bpad + icon + round(6 * s)
        bx1 = lx + self.font.measure("Copy") + bpad
        if art:
            fill, hover = OUTLINED[0], OUTLINED_HOVER[0]
            looks = [ImageTk.PhotoImage(Image.fromarray(pillart.capsule(
                round(bx1 - bx0), bh, f, b, max(1.0, s), BG))) for f, b in (OUTLINED, OUTLINED_HOVER)]
            self._images += looks
            c.create_image(bx0, inset, image=looks[0], anchor="nw", tags=("content", "btnimg"))
        else:
            fill, hover = BUTTON, BUTTON_HOVER
            c.create_oval(bx0, inset, bx0 + bh, h - inset, fill=fill, outline="", tags=("content", "btn"))
            c.create_oval(bx1 - bh, inset, bx1, h - inset, fill=fill, outline="", tags=("content", "btn"))
            c.create_rectangle(bx0 + bh / 2, inset, bx1 - bh / 2, h - inset, fill=fill, outline="",
                               tags=("content", "btn"))
        ix, iy, sheet, off = bx0 + bpad, h / 2 - icon / 2, icon * 0.72, icon * 0.28
        width = max(1.0, 1.4 * s)
        c.create_rectangle(ix + off, iy, ix + off + sheet, iy + sheet, outline=FG, width=width,
                           tags="content")
        c.create_rectangle(ix, iy + off, ix + sheet, iy + off + sheet, outline=FG, width=width,
                           fill=fill, tags=("content", "btn"))
        self._text(lx, "Copy")
        self._set_width(bx1 + self.pad - inset)

        def click(_event: object) -> None:
            on_copy()

        def enter(_event: object) -> None:
            c.itemconfig("btn", fill=hover)
            if art:
                c.itemconfig("btnimg", image=looks[1])
            self._cancel("hide")

        def leave(_event: object) -> None:
            c.itemconfig("btn", fill=fill)
            if art:
                c.itemconfig("btnimg", image=looks[0])
            self._after("hide", 1500, self.hide)

        if art:
            self._pop_badge(badge, kind)

        c.bind("<Button-1>", click)
        c.bind("<Enter>", enter)
        c.bind("<Leave>", leave)
        c.configure(cursor="hand2")
        self._set_clickable(True)
        self._fade(ALPHA)
        self._after("hide", int(seconds * 1000), self.hide)

    def show_recording(self) -> None:
        self._clear()
        classic = self.style == "classic"
        # classic: a pulsing red dot; the colored styles show only the waves (or bars)
        dot = self._make_dot(RED) if classic else None
        bx = self.pad + self.dot_d + self.gap if classic else self.pad
        bars = self._make_bars(RED) if classic else []
        photo = None if classic else self._make_art(bx)
        tx = bx + (self.bars_w if classic else self.art_w) + self.gap
        timer = self._text(tx, "0:00")
        self._set_width(tx + self.font.measure("00:00") + self.pad)
        n = N_WAVE if self.style == "wave" else N_BARS
        history = deque([0.0] * n, maxlen=n)
        smooth = [0.0]
        meter = self.meter = LevelMeter()

        def frame(t: float) -> None:
            target = meter(self.level_fn())
            # fast attack, slow release: bars jump up with the voice and settle gently
            smooth[0] += (target - smooth[0]) * (0.55 if target > smooth[0] else 0.18)
            history.append(smooth[0])
            if photo is not None:
                self._draw_art(photo, np.fromiter(history, np.float32, n), t, "recording")
            for i, (item, v) in enumerate(zip(bars, history)):
                self._set_bar(item, i, bx, v)
            secs = int(t)
            self.canvas.itemconfig(timer, text=f"{secs // 60}:{secs % 60:02d}")
            if dot is not None:
                self._place_dot(dot, self.dot_d / 2 * (0.8 + 0.2 * math.sin(t * 4)))

        self._animate(frame)
        self._fade(ALPHA)

    def show_transcribing(self, label: str = "Transcribing…", kind: str = "transcribing") -> None:
        """kind: "transcribing" (amber) or "translating" (blue)."""
        self._clear()
        bx = self.pad
        classic = self.style == "classic"
        bars = self._make_bars(CLASSIC_COLORS[kind]) if classic else []
        photo = None if classic else self._make_art(bx)
        tx = bx + (self.bars_w if classic else self.art_w) + self.gap
        self._text(tx, label)
        self._set_width(tx + self.font.measure(label) + self.pad)

        def frame(t: float) -> None:
            values = [0.5 + 0.5 * math.sin(t * 7 - i * 0.55) for i in range(N_BARS)]
            values = [0.1 + 0.6 * v * v for v in values]
            if photo is not None:
                art = pillart.flowing(24, t) if self.style == "wave" else np.array(values)
                self._draw_art(photo, art, t, kind)
            for i, item in enumerate(bars):
                self._set_bar(item, i, bx, values[i])

        self._animate(frame)
        self._fade(ALPHA)

    def show_starting(self, text: str = "Loading model…") -> None:
        """Start animation: circle grows into the pill, grey dot pulses (colored styles: a
        spinner turns) until show_ready()."""
        self._clear()
        art = self.style != "classic"
        dot = None if art else self._make_dot(GREY)
        self._dot, badge = dot, self._make_badge() if art else None
        self._badge = badge
        tx = self._text_x()
        self._label = label = self._text(tx, text, BG)
        self._start_target = tx + self.font.measure(text) + self.pad
        self._set_width(self.h)
        self._fade(ALPHA, ms=200)

        def frame(t: float) -> None:
            grow = _ease_out(min(1.0, t / 0.45))
            w = self.h + (self._start_target - self.h) * grow
            self._set_width(w if grow < 1 else self._w + (self._start_target - self._w) * 0.3)
            self.canvas.itemconfig(label, fill=_mix(BG, FG, (t - 0.3) / 0.25))
            if badge is not None:
                self._paint_badge(badge, "loading", _ease_out_back(min(1.0, t / 0.35)), t=t)
            else:
                self._place_dot(dot, self.dot_d / 2 * (0.55 + 0.45 * abs(math.sin(t * 3))))

        self._animate(frame)

    def set_starting_text(self, text: str) -> None:
        """Change the start-animation text (e.g. "Preparing GPU…"); the pill glides to fit."""
        if self._label is None:
            return
        self.canvas.itemconfig(self._label, text=text)
        self._start_target = self._text_x() + self.font.measure(text) + self.pad

    def show_ready(self, text: str, seconds: float = MESSAGE_S) -> None:
        """Second half of the start animation: dot turns green with a pop (colored styles: the
        spinner becomes a check badge), pill resizes."""
        if (self._dot is None and self._badge is None) or self._label is None:
            self.show_message(text, seconds)
            return
        dot, badge, label = self._dot, self._badge, self._label
        if dot is not None:
            self.canvas.itemconfig(dot, fill=GREEN)
        self.canvas.itemconfig(label, text=text, fill=FG)
        w0 = self._w
        w1 = self._text_x() + self.font.measure(text) + self.pad

        def frame(t: float) -> bool | None:
            p = min(1.0, t / 0.3)
            self._set_width(w0 + (w1 - w0) * _ease_out(p))
            if badge is not None:
                self._paint_badge(badge, "ok", *self._pop(t))
                return False if t >= 0.45 else None
            pop = 1 + 0.5 * math.sin(math.pi * p)  # grows to 1.5x and back
            self._place_dot(dot, self.dot_d / 2 * pop)
            return False if p >= 1 else None

        self._animate(frame)
        self._after("hide", int(seconds * 1000), self.hide)

    def show_goodbye(self, text: str, on_done: Callable[[], None],
                     restart: bool = False) -> None:
        """Stop animation: text fades, pill shrinks into a circle and fades out. Colored
        styles: a spinner (restart) or a stop badge shrinks with it."""
        self._clear()
        art = self.style != "classic"
        dot = None if art else self._make_dot(GREY)
        badge = self._make_badge() if art else None
        kind = "loading" if restart else "stopped"
        tx = self._text_x()
        label = self._text(tx, text)
        w0 = tx + self.font.measure(text) + self.pad
        self._set_width(w0)
        if dot is not None:
            self._place_dot(dot, self.dot_d / 2)
        self._fade(ALPHA, ms=100)

        def frame(t: float) -> bool | None:
            if badge is not None:
                grow = 1.0 if t < 0.6 else max(0.0, 1 - (t - 0.6) / 0.35)
                self._paint_badge(badge, kind, grow, t=t)
            if t < 0.6:  # hold so the text can be read
                return None
            p = min(1.0, (t - 0.6) / 0.35)
            self.canvas.itemconfig(label, fill=_mix(FG, BG, p * 2))
            self._set_width(w0 + (self.h - w0) * p * p)
            if dot is not None:
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
        elif state is State.TRANSLATING:
            self.show_transcribing("Translating…", "translating")
        elif msg:  # longer messages stay longer (~13 characters per second, at least 2.5 s)
            self.show_message(msg, max(MESSAGE_S, len(msg) * 0.075))
        else:
            self.hide()
