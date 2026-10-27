"""Tray icon images: one style per look, animated while something happens.

render(style, state, t, level) draws one 64x64 frame: `t` = seconds since the state began,
`level` = 0..1 microphone level (recording). Drawn 4x larger and scaled down, so edges stay
smooth at the real tray size (16–32 px). "classic" is the original static icon.

Every animation repeats exactly every CYCLE_S seconds, so a few cached frames cover it.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

SIZE, SS = 64, 4  # output size, supersampling factor
S = SIZE * SS
GREEN, RED, AMBER, GREY = (76, 175, 80), (255, 77, 79), (245, 166, 35), (154, 160, 166)
DARK, WHITE = (32, 33, 36), (255, 255, 255)
STATES = ("loading", "idle", "recording", "transcribing", "pasted")
COLOR = {"loading": GREY, "idle": GREEN, "recording": RED, "transcribing": AMBER, "pasted": GREEN}
STYLES = ("classic", "pulse", "bars", "ring")
CYCLE_S = 2.0
PI = math.pi
ANIMATED = {"loading", "recording", "transcribing", "pasted"}  # these need frames


def _p(v: float) -> int:  # 64-px coordinates -> supersampled pixels
    return round(v * SS)


def _alpha(rgb: tuple[int, int, int], a: float) -> tuple[int, int, int, int]:
    return (*rgb, max(0, min(255, round(a * 255))))


def _mic(d: ImageDraw.ImageDraw, color=WHITE, scale: float = 1.0, cx: float = 32) -> None:
    """Microphone glyph (capsule, holder, stand, base), centered around (cx, 32)."""
    def x(v: float) -> int:
        return _p(cx + (v - 32) * scale)

    def y(v: float) -> int:
        return _p(32 + (v - 32) * scale)

    w = max(1, _p(3 * scale))
    d.rounded_rectangle((x(25), y(12), x(39), y(38)), radius=_p(7 * scale), fill=color)
    d.arc((x(18), y(22), x(46), y(46)), start=0, end=180, fill=color, width=w)
    d.line((x(32), y(46), x(32), y(52)), fill=color, width=w)
    d.line((x(24), y(52), x(40), y(52)), fill=color, width=w)


def _check(d: ImageDraw.ImageDraw, p: float = 1.0) -> None:
    """White check mark, drawn up to fraction p (for a short "drawing" animation)."""
    pts = [(19, 33), (28, 42), (46, 23)]
    first = min(1.0, p * 2)
    a, b, c = pts
    mid = (a[0] + (b[0] - a[0]) * first, a[1] + (b[1] - a[1]) * first)
    w = _p(6)
    d.line((_p(a[0]), _p(a[1]), _p(mid[0]), _p(mid[1])), fill=WHITE, width=w)
    if p > 0.5:
        q = (p - 0.5) * 2
        end = (b[0] + (c[0] - b[0]) * q, b[1] + (c[1] - b[1]) * q)
        d.line((_p(b[0]), _p(b[1]), _p(end[0]), _p(end[1])), fill=WHITE, width=w)
        r = w // 2  # round joint
        d.ellipse((_p(b[0]) - r, _p(b[1]) - r, _p(b[0]) + r, _p(b[1]) + r), fill=WHITE)


def _lighter(rgb: tuple[int, int, int], p: float) -> tuple[int, int, int]:
    return tuple(round(c + (255 - c) * p) for c in rgb)  # type: ignore[return-value]


def _disc(d: ImageDraw.ImageDraw, color, r: float = 30, a: float = 1.0) -> None:
    d.ellipse((_p(32 - r), _p(32 - r), _p(32 + r), _p(32 + r)), fill=_alpha(color, a))


def _spinner(d: ImageDraw.ImageDraw, t: float, color=WHITE, r: float = 26, width: float = 4,
             speed: float = 1.0, length: float = 100) -> None:  # speed: turns per second
    start = (t * speed * 360) % 360
    d.arc((_p(32 - r), _p(32 - r), _p(32 + r), _p(32 + r)), start=start, end=start + length,
          fill=color, width=_p(width))


def _bars(d: ImageDraw.ImageDraw, heights: list[float], color=WHITE) -> None:
    """Vertical rounded bars (like the pill's level bars), heights 0..1."""
    n, bw, gap = len(heights), 6, 4
    x0 = 32 - (n * bw + (n - 1) * gap) / 2
    for i, h in enumerate(heights):
        half = 4 + 15 * max(0.0, min(1.0, h))
        x = x0 + i * (bw + gap)
        d.rounded_rectangle((_p(x), _p(32 - half), _p(x + bw), _p(32 + half)), radius=_p(3),
                            fill=color)


def render(style: str, state: str, t: float = 0.0, level: float = 0.0) -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = COLOR[state]
    if style == "classic" or (state == "idle" and style != "ring"):
        _disc(d, color)
        _mic(d)
    elif style == "pulse":
        if state == "recording":  # the circle glows brighter with the voice
            _disc(d, _lighter(color, 0.45 * level))
            _mic(d)
        elif state == "transcribing":  # a white arc runs around the rim
            _disc(d, color)
            _spinner(d, t, r=29, width=4)
            _mic(d, scale=0.8)
        elif state == "loading":
            _disc(d, color, a=0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * PI)))
            _mic(d)
        else:  # pasted
            _disc(d, color)
            _check(d, min(1.0, t / 0.25))
    elif style == "bars":
        _disc(d, color)
        if state == "recording":
            # middle bars react most, like a voice meter; a little idle wobble keeps it alive
            wob = [0.5 + 0.5 * math.sin(t * 3 * PI + i * 1.7) for i in range(4)]
            weights = (0.55, 1.0, 0.85, 0.5)
            _bars(d, [0.12 * wob[i] + level * weights[i] for i in range(4)])
        elif state == "transcribing":
            _bars(d, [0.15 + 0.6 * (0.5 + 0.5 * math.sin(t * 2 * PI - i * 0.9)) ** 2
                      for i in range(4)])
        elif state == "loading":
            _disc(d, color, a=0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * PI)))
            _mic(d)
        else:
            _check(d, min(1.0, t / 0.25))
    elif style == "ring":
        _disc(d, DARK, r=30)
        ring = (_p(2), _p(2), _p(62), _p(62))
        if state == "idle":
            d.ellipse(ring, outline=color, width=_p(6))
            _mic(d, scale=0.85)
        elif state == "recording":  # the ring gets thicker with the voice
            d.ellipse(ring, outline=color, width=_p(5 + 6 * level))
            _mic(d, color=color, scale=0.85)
        elif state in ("transcribing", "loading"):
            d.ellipse(ring, outline=_alpha(color, 0.3), width=_p(6))
            _spinner(d, t, color=color, r=30, width=6,
                     speed=1.0 if state == "transcribing" else 0.5)
            _mic(d, scale=0.85)
        else:
            d.ellipse(ring, outline=color, width=_p(6))
            _check(d, min(1.0, t / 0.25))
    return img.resize((SIZE, SIZE), Image.LANCZOS)
