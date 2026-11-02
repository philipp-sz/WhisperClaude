"""Anti-aliased, colored pill animations, drawn with numpy.

Tk's canvas draws without anti-aliasing, gradients or transparency, so the "gradient" and
"wave" pill styles are rendered here into an RGB array (a few hundred microseconds per
frame) and shown as one image. Pure functions, no Tk: (size, values, time) -> pixels.
"""
from __future__ import annotations

import math

import numpy as np

# Three color stops per state, left to right.
PALETTES = {
    "recording": ("#ff5a5f", "#f2459a", "#8b5cf6"),
    "transcribing": ("#ffc23d", "#ff7a45", "#ff4d8d"),
    "translating": ("#2fd5f0", "#4c8bf5", "#a35cf6"),
    "pasted": ("#5eea8a", "#2dd4bf", "#38bdf8"),
    "loading": ("#2fd5f0", "#8b5cf6", "#f2459a"),
    "info": ("#6aa8ff", "#7c83f7", "#a35cf6"),
    "error": ("#ffb347", "#ff6a55", "#f2457f"),
    "stopped": ("#a6abb2", "#7d828a", "#5f6368"),
}


def rgb(color: str) -> np.ndarray:
    """"#rrggbb" -> float RGB in 0..1."""
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255


def gradient(colors: tuple[str, ...], n: int) -> np.ndarray:
    """n colors evenly along the stops: shape (n, 3)."""
    stops = np.stack([rgb(c) for c in colors])
    u = np.linspace(0, len(colors) - 1, n)
    return np.stack([np.interp(u, np.arange(len(colors)), stops[:, ch]) for ch in range(3)],
                    axis=1).astype(np.float32)


def _to_image(pixels: np.ndarray) -> np.ndarray:
    return (np.clip(pixels, 0, 1) * 255 + 0.5).astype(np.uint8)


def bars(width: int, height: int, values: np.ndarray, bar_w: float, gap: float,
         min_h: float, max_h: float, colors: tuple[str, ...], bg: str) -> np.ndarray:
    """Rounded bars (values 0..1 -> min_h..max_h) with a left-to-right color gradient,
    brighter toward their middle. Returns (height, width, 3) uint8."""
    n = len(values)
    x = np.arange(width, dtype=np.float32) + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    pitch = bar_w + gap
    i = np.clip((x // pitch).astype(int), 0, n - 1)
    r = bar_w / 2
    half = np.maximum((min_h + (max_h - min_h) * np.asarray(values, np.float32)[i]) / 2 - r, 0)
    dy = np.maximum(np.abs(y - height / 2) - half, 0)
    d = np.sqrt((x - (i * pitch + r)) ** 2 + dy ** 2) - r # distance to the capsule
    cover = np.clip(0.5 - d, 0, 1)[..., None]
    shine = 1 + 0.35 * np.clip(1 - np.abs(y - height / 2) / (half + r), 0, 1)[..., None]
    color = np.minimum(gradient(colors, n)[i][None] * shine, 1)
    back = rgb(bg)
    return _to_image(back + (color - back) * cover)


# Wave layers, back to front: (cycles across the width, speed, phase, height share).
_LAYERS = ((1.3, 1.7, 0.0, 1.0), (1.9, -1.3, 2.3, 0.85), (2.6, 2.3, 4.4, 0.7))


def _smooth(envelope: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Envelope points -> a smooth curve over u (cosine interpolation, no corners)."""
    e = np.asarray(envelope, np.float32)
    pos = u * (len(e) - 1)
    i = np.minimum(pos.astype(int), len(e) - 2)
    f = (1 - np.cos(np.pi * (pos - i))) / 2
    return e[i] + (e[i + 1] - e[i]) * f


def waves(width: int, height: int, envelope: np.ndarray, t: float,
          colors: tuple[str, ...], bg: str, line: float = 1.0) -> np.ndarray:
    """Overlapping translucent waves, mirrored around a thin bright line, one color stop
    per layer. `envelope` (0..1, any length) sets the height along the width.
    Returns (height, width, 3) uint8."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    mid = height / 2
    reach = mid - 1.5
    e = np.pad(np.asarray(envelope, np.float32), 2, mode="edge")
    e = np.convolve(e, np.array([1, 2, 3, 2, 1], np.float32) / 9, "valid") # calm the jitter
    env = _smooth(e, u) ** 0.75 * np.sin(np.pi * u) ** 0.5 # taper to nothing at both ends
    out = np.broadcast_to(rgb(bg), (height, width, 3)).copy()
    dist = np.abs(y - mid)
    for (cycles, speed, phase, share), color in zip(_LAYERS, colors):
        wobble = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * cycles * u - speed * t + phase))
        a = reach * np.tanh(1.35 * env * wobble * share) # loud: rounded peaks, never clipped
        cover = np.clip(a + 0.5 - dist, 0, 1)[..., None]
        near = np.clip(1 - dist / (a + 1), 0, 1)[..., None] # 1 at the line, 0 at the tip
        c = rgb(color)
        out += (c + (1 - c) * 0.3 * near ** 2 - out) * cover * 0.82
    glow = np.clip(line / 2 + 0.5 - dist, 0, 1)
    shade = (0.35 + 0.65 * np.sin(np.pi * u) ** 0.8)[None, :]
    out += (1 - out) * (glow * shade)[..., None] * 0.9
    return _to_image(out)


def flowing(n: int, t: float, base: float = 0.3, swell: float = 0.6) -> np.ndarray:
    """Envelope for "working" states: a bump that runs through, left to right."""
    u = np.linspace(0, 1, n)
    run = (t * 0.6) % 1.0
    bumps = sum(np.exp(-((u - run - k) / 0.22) ** 2) for k in (-1, 0, 1)) # wraps around
    return base + swell * bumps + 0.08 * np.sin(2 * math.pi * u * 2 - t * 3)


def _segment_distance(px: np.ndarray, py: np.ndarray, a: tuple[float, float],
                      b: tuple[float, float]) -> np.ndarray:
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px - ax - t * dx, py - ay - t * dy)


# Symbols in units of the badge size: (polyline strokes, dots, stroke half-width).
GLYPHS = {
    "check": ([((0.29, 0.52), (0.44, 0.67), (0.72, 0.37))], [], 0.1),
    "bang": ([((0.5, 0.26), (0.5, 0.55))], [(0.5, 0.73)], 0.075),
    "info": ([((0.5, 0.46), (0.5, 0.74))], [(0.5, 0.28)], 0.075),
}
_STOP = (0.17, 0.05) # the "stop" symbol: a rounded square, half side and corner radius


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    c = np.arange(size, dtype=np.float32) + 0.5
    return c[None, :], c[:, None]


def _gradient_disk(size: int, grow: float, colors: tuple[str, ...],
                   bg: str) -> tuple[np.ndarray, np.ndarray, float]:
    """A disk with a diagonal gradient; (pixels, coverage, scale of the symbol)."""
    x, y = _grid(size)
    mid = size / 2
    scale = max(0.0, grow) / 1.1 # at grow 1 there is room for a 10 % overshoot
    disk = np.clip((mid - 0.5) * scale + 0.5 - np.hypot(x - mid, y - mid), 0, 1)[..., None]
    stops = np.stack([rgb(col) for col in colors])
    pos = np.clip((x + y) / (2 * size), 0, 1) * (len(colors) - 1)
    i = np.minimum(pos.astype(int), len(colors) - 2)
    fill = stops[i] + (stops[i + 1] - stops[i]) * (pos - i)[..., None]
    return rgb(bg) + (fill - rgb(bg)) * disk, disk, scale


def _glyph_distance(size: int, glyph: str, scale: float, draw: float) -> np.ndarray:
    """Distance to the symbol's strokes, of which `draw` (0..1) is drawn so far."""
    x, y = _grid(size)
    mid = size / 2
    to_px = lambda u, v: (mid + (u - 0.5) * size * scale, mid + (v - 0.5) * size * scale)
    strokes, dots, _ = GLYPHS[glyph]
    lines = [[to_px(*point) for point in stroke] for stroke in strokes]
    pieces = [(a, b) for line in lines for a, b in zip(line, line[1:])]
    left = draw * sum(math.dist(a, b) for a, b in pieces)
    dist = np.full((size, size), np.inf, np.float32)
    for a, b in pieces:
        if left <= 0:
            break
        f = min(1.0, left / max(math.dist(a, b), 1e-6))
        dist = np.minimum(dist, _segment_distance(x, y, a, (a[0] + (b[0] - a[0]) * f,
                                                            a[1] + (b[1] - a[1]) * f)))
        left -= math.dist(a, b)
    if draw >= 1: # dots appear once the strokes are complete
        for point in dots:
            px, py = to_px(*point)
            dist = np.minimum(dist, np.hypot(x - px, y - py))
    return dist


def badge(size: int, grow: float, draw: float, colors: tuple[str, ...], bg: str,
          glyph: str = "check") -> np.ndarray:
    """A round badge with a diagonal gradient and a white symbol (GLYPHS or "stop"). grow (0..1.1)
    scales it, draw (0..1) is how much of the symbol is drawn. (size, size, 3) uint8."""
    out, disk, scale = _gradient_disk(size, grow, colors, bg)
    if draw <= 0 or scale <= 0:
        return _to_image(out)
    if glyph == "stop": # fades in instead of being drawn
        x, y = _grid(size)
        half, corner = (v * size * scale for v in _STOP)
        qx, qy = np.abs(x - size / 2) - half + corner, np.abs(y - size / 2) - half + corner
        dist = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
        cover = np.clip(corner + 0.5 - dist, 0, 1) * draw
    else:
        width = GLYPHS[glyph][2] * size * scale
        cover = np.clip(width + 0.5 - _glyph_distance(size, glyph, scale, draw), 0, 1)
    out += (1 - out) * (cover[..., None] * disk)
    return _to_image(out)


def spinner(size: int, t: float, colors: tuple[str, ...], bg: str, grow: float = 1.0,
            track: str = "#3c4043") -> np.ndarray:
    """A ring with a gradient arc running around it (1 turn per second), the arc's tail
    fading out. (size, size, 3) uint8."""
    x, y = _grid(size)
    mid = size / 2
    scale = max(0.0, grow) / 1.1
    radius, half = (mid - 0.5) * scale * 0.78, max(0.8 * min(1.0, scale), size * 0.09 * scale)
    ring = np.clip(half + 0.5 - np.abs(np.hypot(x - mid, y - mid) - radius), 0, 1)[..., None]
    head = (t % 1.0) * 2 * math.pi
    behind = (head - np.arctan2(y - mid, x - mid)) % (2 * math.pi) # 0 at the head
    arc = np.clip(1 - behind / (1.4 * math.pi), 0, 1)[..., None] # the tail fades out
    color = gradient(colors, 64)[np.minimum((arc[..., 0] * 63).astype(int), 63)]
    out = rgb(bg) + (rgb(track) - rgb(bg)) * ring
    return _to_image(out + (color - out) * ring * arc ** 1.5)


def capsule(width: int, height: int, fill: str, border: str, border_w: float,
            bg: str) -> np.ndarray:
    """An anti-aliased pill-shaped button with a border. (height, width, 3) uint8."""
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    r = height / 2 - 0.5
    dx = np.maximum(np.abs(x - width / 2) - (width / 2 - 0.5 - r), 0)
    d = np.hypot(dx, y - height / 2) - r
    outer = np.clip(0.5 - d, 0, 1)[..., None]
    inner = np.clip(0.5 - d - border_w, 0, 1)[..., None]
    out = rgb(bg) + (rgb(border) - rgb(bg)) * outer
    return _to_image(out + (rgb(fill) - out) * inner)
