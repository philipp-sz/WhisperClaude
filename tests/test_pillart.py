"""Pill animations (pure numpy, no Tk)."""
import numpy as np
import pytest

from whisperclaude import pillart

BG = "#202124"


def test_bars_follow_the_values_and_keep_the_background():
    quiet = pillart.bars(78, 28, np.zeros(16), 3, 2, 4, 20, pillart.PALETTES["recording"], BG)
    loud = pillart.bars(78, 28, np.ones(16), 3, 2, 4, 20, pillart.PALETTES["recording"], BG)
    assert quiet.shape == (28, 78, 3) and quiet.dtype == np.uint8
    assert (quiet[0, 0] == [0x20, 0x21, 0x24]).all() # nothing drawn outside the bars
    lit = lambda img: (img.max(axis=2) > 0x60).sum()
    assert lit(loud) > 3 * lit(quiet)
    # left to right: red end of the gradient on the left, violet on the right
    assert loud[14, 1, 0] > loud[14, 1, 2] and loud[14, 76, 2] > loud[14, 76, 1]


def test_waves_grow_with_the_level_and_taper_at_the_ends():
    palette = pillart.PALETTES["recording"]
    silent = pillart.waves(96, 28, np.zeros(36), 0.0, palette, BG)
    speech = pillart.waves(96, 28, np.full(36, 0.8), 0.0, palette, BG)
    colored = lambda img: (np.ptp(img.astype(int), axis=2) > 40).sum()
    assert colored(silent) == 0 # only the thin grey-white center line
    assert colored(speech) > 500
    assert colored(speech[:, :1]) * 4 < colored(speech[:, 48:49]) # tapered at the edges
    assert (speech[0, :] == [0x20, 0x21, 0x24]).all() # stays inside the image


def _waves_full(width, height, envelope, t, colors, bg, line=1.0):
    """Reference: the waves rendered row by row over the whole height (before mirroring)."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    mid = height / 2
    reach = mid - 1.5
    e = np.pad(np.asarray(envelope, np.float32), 2, mode="edge")
    e = np.convolve(e, np.array([1, 2, 3, 2, 1], np.float32) / 9, "valid")
    env = pillart._smooth(e, u) ** 0.75 * np.sin(np.pi * u) ** 0.5
    out = np.broadcast_to(pillart.rgb(bg), (height, width, 3)).copy()
    dist = np.abs(y - mid)
    for (cycles, speed, phase, share), color in zip(pillart._LAYERS, colors):
        wobble = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * cycles * u - speed * t + phase))
        a = reach * np.tanh(1.35 * env * wobble * share)
        cover = np.clip(a + 0.5 - dist, 0, 1)[..., None]
        near = np.clip(1 - dist / (a + 1), 0, 1)[..., None]
        c = pillart.rgb(color)
        out += (c + (1 - c) * 0.3 * near ** 2 - out) * cover * 0.82
    glow = np.clip(line / 2 + 0.5 - dist, 0, 1)
    shade = (0.35 + 0.65 * np.sin(np.pi * u) ** 0.8)[None, :]
    out += (1 - out) * (glow * shade)[..., None] * 0.9
    return pillart._to_image(out)


@pytest.mark.parametrize("height", [55, 56])
def test_waves_mirrored_half_is_exact(height):
    rng = np.random.default_rng(0)
    palette = pillart.PALETTES["recording"]
    for level in (0.0, 0.3, 1.0, 1.4):
        for t in (0.0, 0.37, 2.5):
            envelope = level * rng.random(int(rng.integers(3, 40)))
            fast = pillart.waves(192, height, envelope, t, palette, BG)
            assert np.array_equal(fast, _waves_full(192, height, envelope, t, palette, BG))


def test_working_envelope_moves():
    a, b = pillart.flowing(24, 0.0), pillart.flowing(24, 0.4)
    assert a.shape == (24,) and not np.allclose(a, b)
    assert 0 < a.min() and a.max() <= 1.2


@pytest.mark.parametrize("glyph", list(pillart.GLYPHS) + ["stop"])
def test_badge_symbols_are_white_on_the_gradient_and_draw_in(glyph):
    palette = pillart.PALETTES["info"]
    done = pillart.badge(40, 1.0, 1.0, palette, BG, glyph)
    empty = pillart.badge(40, 1.0, 0.0, palette, BG, glyph)
    white = lambda img: (img.min(axis=2) > 230).sum()
    assert white(empty) == 0 and white(done) > 20
    assert (done[0, 0] == [0x20, 0x21, 0x24]).all() # round: the corners stay background


def test_spinner_turns():
    a = pillart.spinner(20, 0.0, pillart.PALETTES["loading"], BG)
    b = pillart.spinner(20, 0.5, pillart.PALETTES["loading"], BG)
    assert a.shape == (20, 20, 3) and not np.array_equal(a, b)
    assert (pillart.spinner(20, 0.0, pillart.PALETTES["loading"], BG, grow=0.0)
            == [0x20, 0x21, 0x24]).all() # shrunk to nothing
