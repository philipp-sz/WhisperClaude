"""Pill animations (pure numpy, no Tk)."""
import numpy as np
import pytest

from whisperclaude import pillart

BG = "#202124"


def test_bars_follow_the_values_and_keep_the_background():
    quiet = pillart.bars(78, 28, np.zeros(16), 3, 2, 4, 20, pillart.PALETTES["recording"], BG)
    loud = pillart.bars(78, 28, np.ones(16), 3, 2, 4, 20, pillart.PALETTES["recording"], BG)
    assert quiet.shape == (28, 78, 3) and quiet.dtype == np.uint8
    assert (quiet[0, 0] == [0x20, 0x21, 0x24]).all()  # nothing drawn outside the bars
    lit = lambda img: (img.max(axis=2) > 0x60).sum()
    assert lit(loud) > 3 * lit(quiet)
    # left to right: red end of the gradient on the left, violet on the right
    assert loud[14, 1, 0] > loud[14, 1, 2] and loud[14, 76, 2] > loud[14, 76, 1]


def test_waves_grow_with_the_level_and_taper_at_the_ends():
    palette = pillart.PALETTES["recording"]
    silent = pillart.waves(96, 28, np.zeros(36), 0.0, palette, BG)
    speech = pillart.waves(96, 28, np.full(36, 0.8), 0.0, palette, BG)
    colored = lambda img: (np.ptp(img.astype(int), axis=2) > 40).sum()
    assert colored(silent) == 0  # only the thin grey-white center line
    assert colored(speech) > 500
    assert colored(speech[:, :1]) * 4 < colored(speech[:, 48:49])  # tapered at the edges
    assert (speech[0, :] == [0x20, 0x21, 0x24]).all()  # stays inside the image


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
    assert (done[0, 0] == [0x20, 0x21, 0x24]).all()  # round: the corners stay background


def test_spinner_turns():
    a = pillart.spinner(20, 0.0, pillart.PALETTES["loading"], BG)
    b = pillart.spinner(20, 0.5, pillart.PALETTES["loading"], BG)
    assert a.shape == (20, 20, 3) and not np.array_equal(a, b)
    assert (pillart.spinner(20, 0.0, pillart.PALETTES["loading"], BG, grow=0.0)
            == [0x20, 0x21, 0x24]).all()  # shrunk to nothing
