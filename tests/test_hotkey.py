import pytest
from pynput.keyboard import Key, KeyCode

from whisperclaude.hotkey import TapDetector, parse_key


@pytest.fixture
def tap():
    clock = [0.0]
    detector = TapDetector(Key.ctrl_l, max_hold=0.5, clock=lambda: clock[0])
    detector.clock_value = clock
    return detector


def press_release(d, hold=0.1, key=Key.ctrl_l, between=None):
    d.press(key)
    d.press(key)  # auto-repeat while held
    if between:
        between(d)
    d.clock_value[0] += hold
    return d.release(key)


def test_short_tap_triggers(tap):
    assert press_release(tap) is True
    assert press_release(tap) is True  # and again


@pytest.mark.parametrize("between", [
    lambda d: (d.press(KeyCode.from_char("c")), d.release(KeyCode.from_char("c"))),  # Ctrl+C
    lambda d: d.press(Key.alt_gr),  # AltGr sends Ctrl+Alt on German layouts
    lambda d: d.interrupt(),        # mouse click / scroll while held
])
def test_combinations_do_not_trigger(tap, between):
    assert press_release(tap, between=between) is False


def test_long_hold_does_not_trigger(tap):
    assert press_release(tap, hold=1.0) is False


def test_other_ctrl_does_not_trigger(tap):
    assert press_release(tap, key=Key.ctrl_r) is False


def test_parse_key():
    assert parse_key("ctrl_l") == Key.ctrl_l
    assert parse_key("f9") == Key.f9
    assert parse_key("x") == KeyCode.from_char("x")
    with pytest.raises(ValueError):
        parse_key("ctrl_x")
