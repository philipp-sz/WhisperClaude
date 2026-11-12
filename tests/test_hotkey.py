import pytest
from pynput.keyboard import Key, KeyCode

from whisperclaude.hotkey import (OWN_EVENT, WM_KEYDOWN, WM_SYSKEYDOWN, TapDetector,
                                   TranslateKeyFilter, key_vk, parse_key)


@pytest.fixture
def tap():
    clock = [0.0]
    detector = TapDetector(Key.ctrl_l, max_hold=0.5, clock=lambda: clock[0])
    detector.clock_value = clock
    return detector


def press_release(d, hold=0.1, key=Key.ctrl_l, between=None):
    d.press(key)
    d.press(key) # auto-repeat while held
    if between:
        between(d)
    d.clock_value[0] += hold
    return d.release(key)


def test_short_tap_triggers(tap):
    assert press_release(tap) is True
    assert press_release(tap) is True # and again


@pytest.mark.parametrize("between", [
    lambda d: (d.press(KeyCode.from_char("c")), d.release(KeyCode.from_char("c"))), # Ctrl+C
    lambda d: d.press(Key.alt_gr), # AltGr sends Ctrl+Alt on German layouts
    lambda d: d.interrupt(), # mouse click / scroll while held
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


class KeyData:
    def __init__(self, vk, flags=0, scan=0x38, extra=None):
        self.vkCode, self.flags, self.scanCode, self.dwExtraInfo = vk, flags, scan, extra


class Suppressed(Exception):
    pass


@pytest.fixture
def alt_filter():
    clock, taps, sent = [0.0], [], []

    def suppress():
        raise Suppressed # like pynput's suppress_event()

    f = TranslateKeyFilter(0xA4, 0.5, lambda: taps.append(1), suppress, clock=lambda: clock[0],
                           send=lambda *a: sent.append(a))
    return f, clock, taps, sent


def test_alt_tap_is_swallowed_and_replayed(alt_filter):
    f, clock, taps, sent = alt_filter
    assert f(WM_SYSKEYDOWN, KeyData(0xA4)) is True # the press goes through
    clock[0] += 0.1
    with pytest.raises(Suppressed):
        f(0x101, KeyData(0xA4, flags=0x80)) # WM_KEYUP
    assert taps == [1] and sent == [(0xA4, 0x38, 0x80)]


def test_alt_combinations_pass_through(alt_filter):
    f, clock, taps, sent = alt_filter
    f(WM_SYSKEYDOWN, KeyData(0xA4))
    f(WM_SYSKEYDOWN, KeyData(0x09)) # Alt+Tab
    assert f(0x101, KeyData(0xA4)) is True
    f(WM_SYSKEYDOWN, KeyData(0xA4))
    f.interrupt() # Alt+click
    assert f(0x101, KeyData(0xA4)) is True
    f(WM_SYSKEYDOWN, KeyData(0xA4))
    clock[0] += 1.0 # held too long
    assert f(0x101, KeyData(0xA4)) is True
    assert taps == [] and sent == []


def test_own_events_are_ignored(alt_filter):
    f, clock, taps, sent = alt_filter
    f(WM_SYSKEYDOWN, KeyData(0xA4))
    f(WM_KEYDOWN, KeyData(0xE8, flags=0x10, extra=OWN_EVENT)) # our mask key: no interruption
    with pytest.raises(Suppressed):
        f(0x101, KeyData(0xA4))
    assert f(0x101, KeyData(0xA4, flags=0x10, extra=OWN_EVENT)) is True # our replay passes
    assert taps == [1]


def test_unmasked_key_tap_is_not_swallowed():
    taps = []
    f = TranslateKeyFilter(0x78, 0.5, lambda: taps.append(1), lambda: 1 / 0, clock=lambda: 0.0)
    f(WM_KEYDOWN, KeyData(0x78)) # F9: nothing to mask
    assert f(0x101, KeyData(0x78)) is True and taps == [1]


def test_filter_errors_never_escape(alt_filter):
    f, clock, taps, sent = alt_filter
    f.send = lambda *a: 1 / 0
    f(WM_SYSKEYDOWN, KeyData(0xA4))
    assert f(0x101, KeyData(0xA4)) is True # an exception would stop pynput's listener


def test_key_vk():
    assert key_vk(Key.alt_l) == 0xA4
    assert key_vk(Key.ctrl_l) == 0xA2


def test_every_accepted_key_name_is_what_the_hook_delivers():
    """pynput's Windows hook reports left/right variants (Key.ctrl_l, Key.alt_l, Key.alt_gr for
    the right Alt), never the generic Key.ctrl (0x11) / Key.alt (0x12); a name that parse_key
    accepts but the hook never delivers is a hotkey that silently does nothing."""
    from pynput.keyboard import Listener
    delivered = Listener._SPECIAL_KEYS
    for name in Key.__members__:
        try:
            key = parse_key(name)
        except ValueError:
            continue
        assert delivered.get(key_vk(key)) == key, name


def test_generic_modifier_names_are_rejected_with_a_hint():
    for name, hint in (("ctrl", "ctrl_l"), ("alt", "alt_l")):
        with pytest.raises(ValueError, match=hint):
            parse_key(name)
    assert parse_key("alt_r") == Key.alt_gr # what the hook reports for VK 0xA5
