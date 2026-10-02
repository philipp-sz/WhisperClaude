"""Robustness after sleep / session changes: hotkey re-arming, resume detection, audio refresh."""
import queue

import numpy as np
import pytest
import sounddevice as sd

from whisperclaude import hotkey, recorder
from whisperclaude.hotkey import GapDetector, HotkeyListener


# ---------- GapDetector ----------

def test_gap_detector_flags_only_long_gaps():
    now = [1000.0]
    d = GapDetector(threshold_s=15, clock=lambda: now[0])
    for _ in range(5):  # regular 1 s polls
        now[0] += 1.0
        assert d.check() == 0.0
    now[0] += 3600  # the laptop slept for an hour
    assert d.check() == pytest.approx(3600)
    now[0] += 1.0   # and the next normal poll is quiet again
    assert d.check() == 0.0


def test_gap_detector_ignores_short_stalls():
    now = [0.0]
    d = GapDetector(threshold_s=15, clock=lambda: now[0])
    now[0] += 10
    assert d.check() == 0.0


# ---------- HotkeyListener.restart ----------

class FakeListener:
    instances = []

    def __init__(self, **callbacks):
        self.callbacks, self.started, self.stopped, self.daemon = callbacks, False, False, False
        FakeListener.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_listeners(monkeypatch):
    FakeListener.instances = []
    monkeypatch.setattr(hotkey, "Listener", FakeListener)
    monkeypatch.setattr(hotkey.mouse, "Listener", FakeListener)
    return FakeListener


def test_restart_replaces_hooks_and_still_toggles(fake_listeners):
    events = queue.Queue()
    hl = HotkeyListener(events)
    first = list(fake_listeners.instances)
    assert len(first) == 2 and all(i.started for i in first)

    hl.restart()
    assert all(i.stopped for i in first)  # old hooks removed
    fresh = fake_listeners.instances[2:]
    assert len(fresh) == 2 and all(i.started for i in fresh)  # new hooks installed

    kb = next(i for i in fresh if "on_press" in i.callbacks)  # the keyboard listener
    kb.callbacks["on_press"](hotkey.DEFAULT_KEY)
    kb.callbacks["on_release"](hotkey.DEFAULT_KEY)
    assert events.get_nowait() == (hotkey.TOGGLE, None)


# ---------- Recorder: refresh PortAudio's device list ----------

class FakeStream:
    def __init__(self, **kwargs):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def audio(monkeypatch):
    calls = {"terminate": 0, "initialize": 0, "opened": 0, "fail_first": 0}

    def input_stream(**kwargs):
        calls["opened"] += 1
        if calls["fail_first"] > 0:
            calls["fail_first"] -= 1
            raise sd.PortAudioError("Error querying device -1")
        return FakeStream(**kwargs)

    monkeypatch.setattr(recorder.sd, "InputStream", input_stream)
    monkeypatch.setattr(recorder.sd, "_terminate", lambda: calls.__setitem__("terminate", calls["terminate"] + 1))
    monkeypatch.setattr(recorder.sd, "_initialize", lambda: calls.__setitem__("initialize", calls["initialize"] + 1))
    return calls


def test_start_refreshes_devices_and_retries_once(audio):
    audio["fail_first"] = 1  # e.g. stale device list after waking up
    r = recorder.Recorder()
    r.start()
    assert r.is_recording
    assert audio["opened"] == 2 and audio["terminate"] == 1 and audio["initialize"] == 1


def test_start_gives_up_after_one_retry(audio):
    audio["fail_first"] = 5
    r = recorder.Recorder()
    with pytest.raises(sd.PortAudioError):
        r.start()
    assert audio["opened"] == 2 and not r.is_recording


def test_refresh_devices_skipped_while_recording(audio):
    r = recorder.Recorder()
    r.start()
    r.refresh_devices()
    assert audio["terminate"] == 0
    r.stop()
    r.refresh_devices()
    assert audio["terminate"] == 1 and audio["initialize"] == 1


def test_healthy_start_does_not_touch_portaudio(audio):
    r = recorder.Recorder()
    r.start()
    assert audio["opened"] == 1 and audio["terminate"] == 0
    assert isinstance(r.stop(), np.ndarray)
