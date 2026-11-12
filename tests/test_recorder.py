"""Recorder without a microphone: a fake stream stands in for sounddevice."""
import time

import numpy as np

from whisperclaude.recorder import Recorder


class FakeStream:
    def __init__(self, fail_stop=False):
        self.fail_stop, self.closed = fail_stop, False

    def stop(self):
        if self.fail_stop:
            raise OSError("device unplugged")

    def close(self):
        self.closed = True


def recording(stream):
    r = Recorder()
    r._stream = stream
    r._callback(np.ones((160, 1), np.float32), 160, None, None)
    r._callback(np.full((160, 1), 0.5, np.float32), 160, None, None)
    return r


def test_snapshot_while_recording_and_stop_returns_everything():
    r = recording(FakeStream())
    assert len(r.snapshot()) == 320 and r.is_recording
    audio = r.stop()
    assert len(audio) == 320 and not r.is_recording
    assert len(r.snapshot()) == 0


def test_stop_with_a_vanished_device_keeps_the_audio():
    stream = FakeStream(fail_stop=True)
    r = recording(stream)
    audio = r.stop()
    assert len(audio) == 320 # the dictation isn't lost
    assert stream.closed and not r.is_recording # and the next start() opens a new stream


class SlowCloseStream(FakeStream):
    """close() takes 0.3 s, like the real stream."""

    def close(self):
        time.sleep(0.3)
        self.closed = True


def test_stop_returns_before_the_slow_close_finishes():
    stream = SlowCloseStream()
    r = recording(stream)
    start = time.monotonic()
    audio = r.stop()
    assert time.monotonic() - start < 0.2
    assert len(audio) == 320 and not stream.closed
    r._wait_for_close()
    assert stream.closed


def test_start_after_stop_waits_until_the_close_finished(monkeypatch):
    stream = SlowCloseStream()
    r = recording(stream)
    r.stop()
    seen = []

    def open_stream():
        seen.append(stream.closed) # what the new stream sees: the old one must be closed
        return FakeStream()

    monkeypatch.setattr(r, "_open", open_stream)
    r.start()
    assert seen == [True] and r.is_recording


def test_refresh_devices_waits_until_the_close_finished(monkeypatch):
    import sounddevice as sd

    stream = SlowCloseStream()
    r = recording(stream)
    r.stop()
    seen = []
    monkeypatch.setattr(sd, "_terminate", lambda: seen.append(stream.closed))
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    r.refresh_devices()
    assert seen == [True]


def test_a_close_that_outlasts_the_timeout_is_logged_and_start_goes_on(monkeypatch, caplog):
    import whisperclaude.recorder as recorder_module

    monkeypatch.setattr(recorder_module, "CLOSE_TIMEOUT_S", 0.05)
    r = recording(SlowCloseStream())
    r.stop()
    monkeypatch.setattr(r, "_open", lambda: FakeStream())
    with caplog.at_level("WARNING", logger="whisperclaude.recorder"):
        r.start()
    assert r.is_recording
    assert any("still closing" in m for m in caplog.messages)


BLOCK = np.ones((160, 1), np.float32)


def test_first_block_callback_fires_once_when_audio_arrives():
    r = Recorder()
    r._open = lambda: FakeStream()
    calls = []
    r.start(lambda: calls.append(1))
    assert calls == [] and r.is_recording # stream open, but no audio yet
    r._callback(BLOCK, 160, None, None)
    r._callback(BLOCK, 160, None, None)
    assert calls == [1] and len(r.stop()) == 320


def test_start_on_a_running_stream_reports_at_once_and_stop_drops_the_callback():
    r = recording(FakeStream())
    calls = []
    r.start(lambda: calls.append("running")) # already open: audio flows
    assert calls == ["running"]
    r.stop()
    r._open = lambda: FakeStream()
    r.start(lambda: calls.append("stale"))
    r.stop() # closed before any audio: the callback must not fire later
    r._callback(BLOCK, 160, None, None)
    assert calls == ["running"]
