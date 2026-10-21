"""Recorder without a microphone: a fake stream stands in for sounddevice."""
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
    assert len(audio) == 320  # the dictation isn't lost
    assert stream.closed and not r.is_recording  # and the next start() opens a new stream
