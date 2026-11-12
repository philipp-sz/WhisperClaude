"""Paste path reliability: a finished transcript must never be lost to a side issue."""
import numpy as np

from whisperclaude.app import SAMPLE_RATE, TOGGLE, App

from test_app import BlockingTranscriber, FakeRecorder, begin, finish, saver


class SnapshotRecorder(FakeRecorder):
    def snapshot(self):
        return np.zeros(SAMPLE_RATE, np.float32)


class BrokenStatsLive:
    """Live transcription that works, but whose timing stats raise (not AttributeError)."""

    def feed(self, audio):
        pass

    def finish(self, audio):
        return "live text"

    @property
    def tail_s(self):
        raise RuntimeError("stats broken")


class LiveTranscriber(BlockingTranscriber):
    def __init__(self, live):
        super().__init__()
        self.release.set()
        self.made = live

    def live(self):
        return self.made


def test_failing_timing_log_does_not_lose_the_transcript():
    pasted, saved = [], []
    app = App(SnapshotRecorder(), LiveTranscriber(BrokenStatsLive()), pasted.append,
              remember=saver(saved))
    begin(app)
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["live text"]
    assert saved == [("live text", {})]


def test_failed_paste_offers_the_text_to_copy():
    def broken_paste(text):
        raise OSError("clipboard locked")

    kept, saved = [], []
    tr = BlockingTranscriber(text="text")
    tr.release.set()
    app = App(FakeRecorder(), tr, broken_paste, remember=saver(saved),
              on_kept=lambda text, msg: kept.append((text, msg)))
    begin(app)
    app.handle(TOGGLE)
    finish(app)
    assert kept and kept[0][0] == "text" and "clipboard locked" in kept[0][1]
    assert saved[0][0] == "text"
