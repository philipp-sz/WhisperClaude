"""State machine (App) with fake recorder / transcriber / paste: no hardware needed."""
import threading
import time

import numpy as np

from whisperclaude import app as app_module
from whisperclaude.app import QUIT, RESTART, SAMPLE_RATE, TOGGLE, App, State, WakeQueue


class FakeRecorder:
    def __init__(self, seconds=1.0, fail=False):
        self.seconds, self.fail, self.started = seconds, fail, 0

    def start(self):
        if self.fail:
            raise OSError("no microphone")
        self.started += 1

    def stop(self):
        return np.zeros(int(self.seconds * SAMPLE_RATE), np.float32)


class BlockingTranscriber:
    """Stays 'busy' until release is set, so we can act while TRANSCRIBING."""

    def __init__(self, text="hallo welt"):
        self.text, self.calls, self.release = text, 0, threading.Event()

    def transcribe(self, audio):
        self.calls += 1
        self.release.wait(5)
        return self.text


def make_app(recorder=None, transcriber=None, paste=None):
    states, pasted = [], []
    app = App(recorder or FakeRecorder(), transcriber or BlockingTranscriber(),
              paste or pasted.append, on_state=lambda s, msg="": states.append((s, msg)))
    return app, states, pasted


def finish(app):
    """Wait for the worker's result event and handle it (like the Tk loop would)."""
    kind, payload = app.events.get(timeout=5)
    app.handle(kind, payload)


def test_full_cycle_pastes_text():
    app, states, pasted = make_app()
    app.transcriber.release.set()
    app.handle(TOGGLE)
    assert app.state is State.RECORDING
    app.handle(TOGGLE)
    assert app.state is State.TRANSCRIBING
    finish(app)
    assert app.state is State.IDLE
    assert pasted == ["hallo welt"]
    assert states[-1] == (State.IDLE, "")


def test_toggle_during_transcribing_is_ignored():
    app, _, pasted = make_app()
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    app.handle(TOGGLE)  # ignored
    app.handle(TOGGLE)  # ignored
    assert app.state is State.TRANSCRIBING
    assert app.recorder.started == 1
    app.transcriber.release.set()
    finish(app)
    assert app.state is State.IDLE
    assert app.transcriber.calls == 1
    assert pasted == ["hallo welt"]


def test_empty_transcript_is_not_pasted():
    tr = BlockingTranscriber(text="")
    tr.release.set()
    app, states, pasted = make_app(transcriber=tr)
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    finish(app)
    assert pasted == []
    assert states[-1] == (State.IDLE, "Nothing recognized")


def test_too_short_recording_skips_transcription():
    app, states, _ = make_app(recorder=FakeRecorder(seconds=0.1))
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    assert app.transcriber.calls == 0
    assert states[-1] == (State.IDLE, "Too short")


def test_mic_error_returns_to_idle():
    app, states, _ = make_app(recorder=FakeRecorder(fail=True))
    app.handle(TOGGLE)
    assert app.state is State.IDLE
    assert "no microphone" in states[-1][1]


def test_paste_error_is_reported():
    def broken_paste(text):
        raise OSError("clipboard locked")

    app, states, _ = make_app(paste=broken_paste)
    app.transcriber.release.set()
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    finish(app)
    assert app.state is State.IDLE
    assert "clipboard locked" in states[-1][1]


def test_process_pending_stops_on_quit():
    app, _, _ = make_app()
    app.events.put((TOGGLE, None))
    assert app.process_pending() is True
    assert app.state is State.RECORDING
    app.events.put((QUIT, None))
    assert app.process_pending() is False
    assert app.exit_reason == QUIT


def test_restart_stops_with_reason():
    app, _, _ = make_app()
    app.events.put((RESTART, None))
    assert app.process_pending() is False
    assert app.exit_reason == RESTART


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_wake_queue_calls_wake_and_swallows_errors():
    q = WakeQueue()
    calls = []
    q.wake = lambda: calls.append(1)
    q.put("a")
    assert wait_for(lambda: calls == [1])

    def broken():
        raise RuntimeError("main thread is not in main loop")

    q.wake = broken
    q.put("b")  # must not raise: the backup poll picks it up
    assert [q.get_nowait(), q.get_nowait()] == ["a", "b"]


def test_wake_queue_put_never_waits_for_the_consumer():
    """put() runs inside the keyboard hook callback: if it blocked on a busy Tk thread,
    Windows would silently remove the hook."""
    q = WakeQueue()
    release = threading.Event()
    q.wake = lambda: release.wait(5)  # a consumer that is stuck
    t0 = time.monotonic()
    for i in range(3):
        q.put(i)
    assert time.monotonic() - t0 < 0.2
    release.set()


# ---------- LOADING state: the app is alive while the model loads ----------

def make_loading_app():
    states = []
    app = App(FakeRecorder(), None, lambda text: None,
              on_state=lambda s, msg="": states.append((s, msg)))
    return app, states


def test_starts_in_loading_without_transcriber():
    app, _ = make_loading_app()
    assert app.state is State.LOADING


def test_toggle_while_loading_says_so_and_does_not_record():
    app, states = make_loading_app()
    app.handle(TOGGLE)
    assert app.state is State.LOADING
    assert app.recorder.started == 0
    assert states == [(State.LOADING, "Still loading – one moment…")]


def test_set_transcriber_ends_loading():
    app, states = make_loading_app()
    tr = BlockingTranscriber()
    tr.release.set()
    app.set_transcriber(tr)
    assert app.state is State.IDLE and states == []  # caller shows its own "ready" animation
    app.handle(TOGGLE)
    assert app.state is State.RECORDING


def test_quit_and_restart_work_while_loading():
    for kind in (QUIT, RESTART):
        app, _ = make_loading_app()
        app.events.put((TOGGLE, None))
        app.events.put((kind, None))
        assert app.process_pending() is False
        assert app.exit_reason == kind


def test_custom_event_handlers():
    app, _ = make_loading_app()
    seen = []
    app.handlers["loaded"] = seen.append
    app.handle("loaded", "payload")
    app.handle("unknown kind", "ignored")  # no handler: ignored
    assert seen == ["payload"]


# ---------- postprocess (spoken formatting commands) ----------

def run_cycle(app):
    app.transcriber.release.set()
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    finish(app)


def test_postprocess_runs_between_transcription_and_paste():
    tr = BlockingTranscriber(text="hello command new line world")
    pasted = []
    app = App(FakeRecorder(), tr, pasted.append,
              postprocess=lambda t: t.replace("command new line", "\n"))
    run_cycle(app)
    assert pasted == ["hello \n world"]


def test_postprocess_result_that_is_only_a_line_break_is_still_pasted():
    tr = BlockingTranscriber(text="command new line")
    pasted = []
    app = App(FakeRecorder(), tr, pasted.append, postprocess=lambda t: "\n")
    run_cycle(app)
    assert pasted == ["\n"]  # "nothing recognized" is judged on the raw transcript


def test_empty_transcript_skips_postprocess_and_paste():
    called = []
    tr = BlockingTranscriber(text="")
    app = App(FakeRecorder(), tr, lambda t: called.append("paste"),
              postprocess=lambda t: called.append("post") or t)
    states = []
    app.on_state = lambda s, msg="": states.append((s, msg))
    run_cycle(app)
    assert called == [] and states[-1] == (State.IDLE, "Nothing recognized")


def test_failing_postprocess_falls_back_to_the_raw_text():
    def broken(text):
        raise ValueError("bug")

    tr = BlockingTranscriber(text="raw words")
    pasted = []
    app = App(FakeRecorder(), tr, pasted.append, postprocess=broken)
    run_cycle(app)
    assert pasted == ["raw words"]


# ---------- live transcription while recording ----------

class SnapshotRecorder(FakeRecorder):
    def snapshot(self):
        return np.zeros(SAMPLE_RATE, np.float32)


class FakeLive:
    def __init__(self, fail=False):
        self.fail, self.feeds, self.finished = fail, 0, None

    def feed(self, audio):
        self.feeds += 1
        if self.fail:
            raise RuntimeError("GPU hiccup")

    def finish(self, audio):
        self.finished = len(audio)
        return "live text"


class LiveTranscriber(BlockingTranscriber):
    def __init__(self, live):
        super().__init__()
        self.release.set()
        self.made = live

    def live(self):
        return self.made


def test_live_transcription_is_fed_while_recording_and_finished_at_the_stop(monkeypatch):
    monkeypatch.setattr(app_module, "LIVE_FEED_S", 0.01)
    live = FakeLive()
    pasted = []
    app = App(SnapshotRecorder(), LiveTranscriber(live), pasted.append)
    app.handle(TOGGLE)
    assert wait_for(lambda: live.feeds >= 2)
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["live text"] and live.finished == SAMPLE_RATE
    assert app.transcriber.calls == 0  # no second pass over the whole recording


def test_failing_live_feed_still_finishes_at_the_stop(monkeypatch):
    monkeypatch.setattr(app_module, "LIVE_FEED_S", 0.01)
    live = FakeLive(fail=True)
    pasted = []
    app = App(SnapshotRecorder(), LiveTranscriber(live), pasted.append)
    app.handle(TOGGLE)
    assert wait_for(lambda: live.feeds >= 1)
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["live text"] and live.feeds == 1  # feeding stopped after the error


def test_failing_recorder_stop_returns_to_idle():
    class BrokenStop(FakeRecorder):
        def stop(self):
            raise OSError("device unplugged")

    app, states, _ = make_app(recorder=BrokenStop())
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    assert app.state is State.IDLE and "device unplugged" in states[-1][1]
    app.handle(TOGGLE)
    assert app.state is State.RECORDING  # not stuck


# ---------- auto-stop of a forgotten recording ----------

class SilentLive(FakeLive):
    def __init__(self, speech):
        super().__init__()
        self.speech, self.checks = speech, 0

    def has_speech(self, audio):
        self.checks += 1
        return self.speech


def auto_app(monkeypatch, speech):
    monkeypatch.setattr(app_module, "LIVE_FEED_S", 0.01)
    live, pasted, copied, states = SilentLive(speech), [], [], []
    app = App(SnapshotRecorder(), LiveTranscriber(live), pasted.append,
              on_state=lambda s, msg="": states.append((s, msg)),
              stop_after_silence_s=0.5, copy=copied.append)  # the fake mic always has 1 s
    app.handle(TOGGLE)
    return app, live, pasted, copied, states


def test_silent_recording_stops_itself_and_copies_instead_of_pasting(monkeypatch):
    app, live, pasted, copied, states = auto_app(monkeypatch, speech=False)
    kind, payload = app.events.get(timeout=5)
    assert kind == app_module.AUTO_STOP
    app.handle(kind, payload)
    assert app.state is State.TRANSCRIBING
    finish(app)
    assert copied == ["live text"] and pasted == []
    assert states[-1] == (State.IDLE, "Stopped after 0 s of silence – text is in the clipboard")


def test_recording_with_speech_keeps_going(monkeypatch):
    app, live, pasted, _, _ = auto_app(monkeypatch, speech=True)
    assert wait_for(lambda: live.checks >= 5)
    assert app.events.empty() and app.state is State.RECORDING
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["live text"]


def test_auto_stop_after_a_manual_stop_is_ignored():
    app, states, _ = make_app()
    app.handle(app_module.AUTO_STOP)  # arrives late: the user already stopped
    assert app.state is State.IDLE and states == []


# ---------- cancel (Esc) ----------

def test_cancel_discards_the_recording():
    app, states, pasted = make_app()
    app.handle(TOGGLE)
    app.handle(app_module.CANCEL)
    assert app.state is State.IDLE and states[-1] == (State.IDLE, "Cancelled")
    assert app.transcriber.calls == 0 and pasted == []
    app.handle(TOGGLE)
    assert app.state is State.RECORDING  # next recording works


def test_cancel_outside_a_recording_is_ignored():
    app, states, _ = make_app()
    app.handle(app_module.CANCEL)  # Esc pressed somewhere while idle
    assert app.state is State.IDLE and states == []
