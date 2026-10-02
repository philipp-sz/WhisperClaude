"""State machine (App) with fake recorder / transcriber / paste: no hardware needed."""
import threading
import time

import numpy as np

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
