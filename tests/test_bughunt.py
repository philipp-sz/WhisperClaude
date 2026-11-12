"""Robustness gaps found by a bug hunt (2026-11-12): each test failed before its fix."""
import threading
from types import SimpleNamespace

import numpy as np

from whisperclaude import __main__ as main
from whisperclaude import app as app_module
from whisperclaude.app import TOGGLE, App, State
from whisperclaude.recorder import Recorder

from test_app import BlockingTranscriber, FakeRecorder, begin, finish


# ---------- an exception in the UI callback must not strand the state machine ----------

def test_failing_pill_does_not_lose_the_dictation_or_stick_in_transcribing():
    """on_state raising (a Tk or tray error) at the stop tap used to skip starting the worker:
    the audio was lost and the app stayed in TRANSCRIBING, ignoring every tap."""
    pasted = []

    def on_state(state, msg=""):
        if state is State.TRANSCRIBING:
            raise RuntimeError("pill failed")

    tr = BlockingTranscriber()
    tr.release.set()
    app = App(FakeRecorder(), tr, pasted.append, on_state=on_state)
    begin(app)
    app.handle(TOGGLE) # stop tap
    finish(app)
    assert pasted == ["hallo welt"] and app.state is State.IDLE


def test_failing_pill_at_the_first_audio_block_still_starts_recording():
    seen = []

    def on_state(state, msg=""):
        seen.append(state)
        if state is State.RECORDING:
            raise RuntimeError("pill failed")

    tr = BlockingTranscriber()
    tr.release.set()
    app = App(FakeRecorder(), tr, lambda t: None, on_state=on_state)
    begin(app)
    assert app.state is State.RECORDING
    app.handle(TOGGLE) # the stop tap works
    assert app.state is State.TRANSCRIBING


# ---------- Recorder.refresh_devices must not tear PortAudio down under a starting stream ----------

class _Stream:
    def stop(self):
        pass

    def close(self):
        pass


def test_refresh_devices_waits_out_a_stream_that_is_opening(monkeypatch):
    """The resume check (Tk thread) calls refresh_devices() while the mic-start thread is inside
    start(): _stream is still None then, so PortAudio was terminated under the opening stream."""
    import sounddevice as sd

    terminated = []
    monkeypatch.setattr(sd, "_terminate", lambda: terminated.append(1))
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    r = Recorder()
    opening, go = threading.Event(), threading.Event()

    def slow_open():
        opening.set()
        go.wait(5)
        return _Stream()

    monkeypatch.setattr(r, "_open", slow_open)
    starter = threading.Thread(target=r.start)
    starter.start()
    assert opening.wait(2)
    r.refresh_devices() # while start() is inside _open()
    go.set()
    starter.join(5)
    assert terminated == [] and r.is_recording


def test_start_retry_after_a_portaudio_error_may_still_refresh_devices(monkeypatch):
    """The guard above must not block start()'s own refresh-and-retry."""
    import sounddevice as sd

    terminated = []
    monkeypatch.setattr(sd, "_terminate", lambda: terminated.append(1))
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    r = Recorder()
    calls = []

    def flaky_open():
        calls.append(1)
        if len(calls) == 1:
            raise sd.PortAudioError("stale device list")
        return _Stream()

    monkeypatch.setattr(r, "_open", flaky_open)
    r.start()
    assert terminated == [1] and r.is_recording


# ---------- Session: one failing step must not end the backup poll or the shutdown ----------

class _Root:
    def __init__(self):
        self.scheduled = []

    def after(self, ms, fn):
        self.scheduled.append(ms)


def _poll_session(tmp_path, pump):
    s = main.Session.__new__(main.Session)
    s.root, s.data_dir, s.stopped = _Root(), tmp_path, False
    s.last_poll = s.last_beat = 0.0
    s.gap = SimpleNamespace(threshold_s=15.0, check=lambda: 0.0)
    s.watch = SimpleNamespace(due=lambda resumed=False: False)
    s._apply_upgrade, s.pump = lambda: None, pump
    return s


def test_backup_poll_keeps_running_after_a_failing_pump(tmp_path):
    """backup_poll is the safety net when a wake-up is missed and the only resume / heartbeat /
    GPU check; an exception in a handler ended its `after` chain for good."""
    def pump():
        raise RuntimeError("handler failed")

    s = _poll_session(tmp_path, pump)
    s.backup_poll()
    assert s.root.scheduled == [main.BACKUP_POLL_MS]


def test_quit_shows_the_goodbye_even_if_the_tray_fails_to_stop(tmp_path):
    shown = []
    s = main.Session.__new__(main.Session)
    s.stopped, s.events = False, SimpleNamespace(wake=None)
    s.app = SimpleNamespace(process_pending=lambda: False, exit_reason=app_module.QUIT)

    def tray_stop():
        raise RuntimeError("tray gone")

    s.listener = SimpleNamespace(stop=lambda: None)
    s.tray = SimpleNamespace(stop=tray_stop)
    s.overlay = SimpleNamespace(show_goodbye=lambda *a, **k: shown.append(a))
    s.pump()
    assert len(shown) == 1
