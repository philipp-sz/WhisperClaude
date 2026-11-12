"""State machine (App) with fake recorder / transcriber / paste: no hardware needed."""
import threading
import time

import numpy as np
import pytest

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


def saver(saved):
    """A remember() that records (text, {the arguments that aren't the default})."""
    return lambda text, **kw: saved.append((text, {k: v for k, v in kw.items() if v}))


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
    app.handle(TOGGLE) # ignored
    app.handle(TOGGLE) # ignored
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
    q.put("b") # must not raise: the backup poll picks it up
    assert [q.get_nowait(), q.get_nowait()] == ["a", "b"]


def test_wake_queue_put_never_waits_for_the_consumer():
    """put() runs inside the keyboard hook callback: if it blocked on a busy Tk thread,
    Windows would silently remove the hook."""
    q = WakeQueue()
    release = threading.Event()
    q.wake = lambda: release.wait(5) # a consumer that is stuck
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
    assert states == [(State.LOADING, "Still loading, one moment…")]


def test_set_transcriber_ends_loading():
    app, states = make_loading_app()
    tr = BlockingTranscriber()
    tr.release.set()
    app.set_transcriber(tr)
    assert app.state is State.IDLE and states == [] # caller shows its own "ready" animation
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
    app.handle("unknown kind", "ignored") # no handler: ignored
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
    assert pasted == ["\n"] # "nothing recognized" is judged on the raw transcript


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
    assert app.transcriber.calls == 0 # no second pass over the whole recording


def test_failing_live_feed_still_finishes_at_the_stop(monkeypatch):
    monkeypatch.setattr(app_module, "LIVE_FEED_S", 0.01)
    live = FakeLive(fail=True)
    pasted = []
    app = App(SnapshotRecorder(), LiveTranscriber(live), pasted.append)
    app.handle(TOGGLE)
    assert wait_for(lambda: live.feeds >= 1)
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["live text"] and live.feeds == 1 # feeding stopped after the error


def test_failing_recorder_stop_returns_to_idle():
    class BrokenStop(FakeRecorder):
        def stop(self):
            raise OSError("device unplugged")

    app, states, _ = make_app(recorder=BrokenStop())
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    assert app.state is State.IDLE and "device unplugged" in states[-1][1]
    app.handle(TOGGLE)
    assert app.state is State.RECORDING # not stuck


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
    live, pasted, kept, states = SilentLive(speech), [], [], []
    app = App(SnapshotRecorder(), LiveTranscriber(live), pasted.append,
              on_state=lambda s, msg="": states.append((s, msg)), stop_after_silence_s=0.5,
              on_kept=lambda text, msg: kept.append((text, msg))) # the fake mic has 1 s
    app.handle(TOGGLE)
    return app, live, pasted, kept, states


def test_silent_recording_stops_itself_and_keeps_the_text_instead_of_pasting(monkeypatch):
    app, live, pasted, kept, states = auto_app(monkeypatch, speech=False)
    kind, payload = app.events.get(timeout=5)
    assert kind == app_module.AUTO_STOP
    app.handle(kind, payload)
    assert app.state is State.TRANSCRIBING
    finish(app)
    assert kept == [("live text", "Stopped after 0 s of silence")] and pasted == []
    assert app.state is State.IDLE


def test_recording_with_speech_keeps_going(monkeypatch):
    app, live, pasted, _, _ = auto_app(monkeypatch, speech=True)
    assert wait_for(lambda: live.checks >= 5)
    assert app.events.empty() and app.state is State.RECORDING
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["live text"]


def test_auto_stop_after_a_manual_stop_is_ignored():
    app, states, _ = make_app()
    app.handle(app_module.AUTO_STOP) # arrives late: the user already stopped
    assert app.state is State.IDLE and states == []


# ---------- cancel (Esc) ----------

def test_cancel_keeps_the_text_in_the_history_without_pasting():
    saved, kept, pasted = [], [], []
    tr = BlockingTranscriber(text="halber Satz")
    tr.release.set()
    app = App(FakeRecorder(), tr, pasted.append, remember=saver(saved),
              on_kept=lambda text, msg: kept.append((text, msg)))
    app.handle(TOGGLE)
    app.handle(app_module.CANCEL)
    assert app.state is State.TRANSCRIBING
    finish(app)
    assert pasted == [] and kept == [("halber Satz", "Cancelled")]
    assert saved == [("halber Satz", {"note": "cancelled"})]
    app.handle(TOGGLE)
    assert app.state is State.RECORDING # next recording works


def test_cancel_with_nothing_said_just_says_cancelled():
    app, states, pasted = make_app(transcriber=BlockingTranscriber(text=""))
    app.transcriber.release.set()
    app.handle(TOGGLE)
    app.handle(app_module.CANCEL)
    finish(app)
    assert states[-1] == (State.IDLE, "Cancelled") and pasted == []


def test_cancel_outside_a_recording_is_ignored():
    app, states, _ = make_app()
    app.handle(app_module.CANCEL) # Esc pressed somewhere while idle
    assert app.state is State.IDLE and states == []


# ---------- history and the pill's "Copy" offer ----------

def test_transcript_is_saved_before_the_paste_and_offered_for_copying():
    order, offered = [], []
    tr = BlockingTranscriber(text="wichtiger Text")
    tr.release.set()
    app = App(FakeRecorder(), tr, lambda t: order.append(("paste", t)),
              remember=lambda t, **kw: order.append(("saved", t)), on_pasted=offered.append)
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    finish(app)
    assert order == [("saved", "wichtiger Text"), ("paste", "wichtiger Text")]
    assert offered == ["wichtiger Text"] and app.state is State.IDLE


def test_failing_history_does_not_stop_the_paste():
    def broken(text, **kw):
        raise OSError("disk full")

    pasted = []
    tr = BlockingTranscriber(text="text")
    tr.release.set()
    app = App(FakeRecorder(), tr, pasted.append, remember=broken)
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    finish(app)
    assert pasted == ["text"]


# ---------- translate key ----------

class FakeTranslation:
    """Picker answers are given by the test through .answer(); translate() tags the text."""

    def __init__(self, selection="Hallo Welt", fail=False):
        self.selection, self.fail = selection, fail
        self.picks, self.confirmed, self.closed, self.translated = [], 0, 0, []
        self.guesses, self.sources, self.prepared = [], [], 0
        self.ready, self.on_translate = True, None # model loaded?; hook run at translate()

    def prepare(self):
        self.prepared += 1

    def model_ready(self):
        return self.ready

    def capture_selection(self):
        return self.selection

    def guess(self, text):
        return ["deu_Latn"]

    def pick(self, on_done, dictation=False, guess=()):
        self.picks.append((on_done, dictation, list(guess)))

    def set_guess(self, codes):
        self.guesses.append(codes)

    def answer(self, choice, focus_back=True, source=None):
        """choice: target code (with `source`), or None = Esc / clicked elsewhere."""
        self.picks[-1][0](None if choice is None else (choice, source), focus_back)

    def confirm_pick(self):
        self.confirmed += 1

    def close_pick(self):
        self.closed += 1

    def translate(self, text, target, source=None):
        if self.on_translate:
            self.on_translate()
        if self.fail:
            raise RuntimeError("model broken")
        self.translated.append((text, target))
        self.sources.append(source)
        return f"[{target}] {text}"


def make_translate_app(translation, text="hallo welt"):
    states, pasted, kept, saved = [], [], [], []
    tr = BlockingTranscriber(text)
    tr.release.set()
    app = App(FakeRecorder(), tr, pasted.append,
              on_state=lambda s, msg="": states.append((s, msg)),
              on_kept=lambda text, msg: kept.append((text, msg)), remember=saver(saved),
              translation=translation)
    return app, states, pasted, kept, saved


def pump_until(app, condition, timeout=5.0):
    """Handle worker events (like the Tk loop) until condition() is true."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        try:
            kind, payload = app.events.get(timeout=0.05)
        except Exception:
            continue
        app.handle(kind, payload)


def test_translate_selection():
    t = FakeTranslation()
    app, states, pasted, kept, saved = make_translate_app(t)
    app.handle(app_module.TRANSLATE)
    assert app.state is State.TRANSLATING
    pump_until(app, lambda: t.picks)
    assert t.picks[0][1] is False # not a dictation
    app.handle(app_module.TRANSLATE) # tapped again in the picker: take the top language
    assert t.confirmed == 1
    t.answer("deu_Latn")
    pump_until(app, lambda: app.state is State.IDLE)
    assert pasted == ["[deu_Latn] Hallo Welt"]
    assert saved == [("Hallo Welt", {"translation": "[deu_Latn] Hallo Welt",
                                     "target": "deu_Latn"})] # both, one entry
    assert (State.TRANSLATING, "") in states


def test_translate_without_selection_says_so():
    t = FakeTranslation(selection=None)
    app, states, pasted, *_ = make_translate_app(t)
    app.handle(app_module.TRANSLATE)
    pump_until(app, lambda: app.state is State.IDLE)
    assert not t.picks and not pasted
    assert "Select text" in states[-1][1]


@pytest.mark.parametrize("dictation", [False, True])
def test_translate_key_starts_loading_the_model(dictation):
    """Once per tap, so the load overlaps the picker (selection) or the transcription."""
    t = FakeTranslation()
    app, *_ = make_translate_app(t)
    if dictation:
        app.handle(TOGGLE)
        assert t.prepared == 0
    app.handle(app_module.TRANSLATE)
    assert t.prepared == 1
    pump_until(app, lambda: t.picks)
    app.handle(app_module.TRANSLATE) # tapped again in the picker: no second load
    assert t.prepared == 1 and t.confirmed == 1


def test_translate_selection_cancelled_leaves_it_alone():
    t = FakeTranslation()
    app, states, pasted, kept, saved = make_translate_app(t)
    app.handle(app_module.TRANSLATE)
    pump_until(app, lambda: t.picks)
    t.answer(None)
    pump_until(app, lambda: app.state is State.IDLE)
    assert not pasted and not kept and not saved and not t.translated


def test_translated_selection_is_kept_when_the_focus_moved():
    t = FakeTranslation()
    app, states, pasted, kept, saved = make_translate_app(t)
    app.handle(app_module.TRANSLATE)
    pump_until(app, lambda: t.picks)
    t.answer("fra_Latn", focus_back=False)
    pump_until(app, lambda: app.state is State.IDLE)
    assert not pasted and kept == [("[fra_Latn] Hallo Welt", "Not pasted (the focus moved)")]
    assert saved == [("Hallo Welt", {"translation": "[fra_Latn] Hallo Welt",
                                     "target": "fra_Latn", "note": "not pasted"})]


def test_dictate_and_translate():
    t = FakeTranslation()
    app, states, pasted, kept, saved = make_translate_app(t)
    app.handle(TOGGLE)
    app.handle(app_module.TRANSLATE) # instead of the second Ctrl tap
    assert app.state is State.TRANSCRIBING
    assert t.picks and t.picks[0][1] is True # the picker opens right away
    t.answer("fra_Latn")
    pump_until(app, lambda: app.state is State.IDLE)
    assert t.translated == [("hallo welt", "fra_Latn")]
    assert pasted == ["[fra_Latn] hallo welt"]
    # the original first (kept whatever happens), then the translation added to that entry
    assert saved == [("hallo welt", {}), ("hallo welt", {"translation": "[fra_Latn] hallo welt",
                                                         "target": "fra_Latn", "update": True})]


@pytest.mark.parametrize("focus_back, pasted_expected, kept_expected, note", [
    (True, ["hallo welt"], [], None), # Esc: paste the dictation untranslated
    (False, [], [("hallo welt", "Not pasted (the focus moved)")], "not pasted"), # elsewhere
])
def test_dictation_picker_cancelled(focus_back, pasted_expected, kept_expected, note):
    t = FakeTranslation()
    app, states, pasted, kept, saved = make_translate_app(t)
    app.handle(TOGGLE)
    app.handle(app_module.TRANSLATE)
    t.answer(None, focus_back)
    pump_until(app, lambda: app.state is State.IDLE)
    assert pasted == pasted_expected and kept == kept_expected and not t.translated
    assert saved == [("hallo welt", {})] + ([("hallo welt", {"note": note, "update": True})]
                                            if note else [])


def test_failed_translation_keeps_the_dictation():
    t = FakeTranslation(fail=True)
    app, states, pasted, kept, saved = make_translate_app(t)
    app.handle(TOGGLE)
    app.handle(app_module.TRANSLATE)
    t.answer("fra_Latn")
    pump_until(app, lambda: app.state is State.IDLE)
    assert not pasted and kept == [("hallo welt", "Translation failed: model broken")]
    assert saved == [("hallo welt", {})]


def test_nothing_recognized_closes_the_picker():
    t = FakeTranslation()
    app, states, pasted, *_ = make_translate_app(t, text="")
    app.handle(TOGGLE)
    app.handle(app_module.TRANSLATE)
    pump_until(app, lambda: t.closed == 1)
    t.answer(None) # what close_pick() does in the real picker
    pump_until(app, lambda: app.state is State.IDLE)
    assert not pasted and states[-1] == (State.IDLE, "Nothing recognized")


def record_events(app):
    """Log every worker event put on the queue, in order (the worker is the only writer)."""
    seen = []
    put = app.events.put

    def spy(item, *args, **kwargs):
        seen.append(item)
        put(item, *args, **kwargs)

    app.events.put = spy
    return seen


@pytest.mark.parametrize("ready, expected", [(True, None), (False, "Loading translation model…")])
def test_translating_pill_says_when_the_model_is_loading(ready, expected):
    t = FakeTranslation()
    t.ready = ready
    app, states, pasted, kept, saved = make_translate_app(t)
    seen = record_events(app)
    t.on_translate = lambda: seen.append("translate")
    app.handle(app_module.TRANSLATE)
    pump_until(app, lambda: t.picks)
    t.answer("deu_Latn")
    pump_until(app, lambda: app.state is State.IDLE)
    assert pasted == ["[deu_Latn] Hallo Welt"]
    assert (State.TRANSLATING, expected or "") in states
    if expected is None:
        assert not any(isinstance(e, tuple) and e[0] == app_module.TRANSLATING and e[1]
                       for e in seen)
    else: # the message is queued before the translation call
        assert seen.index((app_module.TRANSLATING, expected)) < seen.index("translate")


def test_translate_key_without_feature_or_while_loading():
    app, states, pasted = make_app()
    app.handle(app_module.TRANSLATE) # feature off: nothing happens
    assert app.state is State.IDLE and not states
    t = FakeTranslation()
    loading = App(FakeRecorder(), None, lambda text: None, translation=t,
                  on_state=lambda s, msg="": states.append((s, msg)))
    loading.handle(app_module.TRANSLATE)
    assert loading.state is State.LOADING and "loading" in states[-1][1]


def test_selection_picker_shows_the_guess_and_the_picked_source_is_used():
    t = FakeTranslation()
    app, states, pasted, *_ = make_translate_app(t)
    app.handle(app_module.TRANSLATE)
    pump_until(app, lambda: t.picks)
    assert t.picks[0][2] == ["deu_Latn"] # "Auto · German" in the picker
    t.answer("fra_Latn", source="nld_Latn") # the user overrode the source
    pump_until(app, lambda: app.state is State.IDLE)
    assert t.sources == ["nld_Latn"] and pasted == ["[fra_Latn] Hallo Welt"]


def test_dictation_guess_reaches_the_open_picker():
    t = FakeTranslation()
    app, states, pasted, *_ = make_translate_app(t)
    app.handle(TOGGLE)
    app.handle(app_module.TRANSLATE)
    assert t.picks[0][2] == [] # opened before the transcript exists
    pump_until(app, lambda: t.guesses)
    assert t.guesses == [["deu_Latn"]]
    t.answer("fra_Latn")
    pump_until(app, lambda: app.state is State.IDLE)
    assert t.sources == [None] and pasted == ["[fra_Latn] hallo welt"]


class FailingTranscriber:
    """Raises like a model whose GPU was reset."""

    def transcribe(self, audio):
        raise RuntimeError("device lost")


class InstantTranscriber:
    def __init__(self, text="hallo welt"):
        self.text = text

    def transcribe(self, audio):
        return self.text


def run_dictation(app):
    """Toggle on and off, then handle the worker's events until the paste is done."""
    app.handle(TOGGLE)
    app.handle(TOGGLE)
    events = []
    while app.state is not State.IDLE:
        kind, payload = app.events.get(timeout=5)
        events.append(kind)
        app.handle(kind, payload)
    return events


def test_failed_transcription_reloads_and_retries():
    """Test that a failing model is loaded again and the same dictation is still pasted."""
    pasted, reloaded = [], InstantTranscriber("gerettet")
    app = App(FakeRecorder(), FailingTranscriber(), pasted.append, reload=lambda: reloaded)
    events = run_dictation(app)
    assert pasted == ["gerettet"]
    assert app.transcriber is reloaded and app_module.RELOADED in events


def test_failed_reload_reports_the_error():
    """Test that a reload that fails too ends in an error message, not a stuck state."""
    def broken():
        raise RuntimeError("no model")
    app = App(FakeRecorder(), FailingTranscriber(), lambda text: None, reload=broken)
    assert app_module.ERROR in run_dictation(app)
