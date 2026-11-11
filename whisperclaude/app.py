"""State machine: LOADING -> IDLE -> RECORDING -> TRANSCRIBING -> IDLE, driven by queue events.

The translate key adds TRANSLATING: IDLE -> TRANSLATING -> IDLE for selected text, and
RECORDING -> TRANSCRIBING -> TRANSLATING -> IDLE for a dictation. Both ask for the target
language in a picker (Tk thread) while the worker copies the selection or transcribes.

Hardware-free (recorder, transcriber and paste are injected), so it can be tested.
"""
from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from whisperclaude.constants import SAMPLE_RATE

log = logging.getLogger(__name__)

# Event kinds (queue items are (kind, payload) tuples)
TOGGLE = "toggle"
DONE = "done" # payload: message for the user
PASTED = "pasted" # payload: the pasted text (the pill offers to copy it)
ERROR = "error" # payload: message
QUIT = "quit"
RESTART = "restart" # quit and start a new instance (applies config.toml changes)
AUTO_STOP = "auto_stop" # a recording was silent for too long (probably forgotten)
CANCEL = "cancel" # the cancel key (Esc): discard a running recording, ignored otherwise
TRANSLATE = "translate" # the translate key (Alt): translate the dictation or the selection
PICK = "pick" # worker -> Tk: show the language picker; payload: _TranslateJob
UNPICK = "unpick" # worker -> Tk: close the picker, nothing to translate; payload: _TranslateJob
KEPT = "kept" # worker -> Tk: saved to the history but not pasted; payload: (text, message)
GUESS = "guess" # worker -> Tk: detected source languages of a dictation; payload: (job, codes)
TRANSLATING = "translating" # worker -> Tk: the translation itself has started
RELOADED = "reloaded" # worker -> Tk: the model was loaded again after a failure; payload: transcriber

MIN_AUDIO_S = 0.3
LIVE_FEED_S = 5.0 # while recording, hand the audio so far to the live transcription this often


class WakeQueue(queue.Queue):
    """Queue that wakes its consumer after each put, so the consumer can sleep instead of
    polling (polling every 50 ms cost ~0.8 % of a core while idle).

    put() never waits for the consumer: wake() runs on a helper thread. Callers include the
    keyboard hook callback, and Windows silently removes a low-level hook whose callback is
    too slow. Waking Tk from the callback's own thread blocks until Tk's main thread is free,
    which isn't guaranteed (logon storm, recorder start, ...).
    """

    def __init__(self) -> None:
        super().__init__()
        self.wake: Callable[[], None] = lambda: None
        self._pending = threading.Event()
        threading.Thread(target=self._waker, daemon=True, name="wake-queue").start()

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        self._pending.set()

    def _waker(self) -> None:
        while True:
            self._pending.wait()
            self._pending.clear() # clear first: a put during wake() sets it again
            try:
                self.wake()
            except Exception: # consumer not ready (Tk not in mainloop yet / shut down)
                pass # -> the consumer's slow backup poll picks the item up


NOT_PASTED = "not pasted" # history note of a kept dictation
MOVED = "Not pasted (the focus moved)"


class State(enum.Enum):
    LOADING = "loading" # model still loading: taps get a "still loading" hint
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"


class RecorderLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...


class TranscriberLike(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class LiveLike(Protocol):
    """Optional: transcriber.live() returns this, recorder.snapshot() feeds it while recording."""
    def feed(self, audio: np.ndarray) -> None: ...
    def finish(self, audio: np.ndarray) -> str: ...
    def has_speech(self, audio: np.ndarray) -> bool: ...


Choice = tuple[str, str | None] # (target, source); source None = auto-detect


class TranslationLike(Protocol):
    """The translate feature: selection capture, language picker and translation model."""
    def capture_selection(self) -> str | None: ... # worker thread; None = nothing selected
    def guess(self, text: str) -> list[str]: ... # worker thread: detected source languages
    def pick(self, on_done: Callable[[Choice | None, bool], None], dictation: bool = False,
             guess: list[str] = ...) -> None: ... # Tk thread
    def set_guess(self, codes: list[str]) -> None: ... # Tk thread: update the open picker
    def confirm_pick(self) -> None: ... # Tk thread: take the highlighted language
    def close_pick(self) -> None: ... # Tk thread: close the picker as if Esc was pressed
    def translate(self, text: str, target: str,
                  source: str | None = None) -> str: ... # worker thread


class _TranslateJob:
    """One translation: the picker's answer, handed from the Tk thread to the worker.

    choice: (target, source or None = auto-detect), or None = not translated. guess: the
    source languages auto-detection found, shown in the picker. focus_back: the picker gave the focus
    back to the app it was opened over (chosen, or Esc), so pasting there is safe; False when
    the user clicked somewhere else.
    """

    def __init__(self, dictation: bool = False) -> None:
        self.dictation = dictation
        self.choice: Choice | None = None
        self.guess: list[str] = []
        self.focus_back = False
        self.shown = False
        self.picked = threading.Event()

    def done(self, choice: Choice | None, focus_back: bool) -> None:
        self.choice, self.focus_back = choice, focus_back
        self.picked.set()


class _LiveRun:
    """While recording, a worker thread hands the audio so far to the live transcription and
    watches for a forgotten recording (no speech for stop_after_s; 0 = never)."""

    def __init__(self, live: LiveLike, snapshot: Callable[[], np.ndarray],
                 stop_after_s: float = 0.0, on_silence: Callable[[], None] = lambda: None) -> None:
        self.live = live
        self._stop_after_s, self._on_silence = stop_after_s, on_silence
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._feed, args=(snapshot,), daemon=True,
                                        name="live-transcription")
        self._thread.start()

    def _feed(self, snapshot: Callable[[], np.ndarray]) -> None:
        feeding, watching = True, self._stop_after_s > 0
        heard = checked = 0 # samples: end of the newest audio with speech / already checked
        while not self._stop.wait(LIVE_FEED_S):
            audio = snapshot()
            if watching:
                try: # only the new audio (+1 s overlap), so this stays cheap
                    if self.live.has_speech(audio[max(0, checked - SAMPLE_RATE):]):
                        heard = len(audio)
                    checked = len(audio)
                except Exception:
                    log.exception("speech check failed, no auto-stop for this recording")
                    watching = False
                if watching and len(audio) - heard >= self._stop_after_s * SAMPLE_RATE:
                    log.info("no speech for %.0f s, stopping the recording", self._stop_after_s)
                    self._on_silence()
                    return
            if feeding:
                try:
                    self.live.feed(audio)
                except Exception: # what wasn't fed is transcribed after the stop instead
                    log.exception("live transcription failed")
                    feeding = False

    def stop(self) -> None:
        """Stop feeding (doesn't wait; finish() does)."""
        self._stop.set()

    def finish(self, audio: np.ndarray) -> str:
        self.stop()
        self._thread.join() # a chunk being transcribed right now is part of the text
        return self.live.finish(audio)


class App:
    """Reacts to events; transcription runs on a worker thread so the loop never blocks."""

    def __init__(
        self,
        recorder: RecorderLike,
        transcriber: TranscriberLike | None,
        paste: Callable[[str], None],
        events: queue.Queue | None = None,
        on_state: Callable[[State, str], None] | None = None, # msg = text to show the user
        postprocess: Callable[[str], str] | None = None, # e.g. spoken formatting commands
        stop_after_silence_s: float = 0.0, # auto-stop a forgotten recording; 0 = never
        # saves each transcript before pasting:
        # (text, translation="", target="", note="", update=False)
        remember: Callable[..., None] | None = None,
        on_pasted: Callable[[str], None] | None = None, # after a paste (Tk thread)
        on_kept: Callable[[str, str], None] | None = None, # (text, message): not pasted
        translation: TranslationLike | None = None, # None = translate key off
        reload: Callable[[], TranscriberLike] | None = None, # loads the model again (worker thread)
    ) -> None:
        """transcriber=None starts in LOADING; call set_transcriber() once it's ready.

        Some dictations are kept instead of pasted: Esc while recording, an auto-stopped
        recording (a minute later you may be typing somewhere else), the picker losing the
        focus, a failed translation. They go to the history and on_kept offers to copy them.
        The app never writes the clipboard by itself: it keeps what the user put there.
        """
        self.recorder = recorder
        self.transcriber = transcriber
        self.paste = paste
        self.events: queue.Queue = events if events is not None else WakeQueue()
        self.on_state = on_state or (lambda state, msg: None)
        self.postprocess = postprocess
        self.stop_after_silence_s = stop_after_silence_s
        self.remember = remember
        self.on_pasted = on_pasted
        self.on_kept = on_kept
        self.translation = translation
        self.reload = reload
        self.state = State.IDLE if transcriber is not None else State.LOADING
        self.exit_reason: str | None = None
        # Extra event kinds (e.g. "model loaded" from the loader thread) -> handler(payload)
        self.handlers: dict[str, Callable[[object], None]] = {}
        self._rec_started = 0.0
        self._live: _LiveRun | None = None
        self._job: _TranslateJob | None = None # translation waiting for the picker
        self._own_events = self._event_table()

    def set_transcriber(self, transcriber: TranscriberLike) -> None:
        """Model is ready. Doesn't call on_state: the caller shows its own "ready" animation."""
        self.transcriber = transcriber
        self.state = State.IDLE

    def _set(self, state: State, msg: str = "") -> None:
        self.state = state
        self.on_state(state, msg)

    def handle(self, kind: str, payload: object = None) -> None:
        """Process one event."""
        handler = self._own_events.get(kind) or self.handlers.get(kind)
        if handler is not None:
            handler(payload)

    def _event_table(self) -> dict[str, Callable[[object], None]]:
        """The App's own event kinds -> handler(payload); `handlers` adds the caller's."""
        return {
            TOGGLE: self._on_toggle,
            TRANSLATE: lambda _payload: self._on_translate_key(),
            PICK: self._show_picker,
            UNPICK: self._on_unpick,
            GUESS: self._on_guess,
            TRANSLATING: lambda _payload: self._set(State.TRANSLATING),
            CANCEL: self._on_cancel,
            AUTO_STOP: self._on_auto_stop,
            DONE: lambda payload: self._set(State.IDLE, str(payload or "")),
            PASTED: self._on_pasted_event,
            KEPT: self._on_kept_event,
            ERROR: lambda payload: self._set(State.IDLE, f"Error: {payload}"),
        }

    def _on_toggle(self, _payload: object) -> None:
        if self.state is State.IDLE:
            self._start_recording()
        elif self.state is State.RECORDING:
            self._stop_recording()
        elif self.state is State.LOADING:
            log.info("toggle ignored: model still loading")
            self.on_state(State.LOADING, "Still loading, one moment…")
        else:
            log.info("toggle ignored while transcribing")

    def _on_unpick(self, job: object) -> None:
        if job is self._job and self._job.shown:
            self.translation.close_pick()

    def _on_guess(self, payload: object) -> None:
        job, codes = payload
        if job is self._job and job.shown and not job.picked.is_set():
            self.translation.set_guess(codes)

    def _on_cancel(self, _payload: object) -> None:
        if self.state is State.RECORDING:
            log.info("recording cancelled")
            self._stop_recording(cancelled=True)

    def _on_auto_stop(self, _payload: object) -> None:
        if self.state is State.RECORDING: # not if a tap stopped it in the meantime
            self._stop_recording(auto=True)

    def _on_pasted_event(self, text: object) -> None:
        self._set(State.IDLE)
        if self.on_pasted is not None:
            self.on_pasted(str(text))

    def _on_kept_event(self, payload: object) -> None:
        text, msg = payload
        self._set(State.IDLE, "" if self.on_kept else msg)
        if self.on_kept is not None:
            self.on_kept(text, msg)

    def _on_translate_key(self) -> None:
        if self.translation is None:
            return
        job = self._job
        if job is not None and job.shown and not job.picked.is_set():
            self.translation.confirm_pick() # tapped again in the picker: take the top language
        elif self.state is State.IDLE:
            self._job = _TranslateJob()
            self.state = State.TRANSLATING # the pill stays hidden until a language is picked
            threading.Thread(target=self._translate_selection, args=(self._job,),
                             daemon=True).start()
        elif self.state is State.RECORDING:
            self._stop_recording(translate=True)
        elif self.state is State.LOADING:
            self.on_state(State.LOADING, "Still loading, one moment…")
        else:
            log.info("translate key ignored while busy")

    def _show_picker(self, job: _TranslateJob) -> None:
        if job is not self._job or job.shown:
            return
        job.shown = True

        def done(choice: Choice | None, focus_back: bool) -> None: # Tk thread
            if self._job is job:
                self._job = None
            job.done(choice, focus_back)

        try:
            self.translation.pick(done, dictation=job.dictation, guess=job.guess)
        except Exception:
            log.exception("could not show the language picker")
            done(None, True)

    def _translate_selection(self, job: _TranslateJob) -> None:
        """Worker thread: copy the selection, wait for the language, translate, paste."""
        try:
            text = self.translation.capture_selection()
        except Exception as e:
            log.exception("could not copy the selection")
            self._job = None
            self.events.put((ERROR, f"could not copy the selection: {e}"))
            return
        if not text or not text.strip():
            self._job = None
            self.events.put((DONE, "Select text to translate, or tap the key while dictating"))
            return
        job.guess = self.translation.guess(text)
        self.events.put((PICK, job))
        job.picked.wait()
        if job.choice is None:
            self.events.put((DONE, "")) # cancelled: the selection is left as it was
            return
        self._translate_and_paste(text, job.choice, keep=not job.focus_back)

    def _translate_and_paste(self, text: str, choice: Choice, keep: bool = False,
                             original_saved: bool = False) -> None:
        """Worker thread. keep: the focus moved, so the translation is kept, not pasted.
        A failed translation of a dictation keeps the original (it is in the history)."""
        target, source = choice
        self.events.put((TRANSLATING, None))
        try:
            t0 = time.perf_counter()
            translated = self.translation.translate(text, target, source)
            log.info("translated %d characters from %s into %s in %.2f s", len(text),
                     source or "auto", target, time.perf_counter() - t0)
        except Exception as e:
            log.exception("translation failed")
            if original_saved:
                self.events.put((KEPT, (text, f"Translation failed: {e}")))
            else:
                self.events.put((ERROR, f"translation failed: {e}"))
            return
        # one entry with original and translation (a dictation's original is saved already)
        self._remember(text, translated, target, NOT_PASTED if keep else "", original_saved)
        if keep:
            self.events.put((KEPT, (translated, MOVED)))
        else:
            self._paste(translated, remember=False)

    def _remember(self, text: str, translation: str = "", target: str = "", note: str = "",
                  update: bool = False) -> None:
        """Worker thread: save to the history. update: add to the newest entry (the same
        dictation, saved before it was translated or kept)."""
        if self.remember is None:
            return
        try:
            self.remember(text, translation=translation, target=target, note=note,
                          update=update)
        except Exception:
            log.exception("could not save the dictation to the history")

    def _paste(self, text: str, remember: bool = True) -> None:
        """Worker thread: save to the history, then paste the text."""
        if remember: # first: if the paste goes nowhere, the text is still in the history
            self._remember(text)
        try:
            self.paste(text)
        except Exception as e: # clipboard can be locked by another app
            log.exception("paste failed")
            self.events.put((ERROR, f"paste failed: {e}"))
            return
        log.info("pasted %d characters", len(text)) # privacy: never the text itself
        self.events.put((PASTED, text))

    def _keep(self, text: str, msg: str, note: str, update: bool = False) -> None:
        """Worker thread: save to the history and offer to copy, without pasting."""
        self._remember(text, note=note, update=update)
        log.info("kept %d characters (%s)", len(text), note)
        self.events.put((KEPT, (text, msg)))

    def _start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as e: # e.g. no microphone
            log.exception("could not start recording")
            self._set(State.IDLE, f"Error: {e}")
            return
        self._rec_started = time.monotonic()
        log.info("recording started")
        make_live = getattr(self.transcriber, "live", None)
        snapshot = getattr(self.recorder, "snapshot", None)
        if make_live is not None and snapshot is not None:
            self._live = _LiveRun(make_live(), snapshot, self.stop_after_silence_s,
                                  lambda: self.events.put((AUTO_STOP, None)))
        self._set(State.RECORDING)

    def _stop_live(self) -> _LiveRun | None:
        live, self._live = self._live, None
        if live is not None:
            live.stop()
        return live

    def _stop_recording(self, auto: bool = False, translate: bool = False,
                        cancelled: bool = False) -> None:
        live = self._stop_live()
        try:
            audio = self.recorder.stop()
        except Exception as e: # must not leave us stuck in RECORDING
            log.exception("could not stop recording")
            self._set(State.IDLE, f"Error: {e}")
            return
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            self._set(State.IDLE, "Cancelled" if cancelled else "Too short")
            return
        self._set(State.TRANSCRIBING)
        job = None
        if translate: # pick the language while the worker transcribes
            job = self._job = _TranslateJob(dictation=True)
            self._show_picker(job)
        threading.Thread(target=self._transcribe, args=(audio, live, auto, job, cancelled),
                         daemon=True).start()

    def _transcribe(self, audio: np.ndarray, live: _LiveRun | None = None,
                    auto: bool = False, job: _TranslateJob | None = None,
                    cancelled: bool = False) -> None:
        """Worker thread: transcribe and paste (or keep, see App). Pasting here (not on the
        Tk thread) keeps the main loop free, so the hotkey thread never waits on it. State
        stays TRANSCRIBING until pasted, so toggles can't interfere."""
        text = self._run_transcriber(audio, live, job)
        if text is None:
            return
        stopped = f"Stopped after {self.stop_after_silence_s:.0f} s of silence"
        if not text:
            if job is not None:
                self.events.put((UNPICK, job))
            self.events.put((DONE, "Cancelled" if cancelled else
                             stopped if auto else "Nothing recognized"))
            return
        text = self._postprocessed(text)
        if cancelled:
            self._keep(text, "Cancelled", "cancelled")
        elif job is not None:
            self._translate_dictation(text, job)
        elif auto:
            self._keep(text, stopped, NOT_PASTED)
        else:
            self._paste(text)

    def _run_transcriber(self, audio: np.ndarray, live: _LiveRun | None,
                         job: _TranslateJob | None) -> str | None:
        """Worker thread: the transcript ("" = nothing recognized). None = it failed; the
        error is reported and the picker closed."""
        try:
            t0 = time.perf_counter()
            try:
                text = live.finish(audio) if live else self.transcriber.transcribe(audio)
            except Exception:
                if self.reload is None:
                    raise
                text = self._reload_and_retry(audio)
            early = getattr(live.live, "done_s", 0.0) if live else 0.0
            log.info("transcribed %.1f s audio in %.2f s after the stop (%.0f s while recording)",
                     len(audio) / SAMPLE_RATE, time.perf_counter() - t0, early)
            return text or ""
        except Exception as e:
            log.exception("transcription failed")
            if job is not None:
                self.events.put((UNPICK, job))
            self.events.put((ERROR, str(e)))
            return None

    def _reload_and_retry(self, audio: np.ndarray) -> str:
        """Worker thread: the model failed (e.g. the GPU driver was reset); load it again (on the
        GPU if it is there, else the fallback) and transcribe the same audio, so the dictation
        is not lost. Safe here: in TRANSCRIBING nothing else uses the transcriber."""
        log.exception("transcription failed; loading the model again and retrying once")
        self.transcriber = self.reload()
        self.events.put((RELOADED, self.transcriber))
        return self.transcriber.transcribe(audio)

    def _postprocessed(self, text: str) -> str:
        """Spoken commands, if on. Judged on the raw text before: dictating only "command new
        line" is valid."""
        if self.postprocess is None:
            return text
        try:
            return self.postprocess(text)
        except Exception:
            log.exception("postprocessing failed, pasting the raw transcript")
            return text

    def _translate_dictation(self, text: str, job: _TranslateJob) -> None:
        """Worker thread: wait for the picker, then translate and paste the dictation."""
        self._remember(text) # the original is kept, whatever happens to the translation
        if not job.picked.is_set():
            self.events.put((GUESS, (job, self.translation.guess(text))))
        job.picked.wait()
        if job.choice is not None:
            self._translate_and_paste(text, job.choice, keep=not job.focus_back,
                                      original_saved=True)
        elif job.focus_back: # Esc in the picker: paste the dictation as it is
            self._paste(text, remember=False)
        else: # clicked somewhere else
            self._keep(text, MOVED, NOT_PASTED, update=True)

    def process_pending(self) -> bool:
        """Handle all queued events without blocking (called from the Tk loop).

        Returns False once QUIT or RESTART arrives; exit_reason then says which.
        """
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return True
            if kind in (QUIT, RESTART):
                self.exit_reason = kind
                if self.state is State.RECORDING:
                    self._stop_live()
                    self.recorder.stop() # release the mic
                return False
            self.handle(kind, payload)
