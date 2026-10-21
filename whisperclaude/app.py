"""State machine: LOADING -> IDLE -> RECORDING -> TRANSCRIBING -> IDLE, driven by queue events.

Hardware-free (recorder, transcriber and paste are injected), so it can be tested.
"""
from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from typing import Callable, Protocol

import numpy as np

from whisperclaude.constants import SAMPLE_RATE

log = logging.getLogger(__name__)

# Event kinds (queue items are (kind, payload) tuples)
TOGGLE = "toggle"
DONE = "done"    # payload: message for the user ("" = pasted fine)
ERROR = "error"  # payload: message
QUIT = "quit"
RESTART = "restart"  # quit and start a new instance (applies config.toml changes)
AUTO_STOP = "auto_stop"  # a recording was silent for too long (probably forgotten)
CANCEL = "cancel"  # the cancel key (Esc): discard a running recording, ignored otherwise

MIN_AUDIO_S = 0.3
LIVE_FEED_S = 5.0  # while recording, hand the audio so far to the live transcription this often


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

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        self._pending.set()

    def _waker(self) -> None:
        while True:
            self._pending.wait()
            self._pending.clear()  # clear first: a put during wake() sets it again
            try:
                self.wake()
            except Exception:  # consumer not ready (Tk not in mainloop yet / shut down)
                pass           # -> the consumer's slow backup poll picks the item up


class State(enum.Enum):
    LOADING = "loading"  # model still loading: taps get a "still loading" hint
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


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
        heard = checked = 0  # samples: end of the newest audio with speech / already checked
        while not self._stop.wait(LIVE_FEED_S):
            audio = snapshot()
            if watching:
                try:  # only the new audio (+1 s overlap), so this stays cheap
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
                except Exception:  # what wasn't fed is transcribed after the stop instead
                    log.exception("live transcription failed")
                    feeding = False

    def stop(self) -> None:
        """Stop feeding (doesn't wait; finish() does)."""
        self._stop.set()

    def finish(self, audio: np.ndarray) -> str:
        self.stop()
        self._thread.join()  # a chunk being transcribed right now is part of the text
        return self.live.finish(audio)


class App:
    """Reacts to events; transcription runs on a worker thread so the loop never blocks."""

    def __init__(
        self,
        recorder: RecorderLike,
        transcriber: TranscriberLike | None,
        paste: Callable[[str], None],
        events: queue.Queue | None = None,
        on_state: Callable[[State, str], None] | None = None,  # msg = text to show the user
        postprocess: Callable[[str], str] | None = None,  # e.g. spoken formatting commands
        stop_after_silence_s: float = 0.0,  # auto-stop a forgotten recording; 0 = never
        copy: Callable[[str], None] | None = None,  # puts text in the clipboard (no paste)
    ) -> None:
        """transcriber=None starts in LOADING; call set_transcriber() once it's ready.

        An auto-stopped recording isn't pasted but copied (if `copy` is given): a minute later
        you may be typing somewhere else, and text appearing at the cursor would be a surprise.
        """
        self.recorder = recorder
        self.transcriber = transcriber
        self.paste = paste
        self.events: queue.Queue = events if events is not None else WakeQueue()
        self.on_state = on_state or (lambda state, msg: None)
        self.postprocess = postprocess
        self.stop_after_silence_s = stop_after_silence_s
        self.copy = copy
        self.state = State.IDLE if transcriber is not None else State.LOADING
        self.exit_reason: str | None = None
        # Extra event kinds (e.g. "model loaded" from the loader thread) -> handler(payload)
        self.handlers: dict[str, Callable[[object], None]] = {}
        self._rec_started = 0.0
        self._live: _LiveRun | None = None

    def set_transcriber(self, transcriber: TranscriberLike) -> None:
        """Model is ready. Doesn't call on_state: the caller shows its own "ready" animation."""
        self.transcriber = transcriber
        self.state = State.IDLE

    def _set(self, state: State, msg: str = "") -> None:
        self.state = state
        self.on_state(state, msg)

    def handle(self, kind: str, payload: object = None) -> None:
        """Process one event."""
        if kind == TOGGLE:
            if self.state is State.IDLE:
                self._start_recording()
            elif self.state is State.RECORDING:
                self._stop_recording()
            elif self.state is State.LOADING:
                log.info("toggle ignored: model still loading")
                self.on_state(State.LOADING, "Still loading – one moment…")
            else:
                log.info("toggle ignored while transcribing")
        elif kind == CANCEL:
            if self.state is State.RECORDING:
                self._stop_live()
                try:
                    self.recorder.stop()
                except Exception:
                    log.exception("could not stop recording")
                log.info("recording cancelled")
                self._set(State.IDLE, "Cancelled")
        elif kind == AUTO_STOP:
            if self.state is State.RECORDING:  # not if a tap stopped it in the meantime
                self._stop_recording(auto=True)
        elif kind == DONE:
            self._set(State.IDLE, str(payload or ""))
        elif kind == ERROR:
            self._set(State.IDLE, f"Error: {payload}")
        elif kind in self.handlers:
            self.handlers[kind](payload)

    def _start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as e:  # e.g. no microphone
            log.exception("could not start recording")
            self._set(State.IDLE, f"Error: {e}")
            return
        self._rec_started = time.monotonic()
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

    def _stop_recording(self, auto: bool = False) -> None:
        live = self._stop_live()
        try:
            audio = self.recorder.stop()
        except Exception as e:  # must not leave us stuck in RECORDING
            log.exception("could not stop recording")
            self._set(State.IDLE, f"Error: {e}")
            return
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            self._set(State.IDLE, "Too short")
            return
        self._set(State.TRANSCRIBING)
        threading.Thread(target=self._transcribe, args=(audio, live, auto),
                         daemon=True).start()

    def _transcribe(self, audio: np.ndarray, live: _LiveRun | None = None,
                    auto: bool = False) -> None:
        """Worker thread: transcribe and paste. Pasting here (not on the Tk thread) keeps
        the main loop free, so the hotkey thread never waits on it. State stays
        TRANSCRIBING until pasted, so toggles can't interfere."""
        try:
            t0 = time.perf_counter()
            text = live.finish(audio) if live else self.transcriber.transcribe(audio)
            early = getattr(live.live, "done_s", 0.0) if live else 0.0
            log.info("transcribed %.1f s audio in %.2f s after the stop (%.0f s while recording)",
                     len(audio) / SAMPLE_RATE, time.perf_counter() - t0, early)
        except Exception as e:
            log.exception("transcription failed")
            self.events.put((ERROR, str(e)))
            return
        stopped = f"Stopped after {self.stop_after_silence_s:.0f} s of silence"
        if not text:
            self.events.put((DONE, stopped if auto else "Nothing recognized"))
            return
        if self.postprocess is not None:
            try:  # judged on the raw text above: dictating only "command new line" is valid
                text = self.postprocess(text)
            except Exception:
                log.exception("postprocessing failed, pasting the raw transcript")
        to_clipboard = auto and self.copy is not None
        try:
            (self.copy if to_clipboard else self.paste)(text)
        except Exception as e:  # clipboard can be locked by another app
            log.exception("paste failed")
            self.events.put((ERROR, f"paste failed: {e}"))
            return
        log.info("%s %d characters", "copied" if to_clipboard else "pasted", len(text))  # privacy
        self.events.put((DONE, f"{stopped} – text is in the clipboard" if to_clipboard else ""))

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
                    self.recorder.stop()  # release the mic
                return False
            self.handle(kind, payload)
