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

MIN_AUDIO_S = 0.3


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
    ) -> None:
        """transcriber=None starts in LOADING; call set_transcriber() once it's ready."""
        self.recorder = recorder
        self.transcriber = transcriber
        self.paste = paste
        self.events: queue.Queue = events if events is not None else WakeQueue()
        self.on_state = on_state or (lambda state, msg: None)
        self.postprocess = postprocess
        self.state = State.IDLE if transcriber is not None else State.LOADING
        self.exit_reason: str | None = None
        # Extra event kinds (e.g. "model loaded" from the loader thread) -> handler(payload)
        self.handlers: dict[str, Callable[[object], None]] = {}
        self._rec_started = 0.0

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
        self._set(State.RECORDING)

    def _stop_recording(self) -> None:
        audio = self.recorder.stop()
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            self._set(State.IDLE, "Too short")
            return
        self._set(State.TRANSCRIBING)
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray) -> None:
        """Worker thread: transcribe and paste. Pasting here (not on the Tk thread) keeps
        the main loop free, so the hotkey thread never waits on it. State stays
        TRANSCRIBING until pasted, so toggles can't interfere."""
        try:
            t0 = time.perf_counter()
            text = self.transcriber.transcribe(audio)
            log.info("transcribed %.1f s audio in %.2f s", len(audio) / SAMPLE_RATE,
                     time.perf_counter() - t0)
        except Exception as e:
            log.exception("transcription failed")
            self.events.put((ERROR, str(e)))
            return
        if not text:
            self.events.put((DONE, "Nothing recognized"))
            return
        if self.postprocess is not None:
            try:  # judged on the raw text above: dictating only "command new line" is valid
                text = self.postprocess(text)
            except Exception:
                log.exception("postprocessing failed, pasting the raw transcript")
        try:
            self.paste(text)
        except Exception as e:  # clipboard can be locked by another app
            log.exception("paste failed")
            self.events.put((ERROR, f"paste failed: {e}"))
            return
        log.info("pasted %d characters", len(text))  # not the text itself: privacy
        self.events.put((DONE, ""))

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
                    self.recorder.stop()  # release the mic
                return False
            self.handle(kind, payload)
