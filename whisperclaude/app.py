"""State machine: IDLE -> RECORDING -> TRANSCRIBING -> IDLE, driven by queue events.

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

from whisperclaude.transcriber import SAMPLE_RATE

log = logging.getLogger(__name__)

# Event kinds (queue items are (kind, payload) tuples)
TOGGLE = "toggle"
DONE = "done"    # payload: message for the user ("" = pasted fine)
ERROR = "error"  # payload: message
QUIT = "quit"
RESTART = "restart"  # quit and start a new instance (applies config.toml changes)

MIN_AUDIO_S = 0.3


class WakeQueue(queue.Queue):
    """Queue that calls wake() after each put, so the consumer can sleep instead of polling
    (polling every 50 ms cost ~0.8 % of a core while idle)."""

    def __init__(self) -> None:
        super().__init__()
        self.wake: Callable[[], None] = lambda: None

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        try:
            self.wake()
        except Exception:  # consumer not ready (Tk not in mainloop yet / shut down)
            pass           # -> the consumer's slow backup poll picks the item up


class State(enum.Enum):
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
        transcriber: TranscriberLike,
        paste: Callable[[str], None],
        events: queue.Queue | None = None,
        on_state: Callable[[State, str], None] | None = None,  # msg = text to show the user
    ) -> None:
        self.recorder = recorder
        self.transcriber = transcriber
        self.paste = paste
        self.events: queue.Queue = events if events is not None else WakeQueue()
        self.on_state = on_state or (lambda state, msg: None)
        self.state = State.IDLE
        self.exit_reason: str | None = None
        self._rec_started = 0.0

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
            else:
                log.info("toggle ignored while transcribing")
        elif kind == DONE:
            self._set(State.IDLE, str(payload or ""))
        elif kind == ERROR:
            self._set(State.IDLE, f"Error: {payload}")

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
