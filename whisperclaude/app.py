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
DONE = "done"    # payload: transcript text
ERROR = "error"  # payload: message
QUIT = "quit"

MIN_AUDIO_S = 0.3


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
        on_state: Callable[[State, str], None] | None = None,
    ) -> None:
        self.recorder = recorder
        self.transcriber = transcriber
        self.paste = paste
        self.events: queue.Queue = events if events is not None else queue.Queue()
        self.on_state = on_state or (lambda state, msg: None)
        self.state = State.IDLE
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
            text = str(payload or "")
            if text:
                try:
                    self.paste(text)
                except Exception as e:  # clipboard can be locked by another app
                    log.exception("paste failed")
                    self._set(State.IDLE, f"Paste failed: {e}")
                    return
            self._set(State.IDLE, text or "(nothing recognized)")
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
            self._set(State.IDLE, "(too short)")
            return
        self._set(State.TRANSCRIBING, f"{len(audio) / SAMPLE_RATE:.1f} s audio")
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray) -> None:
        try:
            t0 = time.perf_counter()
            text = self.transcriber.transcribe(audio)
            log.info("transcribed %.1f s audio in %.2f s", len(audio) / SAMPLE_RATE,
                     time.perf_counter() - t0)
            self.events.put((DONE, text))
        except Exception as e:
            log.exception("transcription failed")
            self.events.put((ERROR, str(e)))

    def run(self) -> None:
        """Blocking event loop (console version). Ends on a QUIT event or Ctrl+C."""
        while True:
            try:
                kind, payload = self.events.get(timeout=0.2)  # timeout keeps Ctrl+C responsive
            except queue.Empty:
                continue
            if kind == QUIT:
                return
            self.handle(kind, payload)
