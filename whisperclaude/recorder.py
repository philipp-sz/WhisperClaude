"""Microphone recording: start() opens the stream, stop() returns the audio."""
from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from whisperclaude.constants import SAMPLE_RATE

log = logging.getLogger(__name__)


class Recorder:
    """Records 16 kHz mono float32 audio between start() and stop().

    The stream is only open while recording, so Windows' mic indicator
    isn't permanently on.
    """

    def __init__(self, samplerate: int = SAMPLE_RATE, device: int | str | None = None) -> None:
        self.samplerate = samplerate
        self.device = device
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self.level = 0.0  # RMS of the latest audio block, for the overlay's level bars

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        if status:
            log.warning("audio status: %s", status)
        block = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        with self._lock:
            self._chunks.append(block)

    def _open(self) -> sd.InputStream:
        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        stream.start()
        return stream

    def refresh_devices(self) -> None:
        """Re-scan audio devices. PortAudio lists them once at import, so a device that
        appeared or changed later (new logon session, wake from sleep, headset plugged in)
        stays invisible or stale until re-initialised. Not while recording."""
        if self._stream is not None:
            return
        try:
            sd._terminate()
            sd._initialize()
            log.info("audio devices refreshed")
        except Exception:
            log.exception("could not refresh audio devices")

    def start(self) -> None:
        """Open the mic and start collecting audio. Raises if no input device works."""
        if self._stream is not None:
            return
        self._chunks = []
        try:
            self._stream = self._open()
        except sd.PortAudioError as e:
            log.warning("opening the microphone failed (%s), refreshing devices and retrying", e)
            self.refresh_devices()
            self._stream = self._open()

    def stop(self) -> np.ndarray:
        """Close the mic and return everything recorded as a 1-D float32 array."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.level = 0.0
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
