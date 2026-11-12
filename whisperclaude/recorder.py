"""Microphone recording: start() opens the stream, stop() returns the audio."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from whisperclaude.constants import SAMPLE_RATE

if TYPE_CHECKING: # sounddevice initialises PortAudio on import (~120 ms): load it only when audio is used
    import sounddevice as sd

log = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 5.0 # how long start() / refresh_devices() wait for a pending background close


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
        self._closer: threading.Thread | None = None # background close of the last stream
        self.level = 0.0 # RMS of the latest audio block, for the overlay's level bars
        self._on_first_block: Callable[[], None] | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata: np.ndarray, frames: int, time: object,
                  status: sd.CallbackFlags) -> None:
        if status:
            log.warning("audio status: %s", status)
        first, self._on_first_block = self._on_first_block, None
        block = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        with self._lock:
            self._chunks.append(block)
        if first is not None:
            first() # runs on the audio thread: the app only sets an Event here

    def _open(self) -> sd.InputStream:
        import sounddevice as sd

        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        stream.start()
        return stream

    def _close(self, stream: sd.InputStream) -> None:
        try:
            stream.close()
        except Exception:
            log.warning("closing the microphone failed", exc_info=True)

    def _wait_for_close(self) -> None:
        """Wait for the background close of the previous stream (PortAudio must not be
        re-initialised or reopened while it is still closing)."""
        closer, self._closer = self._closer, None
        if closer is None:
            return
        closer.join(CLOSE_TIMEOUT_S)
        if closer.is_alive():
            log.warning("the microphone is still closing after %.0f s, continuing anyway",
                        CLOSE_TIMEOUT_S)

    def refresh_devices(self) -> None:
        """Re-scan audio devices. PortAudio lists them once at import, so a device that
        appeared or changed later (new logon session, wake from sleep, headset plugged in)
        stays invisible or stale until re-initialised. Not while recording."""
        if self._stream is not None:
            return
        self._wait_for_close()
        try:
            import sounddevice as sd

            sd._terminate()
            sd._initialize()
            log.info("audio devices refreshed")
        except Exception:
            log.exception("could not refresh audio devices")

    def start(self, on_first_block: Callable[[], None] | None = None) -> None:
        """Open the mic and start collecting audio. Raises if no input device works.

        Blocks until the stream runs (0.3-0.7 s after an idle spell), so call it off the UI
        thread. on_first_block is called once, from the audio thread, when the first block of
        audio has arrived: only from then on is anything recorded.
        """
        if self._stream is not None:
            if on_first_block is not None:
                on_first_block() # already running: audio is flowing
            return
        self._wait_for_close()
        self._chunks = []
        self._on_first_block = on_first_block
        import sounddevice as sd

        try:
            self._stream = self._open()
        except sd.PortAudioError as e:
            log.warning("opening the microphone failed (%s), refreshing devices and retrying", e)
            self.refresh_devices()
            self._stream = self._open()

    def snapshot(self) -> np.ndarray:
        """Everything recorded so far, while recording continues (thread-safe)."""
        with self._lock:
            chunks = list(self._chunks)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stop(self) -> np.ndarray:
        """Stop the mic and return everything recorded as a 1-D float32 array.

        stream.stop() runs here so PortAudio delivers the last captured blocks before the audio
        is collected. stream.close() (slow) runs on a background thread; the next start() or
        refresh_devices() waits for it.

        If stopping fails (e.g. the headset was unplugged mid-recording), the error is logged and
        the audio recorded until then is still returned; the stream is closed right away then.
        The next start() opens a new stream.
        """
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        stream, self._stream = self._stream, None
        self.level = 0.0
        self._on_first_block = None
        try:
            stream.stop()
        except Exception:
            log.warning("closing the microphone failed", exc_info=True)
            self._close(stream)
        else:
            self._closer = threading.Thread(target=self._close, args=(stream,),
                                            name="microphone-close", daemon=True)
            self._closer.start()
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
