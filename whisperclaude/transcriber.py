"""faster-whisper wrapper: load the model once, transcribe 16 kHz mono float32 audio."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000


class Transcriber:
    """Keeps one Whisper model warm and turns audio arrays into text.

    Defaults come from the step-1 benchmark (docs/PLAN.md): small, greedy decoding,
    4 threads (8 threads was slower for small on the P/E-core CPU).
    """

    def __init__(
        self,
        model: str = "small",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str | None = None,
        cpu_threads: int = 4,
    ) -> None:
        self.beam_size = beam_size
        self.language = language  # None = auto-detect
        self.model = WhisperModel(
            model, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
        )

    def transcribe_detailed(self, audio: np.ndarray) -> tuple[str, str]:
        """Return (text, detected_language) for a 1-D float32 array at 16 kHz."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        # segments is a lazy generator: the actual decoding happens here.
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language

    def transcribe(self, audio: np.ndarray) -> str:
        """Return the transcript of a 1-D float32 array at 16 kHz."""
        return self.transcribe_detailed(audio)[0]
