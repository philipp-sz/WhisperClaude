"""faster-whisper wrapper: load the model once, transcribe 16 kHz mono float32 audio."""
from __future__ import annotations

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

SAMPLE_RATE = 16_000

# Correctly punctuated bilingual "previous text": Whisper copies its style. Fixes missing
# punctuation, capitals after pauses, and German+English audio being translated to one language.
DEFAULT_PROMPT = "Hallo, das ist ein Test. Okay, so let's see how this works, and then we'll decide."


class Transcriber:
    """Keeps one Whisper model warm and turns audio arrays into text.

    Defaults come from the benchmarks in docs/PLAN.md: small, greedy decoding, 4 threads
    (8 was slower for small on the P/E-core CPU). The batched pipeline cuts long audio at
    speech pauses into <= 30 s chunks, so without_timestamps can't cut words at the
    30 s window boundary.
    """

    def __init__(
        self,
        model: str = "small",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str | None = None,
        cpu_threads: int = 4,
        initial_prompt: str | None = DEFAULT_PROMPT,
        batch_size: int = 4,
    ) -> None:
        self.beam_size = beam_size
        self.language = language  # None = auto-detect
        self.initial_prompt = initial_prompt
        self.batch_size = batch_size
        kwargs = dict(device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
        try:  # cached model: no network request at startup (private, works offline)
            self.model = WhisperModel(model, local_files_only=True, **kwargs)
        except Exception:  # first run: download
            self.model = WhisperModel(model, **kwargs)
        self.pipeline = BatchedInferencePipeline(self.model)

    def transcribe_detailed(self, audio: np.ndarray) -> tuple[str, str]:
        """Return (text, detected_language) for a 1-D float32 array at 16 kHz."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments, info = self.pipeline.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            initial_prompt=self.initial_prompt,
            without_timestamps=True,
            vad_filter=True,
        )
        # segments is a lazy generator: the actual decoding happens here.
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language

    def transcribe(self, audio: np.ndarray) -> str:
        """Return the transcript of a 1-D float32 array at 16 kHz."""
        return self.transcribe_detailed(audio)[0]
