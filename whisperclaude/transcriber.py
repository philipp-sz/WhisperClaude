"""Speech-to-text backends: load the model once, transcribe 16 kHz mono float32 audio.

- OpenVinoTranscriber: Whisper on the Intel iGPU via OpenVINO GenAI (~10x faster, ~half the
  energy per dictation, and fast enough for large-v3-turbo; see docs/PLAN.md).
- Transcriber: faster-whisper on the CPU (fallback when there's no usable GPU).
create_transcriber() picks one based on the hardware.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from whisperclaude.constants import DEFAULT_PROMPT, DEFAULT_VOCABULARY, SAMPLE_RATE  # noqa: F401 (re-exported)

log = logging.getLogger(__name__)

CHUNK_S = 30  # Whisper's window
OV_CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude" / "ov_cache"


def build_prompt(prompt: str | None, vocabulary: Sequence[str] = ()) -> str | None:
    """Append custom words so Whisper spells them right ("Klot" -> "Claude").

    Tested against faster-whisper's `hotwords`: same fixes, but hotwords cost commas.
    """
    words = ", ".join(w.strip() for w in vocabulary if w.strip())
    if not words:
        return prompt or None
    return f"{prompt} {words}." if prompt else f"{words}."


def speech_chunks(audio: np.ndarray) -> list[np.ndarray]:
    """Cut audio at speech pauses into <= 30 s chunks (silence dropped).

    Same as faster-whisper's batched pipeline: transcribing chunks without timestamps
    avoids both the capitals-after-pauses of timestamp mode and words cut at 30 s borders.
    """
    speech = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=CHUNK_S))
    if not speech:
        return []
    chunks, _ = collect_chunks(audio, speech, max_duration=CHUNK_S)
    return [c for c in chunks if len(c)]


class Transcriber:
    """faster-whisper on the CPU.

    Defaults come from the benchmarks in docs/PLAN.md: small, greedy decoding, 4 threads
    (8 was slower for small on the P/E-core CPU).
    """

    device = "CPU"

    def __init__(
        self,
        model: str = "small",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str | None = None,
        cpu_threads: int = 4,
        initial_prompt: str | None = DEFAULT_PROMPT,
        batch_size: int = 4,
        vocabulary: Sequence[str] = (),
    ) -> None:
        self.beam_size = beam_size
        self.language = language  # None = auto-detect
        self.initial_prompt = build_prompt(initial_prompt, vocabulary)
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


def ov_repo(model: str) -> str:
    """Pre-converted OpenVINO Whisper on Hugging Face, e.g. large-v3-turbo -> int8 IR."""
    return f"OpenVINO/whisper-{model}-int8-ov"


def gpu_available() -> bool:
    """True if OpenVINO is installed and sees a GPU (e.g. the Intel iGPU)."""
    try:
        import openvino as ov
        return "GPU" in ov.Core().available_devices
    except Exception:  # not installed, driver problem, ...
        log.info("OpenVINO GPU check failed", exc_info=True)
        return False


class OpenVinoTranscriber:
    """Whisper on an OpenVINO device (the iGPU). Same chunking/prompt as the CPU path."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "GPU",
        language: str | None = None,
        initial_prompt: str | None = DEFAULT_PROMPT,
        vocabulary: Sequence[str] = (),
        on_status: Callable[[str], None] = lambda s: None,
    ) -> None:
        import openvino_genai
        from huggingface_hub import snapshot_download

        self.device = device
        self.language = language
        self.initial_prompt = build_prompt(initial_prompt, vocabulary)
        repo = ov_repo(model)
        try:  # cached: no network request at startup
            path = snapshot_download(repo, local_files_only=True)
        except Exception:
            on_status("Downloading model (one-time)…")
            path = snapshot_download(repo)
        if not OV_CACHE.exists() or not any(OV_CACHE.iterdir()):
            on_status("Preparing GPU (one-time, ~15 s)…")
        OV_CACHE.mkdir(parents=True, exist_ok=True)
        # CACHE_DIR keeps the compiled GPU kernels: first start ~15 s, later < 1 s.
        self.pipeline = openvino_genai.WhisperPipeline(path, device, CACHE_DIR=str(OV_CACHE))

    def transcribe_detailed(self, audio: np.ndarray) -> tuple[str, str]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        cfg = self.pipeline.get_generation_config()
        cfg.task = "transcribe"
        cfg.return_timestamps = False
        cfg.initial_prompt = self.initial_prompt
        if self.language:
            cfg.language = f"<|{self.language}|>"
        texts, language = [], ""
        for chunk in speech_chunks(audio):
            result = self.pipeline.generate(chunk, cfg)
            texts.append(result.texts[0].strip())
            language = language or str(result.language or "").strip("<|>")
        return " ".join(texts).strip(), language

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_detailed(audio)[0]


def create_transcriber(
    device: str = "auto",
    gpu_model: str = "large-v3-turbo",
    cpu_model: str = "small",
    on_status: Callable[[str], None] = lambda s: None,
    language: str | None = None,
    initial_prompt: str | None = DEFAULT_PROMPT,
    vocabulary: Sequence[str] = (),
    **cpu_kwargs,
) -> Transcriber | OpenVinoTranscriber:
    """device "auto": iGPU if OpenVINO sees one, else CPU. "gpu": iGPU, CPU if that fails.
    "cpu": always faster-whisper on the CPU."""
    common = dict(language=language, initial_prompt=initial_prompt, vocabulary=vocabulary)
    if device in ("auto", "gpu"):
        if gpu_available():
            try:
                return OpenVinoTranscriber(gpu_model, "GPU", on_status=on_status, **common)
            except Exception:
                log.exception("GPU backend failed, falling back to CPU")
        else:
            log.info("no OpenVINO GPU found, using CPU")
    on_status("Loading model…")
    return Transcriber(cpu_model, **common, **cpu_kwargs)
