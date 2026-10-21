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
from faster_whisper.vad import VadOptions, get_speech_timestamps

from whisperclaude.constants import DEFAULT_PROMPT, DEFAULT_VOCABULARY, SAMPLE_RATE  # noqa: F401 (re-exported)
from whisperclaude.textfix import join_chunks

log = logging.getLogger(__name__)

CHUNK_S = 30  # Whisper's window
LIVE_MARGIN_S = 5  # live transcription leaves the newest audio alone (the VAD may still change it)
OV_CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude" / "ov_cache"


def build_prompt(prompt: str | None, vocabulary: Sequence[str] = ()) -> str | None:
    """Append custom words so Whisper spells them right ("Klot" -> "Claude").

    Tested against faster-whisper's `hotwords`: same fixes, but hotwords cost commas.
    """
    words = ", ".join(w.strip() for w in vocabulary if w.strip())
    if not words:
        return prompt or None
    return f"{prompt} {words}." if prompt else f"{words}."


def _speech_groups(audio: np.ndarray) -> list[list[dict]]:
    """VAD speech segments ({"start", "end"} in samples), grouped into <= 30 s of speech.

    Same greedy rule as faster-whisper's collect_chunks, whose metadata can't be used here: it
    leaves out the first segment of every chunk after the first.
    """
    groups: list[list[dict]] = []
    size = 0
    for seg in get_speech_timestamps(audio, VadOptions(max_speech_duration_s=CHUNK_S)):
        n = seg["end"] - seg["start"]
        if groups and size + n <= CHUNK_S * SAMPLE_RATE:
            groups[-1].append(seg)
            size += n
        else:
            groups.append([seg])
            size = n
    return groups


def _join(audio: np.ndarray, group: list[dict]) -> np.ndarray:
    return np.concatenate([audio[s["start"]:s["end"]] for s in group])


def speech_chunks(audio: np.ndarray) -> list[np.ndarray]:
    """Cut audio at speech pauses into <= 30 s chunks (silence dropped).

    Same as faster-whisper's batched pipeline: transcribing chunks without timestamps
    avoids both the capitals-after-pauses of timestamp mode and words cut at 30 s borders.
    """
    return [_join(audio, g) for g in _speech_groups(audio)]


class LiveTranscription:
    """Transcribes a recording while it is still running, so stopping only waits for the end.

    feed() gets the audio recorded so far (call it every few seconds from a worker thread). A
    chunk is final once the VAD has started a later chunk after it: more audio can't change it
    any more, so it is transcribed right away. finish() transcribes the rest. Long dictations
    (minutes) then take ~1 s after the stop tap instead of ~1 s per 30 s of speech. Not
    thread-safe: feed() and finish() must not overlap.
    """

    def __init__(self, transcribe_chunks: Callable[[list[np.ndarray]], str]) -> None:
        self._transcribe_chunks = transcribe_chunks
        self._done = 0  # samples already transcribed
        self._texts: list[str] = []

    @property
    def done_s(self) -> float:
        return self._done / SAMPLE_RATE

    def feed(self, audio: np.ndarray) -> None:
        tail = audio[self._done:]
        if len(tail) < (CHUNK_S + LIVE_MARGIN_S) * SAMPLE_RATE:
            return  # a finished chunk needs > 30 s of speech after self._done
        limit = len(tail) - LIVE_MARGIN_S * SAMPLE_RATE
        final = [g for g in _speech_groups(tail)[:-1] if g[-1]["end"] <= limit]
        if final:
            self._texts.append(self._transcribe_chunks([_join(tail, g) for g in final]))
            self._done += final[-1][-1]["end"]

    @staticmethod
    def has_speech(audio: np.ndarray) -> bool:
        """VAD check on a short piece of audio (for the forgotten-recording auto-stop)."""
        return bool(get_speech_timestamps(np.asarray(audio, dtype=np.float32), VadOptions()))

    def finish(self, audio: np.ndarray) -> str:
        """Transcribe what feed() hasn't and return the whole text."""
        rest = speech_chunks(np.asarray(audio[self._done:], dtype=np.float32))
        if rest:
            self._texts.append(self._transcribe_chunks(rest))
        return join_chunks(self._texts)


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
        text = join_chunks(s.text for s in segments)
        return text, info.language

    def transcribe(self, audio: np.ndarray) -> str:
        """Return the transcript of a 1-D float32 array at 16 kHz."""
        return self.transcribe_detailed(audio)[0]

    def transcribe_chunks(self, chunks: list[np.ndarray]) -> str:
        """Transcribe chunks that are already cut (speech_chunks), batched like transcribe()."""
        if not chunks:
            return ""
        bounds = np.cumsum([0] + [len(c) for c in chunks]) / SAMPLE_RATE
        segments, _ = self.pipeline.transcribe(
            np.concatenate(chunks).astype(np.float32),
            language=self.language,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            initial_prompt=self.initial_prompt,
            without_timestamps=True,
            clip_timestamps=[{"start": float(a), "end": float(b)}
                             for a, b in zip(bounds[:-1], bounds[1:])],
        )
        return join_chunks(s.text for s in segments)

    def live(self) -> LiveTranscription:
        return LiveTranscription(self.transcribe_chunks)


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
        # The first dictation was ~0.8 s slower (GPU setup in the first generate(), VAD model
        # loaded on first use): pay that here, on the loader thread.
        silence = np.zeros(SAMPLE_RATE, np.float32)
        speech_chunks(silence)
        self._generate([silence])

    def _generate(self, chunks: list[np.ndarray]) -> tuple[str, str]:
        cfg = self.pipeline.get_generation_config()
        cfg.task = "transcribe"
        cfg.return_timestamps = False
        cfg.initial_prompt = self.initial_prompt
        if self.language:
            cfg.language = f"<|{self.language}|>"
        texts, language = [], ""
        for chunk in chunks:
            result = self.pipeline.generate(chunk, cfg)
            texts.append(result.texts[0].strip())
            language = language or str(result.language or "").strip("<|>")
        return join_chunks(texts), language

    def transcribe_detailed(self, audio: np.ndarray) -> tuple[str, str]:
        return self._generate(speech_chunks(np.asarray(audio, dtype=np.float32).reshape(-1)))

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_detailed(audio)[0]

    def transcribe_chunks(self, chunks: list[np.ndarray]) -> str:
        return self._generate(chunks)[0]

    def live(self) -> LiveTranscription:
        return LiveTranscription(self.transcribe_chunks)


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
