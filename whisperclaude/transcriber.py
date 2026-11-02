"""Speech-to-text backends: load the model once, transcribe 16 kHz mono float32 audio.

- OpenVinoTranscriber: Whisper on the Intel iGPU via OpenVINO GenAI (~10x faster, ~half the
  energy per dictation, and fast enough for large-v3-turbo; see docs/transcription.md).
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

from whisperclaude.constants import DEFAULT_PROMPT, DEFAULT_VOCABULARY, SAMPLE_RATE # noqa: F401 (re-exported)
from whisperclaude.textfix import join_chunks

log = logging.getLogger(__name__)

CHUNK_S = 30 # Whisper's window
LIVE_MARGIN_S = 5 # live transcription leaves the newest audio alone (the VAD may still change it)
LIVE_MIN_TAIL_S = 15 # ... and only looks for finished chunks once this much is untranscribed
LEAD_IN_S = 0.5 # silence put before every chunk (see _join)
OV_CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude" / "ov_cache"


def build_prompt(prompt: str | None, vocabulary: Sequence[str] = ()) -> str | None:
    """Append custom words so Whisper spells them right ("Klot" -> "Claude").

    Tested against faster-whisper's `hotwords`: same fixes, but hotwords cost commas.
    """
    words = ", ".join(w.strip() for w in vocabulary if w.strip())
    if not words:
        return prompt or None
    return f"{prompt} {words}." if prompt else f"{words}."


Detect = Callable[[np.ndarray], str] # audio -> language code (one cheap encoder pass)


def _speech_groups(audio: np.ndarray, detect: Detect | None = None, offset: int = 0,
                   cache: dict | None = None, until: int | None = None) -> list[list[dict]]:
    """VAD speech segments ({"start", "end"} in samples), grouped into chunks of <= 30 s.

    A segment is speech between pauses of >= 2 s (longer stretches are split at 30 s). Whisper
    picks one language per chunk, from its start: with segments merged blindly (as faster-
    whisper does), a German passage after an English one in the same chunk was *translated*
    into English. With `detect`, consecutive segments are only merged if they're in the same
    language. One chunk per segment instead avoids that too, but short half-sentence chunks
    made Whisper invent words (even "Command new paragraph" from the prompt). Test corpus with
    mid-sentence pauses (scripts/eval_quality.py): merged blindly 10 % word errors and whole
    passages translated, one per segment 6.7 %, by language 4.0 % (~0.26 s GPU per detected
    segment; single-segment dictations skip it). `cache` keeps detected languages by absolute
    start position (`offset` + sample) across live feeds. `until`: only segments that end
    before this sample (live: a segment still being spoken is neither detected nor grouped;
    its first second alone was detected wrongly).
    """
    segments = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=CHUNK_S))
    if until is not None:
        segments = [seg for seg in segments if seg["end"] <= until]
    cache = {} if cache is None else cache

    def language(seg: dict) -> str:
        key = offset + seg["start"] # not the end: it moves while the passage is still spoken
        if key not in cache:
            cache[key] = detect(audio[seg["start"]:seg["end"]])
        return cache[key]

    groups: list[list[dict]] = []
    size = 0
    for seg in segments:
        n = seg["end"] - seg["start"]
        if (groups and size + n <= CHUNK_S * SAMPLE_RATE
                and (detect is None or language(groups[-1][0]) == language(seg))):
            groups[-1].append(seg)
            size += n
        else:
            groups.append([seg])
            size = n
    return groups


def _join(audio: np.ndarray, group: list[dict]) -> np.ndarray:
    """The group's speech, after LEAD_IN_S of silence: a chunk that starts right on a word
    made Whisper skip its first words (130 mid-sentence cuts of the test corpus: 13 % lost 3+
    words, 5 % with 0.3-0.5 s of silence first). Free: Whisper pads every chunk to 30 s."""
    lead = np.zeros(int(LEAD_IN_S * SAMPLE_RATE), np.float32)
    return np.concatenate([lead] + [audio[s["start"]:s["end"]] for s in group])


def speech_chunks(audio: np.ndarray, detect: Detect | None = None) -> list[np.ndarray]:
    """Cut audio at speech pauses into <= 30 s chunks (silence dropped), see _speech_groups.

    Transcribing chunks without timestamps avoids both the capitals-after-pauses of timestamp
    mode and words cut at 30 s borders.
    """
    return [_join(audio, g) for g in _speech_groups(audio, detect)]


class LiveTranscription:
    """Transcribes a recording while it is still running, so stopping only waits for the end.

    feed() gets the audio recorded so far (call it every few seconds from a worker thread). A
    chunk is final once the VAD has started a later chunk after it: more audio can't change it
    any more, so it is transcribed right away. finish() transcribes the rest. Long dictations
    (minutes) then take ~1 s after the stop tap instead of ~1 s per 30 s of speech. Not
    thread-safe: feed() and finish() must not overlap.
    """

    def __init__(self, transcribe_chunks: Callable[[list[np.ndarray]], str],
                 detect: Detect | None = None) -> None:
        self._transcribe_chunks = transcribe_chunks
        self._detect = detect
        self._languages: dict = {} # detected once per segment, not on every feed
        self._done = 0 # samples already transcribed
        self._texts: list[str] = []

    def _groups(self, tail: np.ndarray, until: int | None = None) -> list[list[dict]]:
        return _speech_groups(tail, self._detect, self._done, self._languages, until)

    @property
    def done_s(self) -> float:
        return self._done / SAMPLE_RATE

    def feed(self, audio: np.ndarray) -> None:
        tail = audio[self._done:]
        if len(tail) < LIVE_MIN_TAIL_S * SAMPLE_RATE:
            return # not worth a VAD pass yet
        # Only finished segments; the last group may still be joined by the next one.
        final = self._groups(tail, until=len(tail) - LIVE_MARGIN_S * SAMPLE_RATE)[:-1]
        if final:
            self._texts.append(self._transcribe_chunks([_join(tail, g) for g in final]))
            self._done += final[-1][-1]["end"]

    @staticmethod
    def has_speech(audio: np.ndarray) -> bool:
        """VAD check on a short piece of audio (for the forgotten-recording auto-stop)."""
        return bool(get_speech_timestamps(np.asarray(audio, dtype=np.float32), VadOptions()))

    def finish(self, audio: np.ndarray) -> str:
        """Transcribe what feed() hasn't and return the whole text."""
        tail = np.asarray(audio[self._done:], dtype=np.float32)
        rest = [_join(tail, g) for g in self._groups(tail)]
        if rest:
            self._texts.append(self._transcribe_chunks(rest))
        return join_chunks(self._texts)


class Transcriber:
    """faster-whisper on the CPU.

    Defaults come from the benchmarks in docs/transcription.md: small, greedy decoding, 4 threads
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
        self.language = language # None = auto-detect
        self.initial_prompt = build_prompt(initial_prompt, vocabulary)
        self.batch_size = batch_size
        kwargs = dict(device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
        try: # cached model: no network request at startup (private, works offline)
            self.model = WhisperModel(model, local_files_only=True, **kwargs)
        except Exception: # first run: download
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


def intel_gpu() -> str | None:
    """OpenVINO's name for an Intel GPU ("GPU", or "GPU.1" next to another GPU), or None.

    OpenVINO also lists NVIDIA cards as "GPU" (through OpenCL; seen with an RTX 4070), but its
    GPU plugin is made for Intel GPUs, so only devices named Intel count.
    """
    try:
        import openvino as ov
        core = ov.Core()
        for device in core.available_devices:
            if (device.startswith("GPU")
                    and "intel" in str(core.get_property(device, "FULL_DEVICE_NAME")).lower()):
                return device
    except Exception: # not installed, driver problem, ...
        log.info("OpenVINO GPU check failed", exc_info=True)
    return None


def gpu_available() -> bool:
    """True if OpenVINO is installed and sees an Intel GPU (e.g. the iGPU)."""
    return intel_gpu() is not None


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
        try: # cached: no network request at startup
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

    def detect_language(self, audio: np.ndarray) -> str:
        """Spoken language of a short piece of audio: one encoder pass, no decoding."""
        cfg = self.pipeline.get_generation_config()
        cfg.task, cfg.return_timestamps, cfg.max_new_tokens = "transcribe", False, 1
        return str(self.pipeline.generate(audio, cfg).language or "")

    def transcribe_detailed(self, audio: np.ndarray) -> tuple[str, str]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return self._generate(speech_chunks(audio, self._detect))

    @property
    def _detect(self) -> Detect | None:
        return None if self.language else self.detect_language # fixed language: no need

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_detailed(audio)[0]

    def transcribe_chunks(self, chunks: list[np.ndarray]) -> str:
        return self._generate(chunks)[0]

    def live(self) -> LiveTranscription:
        return LiveTranscription(self.transcribe_chunks, self._detect)


def load_cuda_libraries() -> bool:
    """Load cuBLAS and cuDNN from the pip packages of the [cuda] extra (nvidia-cublas-cu12,
    nvidia-cudnn-cu12). CTranslate2 loads them by name only when the GPU is first used, and
    Windows doesn't search the packages' folders: a DLL that is already loaded is found.
    Returns False if they aren't installed."""
    import ctypes
    import importlib.util

    loaded = False
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            spec = None
        for folder in (spec.submodule_search_locations or []) if spec else []:
            for dll in sorted(Path(folder, "bin").glob("*.dll")):
                ctypes.WinDLL(str(dll))
                loaded = True
    return loaded


CUDA_BEAM = 5 # beam_size 0 (auto) on CUDA: fewer errors in all four test sets, +5 % time
LOW_VRAM_MB = 4096 # less free VRAM: int8 weights (turbo: half the memory, same accuracy)
LARGE_VRAM_MB = 7000 # model "auto" with this much free: large-v3 (3.9 GB) + room for NLLB
GPU_MODEL = "large-v3-turbo" # model "auto" otherwise, and always on the iGPU


def free_vram_mb() -> int | None:
    """Free memory of the first NVIDIA GPU in MB, from NVML (installed with the driver);
    None if unknown."""
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    try:
        nvml = ctypes.WinDLL("nvml.dll")
        if nvml.nvmlInit_v2() != 0:
            return None
        try:
            handle, memory = ctypes.c_void_p(), Memory()
            if (nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0
                    or nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0):
                return None
            return memory.free // 2**20
        finally:
            nvml.nvmlShutdown()
    except (OSError, AttributeError):
        return None


def cuda_compute_type(free_mb: int | None) -> str:
    """float16, or int8 weights if the GPU has little free memory. On FLEURS (German and
    English, clean and noisy) float32, float16 and int8 were equally accurate; int8 needs
    1.0 GB instead of 2.2 GB (see docs/transcription.md)."""
    return "int8_float16" if free_mb is not None and free_mb < LOW_VRAM_MB else "float16"


def cuda_choice(model: str, free_mb: int | None) -> tuple[str, str]:
    """(model, compute type) on CUDA, by free VRAM. model "auto":
    - LARGE_VRAM_MB free: large-v3 float16 (3.9 GB; 11-19 % fewer errors than large-v3-turbo
      on FLEURS, clean and noisy; 0.66 s per 10 s on an RTX 4070),
    - LOW_VRAM_MB free: large-v3 int8 (2.1 GB, as much as the turbo in float16, and still
      fewer errors than it),
    - less: large-v3-turbo int8 (1.0 GB). Unknown VRAM: large-v3-turbo float16.
    A model set in config.toml stays; only its precision follows the VRAM."""
    if model != "auto":
        return model, cuda_compute_type(free_mb)
    if free_mb is None:
        return GPU_MODEL, "float16"
    if free_mb >= LARGE_VRAM_MB:
        return "large-v3", "float16"
    return ("large-v3" if free_mb >= LOW_VRAM_MB else GPU_MODEL), "int8_float16"


def cuda_available() -> bool:
    """True if there is an NVIDIA GPU and the CUDA libraries of the [cuda] extra are installed."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0 and load_cuda_libraries()
    except Exception:
        log.info("CUDA check failed", exc_info=True)
        return False


class CudaTranscriber(OpenVinoTranscriber):
    """faster-whisper on an NVIDIA GPU (CUDA; model and precision by free VRAM, see
    cuda_choice). Same chunking and prompt as the iGPU path: chunks split by language, one
    model call per chunk, so each chunk gets its own language (the batched CPU pipeline
    detects one language for all chunks)."""

    device = "CUDA"

    def __init__(
        self,
        model: str = "auto",
        language: str | None = None,
        initial_prompt: str | None = DEFAULT_PROMPT,
        vocabulary: Sequence[str] = (),
        on_status: Callable[[str], None] = lambda s: None,
        compute_type: str = "auto",
        beam_size: int = 0,
    ) -> None:
        """model / compute_type "auto": by free VRAM (cuda_choice); beam_size 0: CUDA_BEAM."""
        load_cuda_libraries()
        self.language = language
        self.initial_prompt = build_prompt(initial_prompt, vocabulary)
        self.beam_size = beam_size or CUDA_BEAM
        free = free_vram_mb()
        model, auto_compute = cuda_choice(model, free)
        if compute_type == "auto":
            compute_type = auto_compute
        log.info("CUDA: %s, %s, beam %d (%s MB VRAM free)", model, compute_type, self.beam_size,
                 free if free is not None else "?")
        self.model_name, self.compute_type = model, compute_type
        kwargs = dict(device="cuda", compute_type=compute_type)
        try: # cached model: no network request at startup
            self.model = WhisperModel(model, local_files_only=True, **kwargs)
        except Exception:
            on_status("Downloading model (one-time)…")
            self.model = WhisperModel(model, **kwargs)
        silence = np.zeros(SAMPLE_RATE, np.float32) # CUDA setup + VAD load, see OpenVINO
        speech_chunks(silence)
        self._generate([silence])

    def _generate(self, chunks: list[np.ndarray]) -> tuple[str, str]:
        texts, language = [], ""
        for chunk in chunks:
            segments, info = self.model.transcribe(
                chunk, language=self.language, beam_size=self.beam_size,
                initial_prompt=self.initial_prompt, without_timestamps=True,
                condition_on_previous_text=False)
            texts.append(join_chunks(s.text for s in segments))
            language = language or info.language
        return join_chunks(texts), language

    def detect_language(self, audio: np.ndarray) -> str:
        return self.model.detect_language(np.asarray(audio, dtype=np.float32))[0]


def create_transcriber(
    device: str = "auto",
    gpu_model: str = "auto",
    cpu_model: str = "small",
    on_status: Callable[[str], None] = lambda s: None,
    language: str | None = None,
    initial_prompt: str | None = DEFAULT_PROMPT,
    vocabulary: Sequence[str] = (),
    beam_size: int = 0,
    **cpu_kwargs: object,
) -> Transcriber | OpenVinoTranscriber:
    """device "auto" / "gpu": NVIDIA GPU (CUDA) if there is one, else the Intel iGPU
    (OpenVINO), else the CPU. "cuda" / "igpu": only that GPU, else the CPU. "cpu": always
    faster-whisper on the CPU. A GPU that fails to load falls back to the next option.
    beam_size 0 = auto: 5 on CUDA, 1 on the CPU (the iGPU pipeline only decodes greedily).
    gpu_model "auto": by free VRAM on CUDA (cuda_choice), large-v3-turbo on the iGPU, where
    large-v3 was 4x slower at the same accuracy."""
    common = dict(language=language, initial_prompt=initial_prompt, vocabulary=vocabulary)
    if device in ("auto", "gpu", "cuda"):
        if cuda_available():
            try:
                return CudaTranscriber(gpu_model, on_status=on_status, beam_size=beam_size,
                                       **common)
            except Exception:
                log.exception("CUDA backend failed, trying the next one")
        else:
            log.info("no usable NVIDIA GPU found")
    if device in ("auto", "gpu", "igpu"):
        ov_device = intel_gpu()
        if ov_device:
            try:
                model = GPU_MODEL if gpu_model == "auto" else gpu_model
                return OpenVinoTranscriber(model, ov_device, on_status=on_status, **common)
            except Exception:
                log.exception("GPU backend failed, falling back to CPU")
        else:
            log.info("no OpenVINO GPU found, using CPU")
    on_status("Loading model…")
    return Transcriber(cpu_model, **common, beam_size=beam_size or 1, **cpu_kwargs)
