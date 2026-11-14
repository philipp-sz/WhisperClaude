"""Speech-to-text backends: load the model once, transcribe 16 kHz mono float32 audio.

- OpenVinoTranscriber: Whisper on the Intel iGPU via OpenVINO GenAI (~10x faster, ~half the
  energy per dictation, and fast enough for large-v3-turbo; see docs/transcription.md).
- Transcriber: faster-whisper on the CPU (fallback when there's no usable GPU).
create_transcriber() picks one based on the hardware.
"""
from __future__ import annotations

import logging
import os
import re
import time
import zlib
from functools import lru_cache
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub import snapshot_download
from faster_whisper.vad import VadOptions, get_speech_timestamps

from whisperclaude.constants import ( # noqa: F401 (re-exported)
    DEFAULT_PROMPT, DEFAULT_PROMPT_LANGUAGES, DEFAULT_VOCABULARY, SAMPLE_RATE)
from whisperclaude.system import cuda_available, free_vram_mb, load_cuda_libraries
from whisperclaude.textfix import join_chunks

log = logging.getLogger(__name__)

CHUNK_S = 30 # Whisper's window
LIVE_MARGIN_S = 5 # live transcription leaves the newest audio alone (the VAD may still change it)
LIVE_MIN_TAIL_S = 15 # ... and only looks for finished chunks once this much is untranscribed
LIVE_DETECT_MARGIN_S = 2.5 # a segment is final (VAD: >= 2 s of silence + padding after it): detect its language
LEAD_IN_S = 0.5 # silence put before every chunk (see _join)
OV_CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude" / "ov_cache"


LOOP_RATIO = 2.4 # Whisper's own compression-ratio limit; normal chunks: at most 1.9 (measured)
LOOP_WORDS_PER_S = 6.0 # speech has at most ~4 words/s (measured 3.5); only above LOOP_MIN_WORDS
LOOP_MIN_WORDS = 20
RETRY_PENALTY = 1.1 # logits of tokens already produced are divided by this in the retry
_REPEATS = re.compile(r"\b(\S+(?:\s+\S+){0,3}?)(?:\s+\1\b){3,}")


GUARD_MIN_WORDS = 6 # shorter texts carry too little signal for the language check
GUARD_MIN_PROB = 0.9 # the text's language must be this certain to override Whisper's language token


@lru_cache(maxsize=4)
def _identifier(languages: tuple[str, ...]):
    """py3langid restricted to `languages` (~0.5 s to load, once). Deferred import: only the
    language guard needs it, and the translator loads it anyway."""
    from py3langid.langid import MODEL_FILE, LanguageIdentifier
    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    identifier.set_languages(list(languages))
    return identifier


def text_language(text: str, languages: Sequence[str]) -> str | None:
    """Which of `languages` the text is written in, or None if too short or not clear."""
    if len(languages) < 2 or len(text.split()) < GUARD_MIN_WORDS:
        return None
    lang, prob = _identifier(tuple(languages)).classify(text)
    return lang if prob >= GUARD_MIN_PROB else None


def is_loop(text: str, seconds: float) -> bool:
    """True if a chunk's text is a decoding loop ("dx dx dx ...") rather than speech.

    Repetition compresses well: zlib shrinks a loop far below the 1/2.4 of its length that
    normal text never reaches. A loop also produces more words than anyone can speak.
    """
    data = text.encode("utf-8")
    if not data:
        return False
    ratio = len(data) / len(zlib.compress(data))
    words = len(text.split())
    return ratio > LOOP_RATIO or (words > LOOP_MIN_WORDS and words / max(seconds, 1e-3) > LOOP_WORDS_PER_S)


def collapse_repeats(text: str) -> str:
    """Runs of the same word or phrase (up to 4 words, 4+ times in a row) -> one occurrence."""
    return _REPEATS.sub(r"\1", text)


def build_prompt(prompt: str | None, vocabulary: Sequence[str] = ()) -> str | None:
    """Append custom words so Whisper spells them right ("Klot" -> "Claude").

    Tested against faster-whisper's `hotwords`: same fixes, but hotwords cost commas.
    """
    words = ", ".join(w.strip() for w in vocabulary if w.strip())
    if not words:
        return prompt or None
    return f"{prompt} {words}." if prompt else f"{words}."


# speech below this drops the prompt sentence only for a language already heard (PromptByLanguage)
MIN_SWITCH_S = 4.0


class PromptByLanguage:
    """The prompt sentence only for the languages it is written in (`prompt_languages`).

    With the German+English sentence, Whisper translated other languages into English (FLEURS:
    French 77 %, Spanish 55 % word errors), even with the chunk's language forced. Chunks in
    another language get only the vocabulary as prompt. Whisper's own language for the chunk
    decides: no extra detection pass.
    """

    initial_prompt: str | None = None
    other_prompt: str | None = None # for chunks in a language the prompt is not written in
    prompt_languages: tuple[str, ...] = DEFAULT_PROMPT_LANGUAGES
    language: str | None = None # fixed language; None = auto-detect
    # language of the last chunk, kept across dictations: OpenVINO picks the first prompt by it.
    # Reset per dictation, every French dictation cost two decodes (FLEURS fr: 6.3 s instead of
    # 2.9 s GPU per audio minute); kept, only a language switch costs the second one.
    _heard: str | None = None
    MIN_SWITCH_S = MIN_SWITCH_S

    def _set_prompts(self, prompt: str | None, vocabulary: Sequence[str],
                     prompt_languages: Sequence[str]) -> None:
        self.initial_prompt = build_prompt(prompt, vocabulary)
        self.other_prompt = build_prompt(None, vocabulary)
        self.prompt_languages = tuple(prompt_languages)

    def _prompt(self, language: str | None) -> str | None:
        """The prompt for speech in `language` (None: not known yet)."""
        return (self.initial_prompt if not language or language in self.prompt_languages
                else self.other_prompt)

    def _trusted(self, language: str | None, seconds: float) -> str | None:
        """The language Whisper heard in a chunk of `seconds`, or None (unknown: the prompt
        sentence stays) if the chunk is shorter than MIN_SWITCH_S and the language is neither a
        prompt language nor the previous chunk's. German and English cut to 1-3 s were heard as
        Russian or Korean in 3.6 % of 507 pieces: with the sentence Whisper still wrote them in
        Latin script, decoded as Russian or Korean it wrote Cyrillic or Hangul."""
        if (not language or language in self.prompt_languages or language == self._heard
                or seconds >= self.MIN_SWITCH_S):
            return language or None
        return None

    def _wrong_language(self, heard: str | None, text: str) -> bool:
        """True if Whisper heard a prompt language but wrote the text in the other one."""
        return (bool(heard) and not self.language and heard in self.prompt_languages
                and text_language(text, self.prompt_languages) not in (None, heard))

    def _forced(self, language: str | None) -> str | None:
        """The language to force when decoding a chunk heard (and trusted) as `language`: only
        outside the prompt languages (inside them Whisper detects, as with mixed audio)."""
        return (language if not self.language and self._prompt(language) != self.initial_prompt
                else None)


VAD_LEVEL_DBFS = -26.0 # quiet audio is raised to this RMS level, for the VAD only
VAD_MAX_GAIN_DB = 20.0


def vad_input(audio: np.ndarray) -> np.ndarray:
    """The audio as the VAD should see it: quiet recordings amplified (never attenuated).

    Silero's speech probability depends on the level: a podcast picked up at -41 dBFS lost
    three passages (probabilities 0.1-0.4 under the 0.5 threshold). Raising quiet audio to
    VAD_LEVEL_DBFS (at most VAD_MAX_GAIN_DB) cut the errors on the author's recordings by 11%.
    Only the VAD sees the gain; the model gets the original audio.
    """
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    if rms <= 0.0:
        return audio
    gain_db = min(max(VAD_LEVEL_DBFS - 20 * np.log10(rms), 0.0), VAD_MAX_GAIN_DB)
    if gain_db == 0.0:
        return audio
    return np.clip(audio * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)


Detect = Callable[[np.ndarray], str] # audio -> language code (one cheap encoder pass)


def _vad_segments(audio: np.ndarray) -> list[dict]:
    """Speech segments ({"start", "end"} in samples) between pauses of >= 2 s, split at 30 s."""
    return get_speech_timestamps(vad_input(audio), VadOptions(max_speech_duration_s=CHUNK_S))


def _speech_groups(audio: np.ndarray, detect: Detect | None = None, offset: int = 0,
                   cache: dict | None = None, until: int | None = None,
                   segments: list[dict] | None = None) -> list[list[dict]]:
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
    its first second alone was detected wrongly). `segments`: the VAD result of `audio`, if the
    caller has it already.
    """
    if segments is None:
        segments = _vad_segments(audio)
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
    speech = [audio[s["start"]:s["end"]] for s in group]
    # Never past 30 s: OpenVINO GenAI splits longer audio into windows (a 0.5 s second window
    # invents text such as "Thank you."), so a full chunk gets a shorter lead-in.
    room = CHUNK_S * SAMPLE_RATE - sum(len(x) for x in speech)
    lead = np.zeros(max(0, min(int(LEAD_IN_S * SAMPLE_RATE), room)), np.float32)
    return np.concatenate([lead] + speech)


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
        self.tail_s = 0.0 # finish(): time for the chunks it still had to transcribe
        self.tail_chunks = 0 # finish(): how many chunks those were

    def _groups(self, tail: np.ndarray, until: int | None = None,
                segments: list[dict] | None = None) -> list[list[dict]]:
        return _speech_groups(tail, self._detect, self._done, self._languages, until, segments)

    def _detect_ended(self, tail: np.ndarray, segments: list[dict]) -> None:
        """Detect (and cache, under the key _speech_groups uses) the language of every segment
        that has ended, now instead of after the stop tap: ~0.27 s of iGPU per segment that
        would otherwise sit between the stop and the text. A segment ended once the VAD has
        seen its silence, so its audio (and language) can't change any more. A lone first
        segment is skipped: its language is only needed once a second segment exists, and a
        short dictation (one segment) should cost no extra pass at all."""
        limit = len(tail) - LIVE_DETECT_MARGIN_S * SAMPLE_RATE
        for seg in segments:
            key = self._done + seg["start"]
            if seg["end"] > limit or key in self._languages or len(segments) < 2:
                continue
            self._languages[key] = self._detect(tail[seg["start"]:seg["end"]])

    @property
    def done_s(self) -> float:
        return self._done / SAMPLE_RATE

    def feed(self, audio: np.ndarray) -> None:
        tail = audio[self._done:]
        if len(tail) < LIVE_DETECT_MARGIN_S * SAMPLE_RATE:
            return
        segments = _vad_segments(tail) # ~4 ms per second of audio
        if self._detect is not None:
            self._detect_ended(tail, segments)
        if len(tail) < LIVE_MIN_TAIL_S * SAMPLE_RATE:
            return # not worth transcribing yet
        # Only finished segments; the last group may still be joined by the next one.
        final = self._groups(tail, until=len(tail) - LIVE_MARGIN_S * SAMPLE_RATE,
                             segments=segments)[:-1]
        if final:
            self._texts.append(self._transcribe_chunks([_join(tail, g) for g in final]))
            self._done += final[-1][-1]["end"]

    @staticmethod
    def has_speech(audio: np.ndarray) -> bool:
        """VAD check on a short piece of audio (for the forgotten-recording auto-stop)."""
        return bool(get_speech_timestamps(np.asarray(audio, dtype=np.float32), VadOptions()))

    def finish(self, audio: np.ndarray) -> str:
        """Transcribe what feed() hasn't and return the whole text. Sets tail_s and
        tail_chunks (grouping included)."""
        t0 = time.perf_counter()
        tail = np.asarray(audio[self._done:], dtype=np.float32)
        rest = [_join(tail, g) for g in self._groups(tail)]
        if rest:
            self._texts.append(self._transcribe_chunks(rest))
        self.tail_s, self.tail_chunks = time.perf_counter() - t0, len(rest)
        return join_chunks(self._texts)


class Transcriber(PromptByLanguage):
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
        prompt_languages: Sequence[str] = DEFAULT_PROMPT_LANGUAGES,
    ) -> None:
        self.beam_size = beam_size
        self.language = language # None = auto-detect
        self._set_prompts(initial_prompt, vocabulary, prompt_languages)
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
        segments, info = self._pipeline_transcribe(audio, vad_filter=True)
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
        segments, _ = self._pipeline_transcribe(
            np.concatenate(chunks).astype(np.float32),
            clip_timestamps=[{"start": float(a), "end": float(b)}
                             for a, b in zip(bounds[:-1], bounds[1:])],
        )
        return join_chunks(s.text for s in segments)

    def _pipeline_transcribe(self, audio: np.ndarray, **kwargs: object):
        """The batched pipeline with the prompt for the detected language. It detects before
        decoding (segments are lazy), so another language costs a second detection, not a
        second decode."""
        def run(language: str | None):
            return self.pipeline.transcribe(
                audio, language=language, beam_size=self.beam_size, batch_size=self.batch_size,
                initial_prompt=self._prompt(language), without_timestamps=True, **kwargs)
        segments, info = run(self.language)
        heard = self._trusted(info.language, len(audio) / SAMPLE_RATE)
        if self._forced(heard):
            segments, info = run(heard)
        self._heard = heard
        return segments, info

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
        import openvino as ov # deferred: optional on NVIDIA/CPU-only machines, ~185 ms
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


class OpenVinoTranscriber(PromptByLanguage):
    """Whisper on an OpenVINO device (the iGPU). Same chunking/prompt as the CPU path."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "GPU",
        language: str | None = None,
        initial_prompt: str | None = DEFAULT_PROMPT,
        vocabulary: Sequence[str] = (),
        on_status: Callable[[str], None] = lambda s: None,
        prompt_languages: Sequence[str] = DEFAULT_PROMPT_LANGUAGES,
    ) -> None:
        import openvino_genai # deferred: optional (CUDA/CPU machines), ~205 ms

        self.device = device
        self.language = language
        self._set_prompts(initial_prompt, vocabulary, prompt_languages)
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
        self._heard = None # not the warmup's guess for silence

    def _config(self, repetition_penalty: float = 1.0, language: str | None = None,
                guess: str | None = None, sentence: bool = True):
        """language: force this one (default: the configured one, else Whisper detects);
        guess: the language to pick the prompt for when Whisper detects; sentence: False
        drops the prompt sentence (vocabulary only)."""
        language = language or self.language
        cfg = self.pipeline.get_generation_config()
        cfg.task = "transcribe"
        cfg.return_timestamps = False
        cfg.initial_prompt = self._prompt(language or guess) if sentence else self.other_prompt
        cfg.repetition_penalty = repetition_penalty
        if language:
            cfg.language = f"<|{language}|>"
        return cfg

    def _generate(self, chunks: list[np.ndarray]) -> tuple[str, str]:
        """Each chunk is decoded with the prompt for the language of the chunk before (Whisper
        detects the language before it reads the prompt) and decoded again only if the
        prompt was the wrong one (PromptByLanguage): a language switch costs one more decode."""
        texts, language = [], ""
        for chunk in chunks:
            guess = self._heard
            result = self.pipeline.generate(chunk, self._config(guess=guess))
            detected = str(result.language or "").strip("<|>")
            heard = self._trusted(detected, len(chunk) / SAMPLE_RATE - LEAD_IN_S)
            if not self.language and self._prompt(heard) != self._prompt(guess):
                result = self.pipeline.generate(chunk, self._config(language=self._forced(heard)))
            self._heard = heard
            text = result.texts[0].strip()
            if self._wrong_language(heard, text):
                text = self._language_guard(chunk, heard, text)
            if is_loop(text, len(chunk) / SAMPLE_RATE):
                text = self._decode_again(chunk, heard)
            texts.append(text)
            language = language or detected
        return join_chunks(texts), language

    def _language_guard(self, chunk: np.ndarray, heard: str, text: str) -> str:
        """Whisper heard `heard` (language token, probability > 0.99 in every case seen) but wrote
        the text in another prompt language: the English half of the prompt sentence pulled a
        German chunk into English (an 80 Hz high-pass or -16 LUFS on the input, one passage per
        recording; 32-46 errors). The chunk is decoded again without the sentence, forced to the
        heard language, and kept if that text is in the heard language. Real switches pass:
        token and text agree."""
        again = self.pipeline.generate(chunk, self._config(language=heard, sentence=False)).texts[0].strip()
        ok = not self._wrong_language(heard, again)
        log.warning("chunk heard as %s but written in another language; decoded again without the "
                    "prompt sentence%s", heard, "" if ok else ", still wrong: first text kept")
        return again if ok else text

    def _decode_again(self, chunk: np.ndarray, language: str | None = None) -> str:
        """A chunk that came out as a repetition loop, decoded again with a repetition penalty.

        Greedy decoding can't leave a loop once a repeat is the likeliest next token. The penalty
        divides the logits of tokens already produced by RETRY_PENALTY, enough to end the loop
        in the case seen ("x² dx dx dx ..." 368 times -> "x² dx"); beam search found it too
        but is not implemented on the iGPU. Only looped chunks pay for it. If the retry loops
        too, the repeats are collapsed, so no speech around the loop is dropped.
        """
        text = self.pipeline.generate(
            chunk, self._config(RETRY_PENALTY, self._forced(language))).texts[0].strip()
        looped = is_loop(text, len(chunk) / SAMPLE_RATE)
        log.warning("repetition loop in a %.0f s chunk; decoded again with a repetition penalty%s",
                    len(chunk) / SAMPLE_RATE, ", still looping: repeats collapsed" if looped else "")
        return collapse_repeats(text) if looped else text

    def detect_language(self, audio: np.ndarray) -> str:
        """Spoken language of a short piece of audio: one encoder pass, no decoding."""
        cfg = self.pipeline.get_generation_config()
        cfg.task, cfg.return_timestamps, cfg.max_new_tokens = "transcribe", False, 1
        audio = audio[-CHUNK_S * SAMPLE_RATE:] # > 30 s: generate(max_new_tokens=1) never returns
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


CUDA_BEAM = 5 # beam_size 0 (auto) on CUDA: fewer errors in all four test sets, +5 % time
LOW_VRAM_MB = 4096 # less free VRAM: int8 weights (turbo: half the memory, same accuracy)
LARGE_VRAM_MB = 7000 # model "auto" with this much free: large-v3 (3.9 GB) + room for NLLB
GPU_MODEL = "large-v3-turbo" # model "auto" otherwise, and always on the iGPU


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
        prompt_languages: Sequence[str] = DEFAULT_PROMPT_LANGUAGES,
    ) -> None:
        """model / compute_type "auto": by free VRAM (cuda_choice); beam_size 0: CUDA_BEAM."""
        load_cuda_libraries()
        self.language = language
        self._set_prompts(initial_prompt, vocabulary, prompt_languages)
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
            segments, info = self._transcribe(chunk, self.language)
            heard = self._trusted(info.language, len(chunk) / SAMPLE_RATE)
            if self._forced(heard): # lazy segments: costs a detection, not a decode
                segments, info = self._transcribe(chunk, heard)
            self._heard = heard
            text = join_chunks(s.text for s in segments)
            if self._wrong_language(heard, text): # see OpenVinoTranscriber._language_guard
                again = join_chunks(s.text for s in self._transcribe(chunk, heard, sentence=False)[0])
                log.warning("chunk heard as %s but written in another language; decoded again "
                            "without the prompt sentence", heard)
                text = text if self._wrong_language(heard, again) else again
            texts.append(text)
            language = language or info.language
        return join_chunks(texts), language

    def _transcribe(self, chunk: np.ndarray, language: str | None, sentence: bool = True):
        return self.model.transcribe(
            chunk, language=language, beam_size=self.beam_size,
            initial_prompt=self._prompt(language) if sentence else self.other_prompt,
            without_timestamps=True,
            condition_on_previous_text=False)

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
    prompt_languages: Sequence[str] = DEFAULT_PROMPT_LANGUAGES,
    **cpu_kwargs: object,
) -> Transcriber | OpenVinoTranscriber:
    """device "auto" / "gpu": NVIDIA GPU (CUDA) if there is one, else the Intel iGPU
    (OpenVINO), else the CPU. "cuda" / "igpu": only that GPU, else the CPU. "cpu": always
    faster-whisper on the CPU. A GPU that fails to load falls back to the next option.
    beam_size 0 = auto: 5 on CUDA, 1 on the CPU (the iGPU pipeline only decodes greedily).
    gpu_model "auto": by free VRAM on CUDA (cuda_choice), large-v3-turbo on the iGPU, where
    large-v3 was 4x slower at the same accuracy."""
    common = dict(language=language, initial_prompt=initial_prompt, vocabulary=vocabulary,
                  prompt_languages=prompt_languages)
    if device in ("auto", "gpu", "cuda"):
        cuda = _cuda_transcriber(gpu_model, on_status, beam_size, common)
        if cuda is not None:
            return cuda
    if device in ("auto", "gpu", "igpu"):
        openvino = _openvino_transcriber(gpu_model, on_status, common)
        if openvino is not None:
            return openvino
    on_status("Loading model…")
    return Transcriber(cpu_model, **common, beam_size=beam_size or 1, **cpu_kwargs)


def _cuda_transcriber(gpu_model: str, on_status: Callable[[str], None], beam_size: int,
                      common: dict) -> CudaTranscriber | None:
    """A CUDA transcriber, or None if there is no usable NVIDIA GPU or it fails to load."""
    if not cuda_available():
        log.info("no usable NVIDIA GPU found")
        return None
    try:
        return CudaTranscriber(gpu_model, on_status=on_status, beam_size=beam_size, **common)
    except Exception:
        log.exception("CUDA backend failed, trying the next one")
        return None


def _openvino_transcriber(gpu_model: str, on_status: Callable[[str], None],
                          common: dict) -> OpenVinoTranscriber | None:
    """An OpenVINO transcriber on the Intel iGPU, or None if there is none or it fails to load."""
    ov_device = intel_gpu()
    if not ov_device:
        log.info("no OpenVINO GPU found, using CPU")
        return None
    model = GPU_MODEL if gpu_model == "auto" else gpu_model
    try:
        return OpenVinoTranscriber(model, ov_device, on_status=on_status, **common)
    except Exception:
        log.exception("GPU backend failed, falling back to CPU")
        return None
