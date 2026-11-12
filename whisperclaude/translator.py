"""Text translation with NLLB-200 (CTranslate2): NVIDIA GPU if there is one, else the CPU.

Heavy (ctranslate2, tokenizers): loaded on the model-loader thread, never on the UI path.
Text is translated sentence by sentence (NLLB was trained on single sentences and drops
content from long inputs), in one batch; line breaks, list markers and indentation are kept.
The source language is detected per sentence (py3langid), with the language of the whole text
for sentences too short to tell, unless the user picked it; sentences already in the target
language stay as they are.
"""
from __future__ import annotations

import functools
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import ctranslate2
from huggingface_hub import snapshot_download
from py3langid.langid import MODEL_FILE, LanguageIdentifier
from tokenizers import Tokenizer

from whisperclaude.languages import NLLB_CODES, SOURCE_CODES
from whisperclaude.system import MIN_RAM_GB, cuda_available, free_vram_mb, total_ram_gb, user_languages

log = logging.getLogger(__name__)

MODELS = { # config name -> pre-converted CTranslate2 model (int8) on Hugging Face
    "nllb-600m": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
    "nllb-1.3b": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
}
LOW_VRAM_MB = 2048 # "auto" on CUDA with less free VRAM (after Whisper): the smaller model
MIN_DETECT_CHARS = 25 # shorter sentences take the language of the whole text
MIN_CONFIDENCE = 0.9

# A sentence ends after . ! ? … (plus closing quotes/brackets) and a space before a capital
# letter, digit or opening quote (no split in "e.g. this"), or after 。！？.
_SENTENCE_END = re.compile(
    r"(?<=[.!?…])[\"'”’“»«›)\]]*\s+(?=[\"'“„«»(\[]?[^\W\d_a-z]|\d)|(?<=[。！？])\s*")
_LAST_WORD = re.compile(r"(\S+)\.[\"'”’“»«›)\]]*$")
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "ca", "bzw", "usw", "vs", "nr", "hr",
                 "fr", "jr", "sr", "inc", "ltd", "no", "vgl", "evtl", "ggf"}
_LINE = re.compile(r"^(\s*(?:[-*•]\s+|\d+[.)]\s+|>\s*)?)(.*?)(\s*)$")


def _is_abbreviation(piece: str) -> bool:
    """"z.", "B.", "e.g.", "Dr." at the end of a piece: no sentence end."""
    m = _LAST_WORD.search(piece)
    word = m.group(1).lstrip("(\"'„“«»") if m else ""
    return bool(word) and (len(word) == 1 or "." in word or word.casefold() in ABBREVIATIONS)


def split_sentences(text: str) -> list[str]:
    """Split one line of text into sentences (see _SENTENCE_END)."""
    parts, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        end = m.start() + len(m.group(0).rstrip())
        if _is_abbreviation(text[start:end]):
            continue
        piece = text[start:end]
        start = m.end()
        if piece.strip():
            parts.append(piece.strip())
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def segment(text: str) -> list[tuple[str, list[str], str]]:
    """Per line: (prefix, sentences, suffix). Prefix = indentation and a list marker
    ("- ", "1. ", "> "), kept untranslated, like empty lines."""
    rows = []
    for line in text.split("\n"):
        prefix, body, suffix = _LINE.match(line).groups()
        rows.append((prefix, split_sentences(body), suffix))
    return rows


_SPACE_BEFORE = re.compile(r"(?<=\S) +([,.](?!\d)|[!?;:](?!\S))")
_SPACE_BEFORE_FR = re.compile(r"(?<=\S) +([,.](?!\d))") # French keeps "Merci !"
_CONTRACTION = re.compile(r"(?<=\w) '(s|re|ll|ve|m|d)\b|(?<=n) '(t)\b|(?<=\w) (n't)\b")


def tidy(source: str, text: str, target: str) -> str:
    """Undo subtitle-style artifacts NLLB produces on short inputs ("- Yes , I 'm sure .")
    unless the source looks like that itself."""
    if text.startswith("- ") and not source.lstrip().startswith("-"):
        text = text[2:]
    if not re.search(r"\s[,.!?;:]", source):
        text = (_SPACE_BEFORE_FR if target == "fra_Latn" else _SPACE_BEFORE).sub(r"\1", text)
    if target == "eng_Latn":
        text = _CONTRACTION.sub(lambda m: m.group(3) or "'" + (m.group(1) or m.group(2)), text)
    return text


@functools.lru_cache(maxsize=1)
def _identifier():
    """The py3langid model restricted to SOURCE_CODES, loaded once (~0.5 s) and shared.
    set_languages() mutates the identifier, but only here and always with the same list, so
    sharing is safe. Per-instance state (`preferred`) stays on LanguageDetector."""
    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    identifier.set_languages([c for c in identifier.nb_classes if c in SOURCE_CODES])
    return identifier


class LanguageDetector:
    """py3langid restricted to languages NLLB has (else it may answer e.g. Latin).

    Short texts carry little signal: "ok" came out as Serbian, "Hallo" as Italian, each with
    under 5 % probability, and NLLB then wrote nonsense. For the whole text the user's own
    languages (`preferred`, first = most likely) therefore count 10x; ties go to the first.
    `preferred` comes from `[translation] my_languages`, else from Windows (user_languages).
    """

    PREFERRED_WEIGHT = 10.0

    def __init__(self, preferred: list[str] | None = None) -> None:
        self._id = _identifier()
        self.preferred = user_languages() if preferred is None else preferred

    def detect(self, text: str) -> tuple[str, float]:
        """(NLLB code, probability 0-1) of the most likely language."""
        lang, p = self._id.classify(text)
        return SOURCE_CODES[lang], float(p)

    def detect_text(self, text: str) -> str:
        """NLLB code of the most likely language, weighted toward the user's languages."""
        order = {lang: i for i, lang in enumerate(self.preferred)}

        def score(item: tuple[str, float]) -> tuple[float, int]:
            lang, p = item
            weight = self.PREFERRED_WEIGHT if lang in order else 1.0
            return (p * weight, -order.get(lang, len(order)))

        return SOURCE_CODES[max(self._id.rank(text), key=score)[0]]


def source_languages(sentences: list[str], detect: Callable[[str], tuple[str, float]],
                     detect_text: Callable[[str], str] | None = None) -> list[str]:
    """NLLB source code per sentence; unsure or short sentences get the whole text's
    (detect_text, if given, else detect)."""
    if not sentences:
        return []
    whole = " ".join(sentences)
    overall = detect_text(whole) if detect_text else detect(whole)[0]
    codes = []
    for s in sentences:
        code = overall
        if len(s) >= MIN_DETECT_CHARS:
            lang, p = detect(s)
            if p >= MIN_CONFIDENCE:
                code = lang
        codes.append(code)
    return codes


def auto_model(device: str, free_vram_mb: int | None, ram_gb: float | None) -> str:
    """The model for `model = "auto"`: nllb-1.3b (1.4 GB, better) unless memory is short. On
    CUDA that is less than LOW_VRAM_MB free after Whisper (unknown VRAM: still the large
    one); on the CPU less than MIN_RAM_GB of RAM (unknown RAM: the small one)."""
    if device == "cuda":
        small = free_vram_mb is not None and free_vram_mb < LOW_VRAM_MB
    else:
        small = ram_gb is None or ram_gb < MIN_RAM_GB
    return "nllb-600m" if small else "nllb-1.3b"


class Model(Protocol):
    """What LazyModel needs of a ctranslate2.Translator (tests use fakes)."""
    def load_model(self) -> None: ...
    def unload_model(self) -> None: ...
    def translate_batch(self, source: list[list[str]], **kwargs: Any) -> list: ...


class LazyModel:
    """The CTranslate2 model, loaded on demand and unloaded after `unload_after_s` of idle time
    (0 = once loaded, kept). The object stays: unload_model() frees the memory in 0.2-0.3 s,
    load_model() brings it back in 3-4 s (NLLB-1.3B on the CPU: 1.4 GB working set, 5.2 GB
    committed with MKL). One lock covers loading, unloading and translating, so an unload never
    happens during a load or a translation, and a load never runs twice. `factory` builds the
    model object; the clock is injectable for tests."""

    def __init__(self, factory: Callable[[], Model], unload_after_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, watch: bool = True) -> None:
        self._factory, self.unload_after_s, self._clock = factory, unload_after_s, clock
        self._watch = watch and unload_after_s > 0
        self._lock = threading.Lock() # held while loading, unloading and translating
        self._flag = threading.Lock() # only for _loading
        self._stop = threading.Event()
        self._model: Model | None = None
        self._loaded = False
        self._loading = False # a background load was started and has not finished
        self._watcher: threading.Thread | None = None # guarded by _lock
        self._last_used = clock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def start_load(self) -> bool:
        """Any thread, never blocks: load in the background unless loaded or loading already
        (True = a load was started). Also counts as use: no unload while the user picks."""
        self._last_used = self._clock()
        with self._flag:
            if self._loaded or self._loading:
                return False
            self._loading = True
        threading.Thread(target=self._background_load, daemon=True,
                         name="translation-load").start()
        return True

    def load(self) -> None:
        """Load now (blocks; no-op if loaded)."""
        with self._lock:
            self._load_locked()

    def _background_load(self) -> None:
        try:
            self.load()
        except Exception:
            log.exception("could not load the translation model") # translate() raises it
        finally:
            with self._flag:
                self._loading = False

    @contextmanager
    def use(self) -> Iterator[Model]:
        """The model, loaded (waits for a running load, or loads), locked against unloading."""
        with self._lock:
            self._load_locked()
            try:
                yield self._model
            finally:
                self._last_used = self._clock()

    def _load_locked(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        if self._model is None:
            self._model = self._factory()
        else:
            self._model.load_model()
        self._loaded = True
        self._last_used = self._clock()
        log.info("translation model loaded in %.1f s", time.perf_counter() - t0)
        if self._watch and self._watcher is None:
            self._watcher = threading.Thread(target=self._watch_idle, daemon=True,
                                             name="translation-idle")
            self._watcher.start()

    def _idle_s(self) -> float:
        return self._clock() - self._last_used

    def _unload_locked(self) -> None:
        idle, t0 = self._idle_s(), time.perf_counter()
        assert self._model is not None # loaded implies built
        self._model.unload_model()
        self._loaded = False
        log.info("translation model unloaded after %.0f min idle in %.1f s", idle / 60,
                 time.perf_counter() - t0)

    def unload_if_idle(self) -> bool:
        """Unload if loaded and idle for unload_after_s; waits for a translation or load that
        is running (which then counts as use). True = unloaded."""
        with self._lock:
            if not self._loaded or self._idle_s() < self.unload_after_s or self.unload_after_s <= 0:
                return False
            self._unload_locked()
            return True

    def _watch_idle(self) -> None:
        """Thread that exists while the model is loaded: sleeps until the idle time could be
        over, unloads, and ends."""
        delay = self.unload_after_s
        while not self._stop.wait(delay):
            with self._lock:
                remaining = self.unload_after_s - self._idle_s()
                if not self._loaded or remaining <= 0:
                    self._watcher = None # first: a failing unload must not leave a dead watcher
                    if self._loaded: # (the next load then starts a new one)
                        try:
                            self._unload_locked()
                        except Exception:
                            log.exception("could not unload the translation model")
                    return
            delay = max(remaining, 0.5)

    def close(self) -> None:
        """Stop the idle watcher (tests)."""
        self._stop.set()


def _run_parallel(*tasks: Callable[[], object]) -> list:
    """Run the tasks on threads, results in order; the first error is raised."""
    with ThreadPoolExecutor(len(tasks)) as pool:
        futures = [pool.submit(task) for task in tasks]
        return [f.result() for f in futures]


class Translator:
    """NLLB-200 on CTranslate2. The CTranslate2 model is a LazyModel: with `unload_after_s` > 0 it
    is unloaded after that idle time and loaded again on demand (prepare() starts it early);
    on the CPU it is not even loaded at startup. translate() is serialized by its lock."""

    def __init__(self, model: str = "auto", device: str = "auto", cpu_threads: int = 4,
                 beam_size: int = 2, on_status: Callable[[str], None] = lambda s: None,
                 languages: tuple[str, ...] = (), unload_after_s: float = 0.0) -> None:
        """model "auto": by memory, see auto_model. languages: the user's own languages (ISO
        codes, most used first); () = Windows'. unload_after_s: see LazyModel, 0 = keep loaded."""
        self.device = self._pick_device(device)
        if model == "auto":
            model = auto_model(self.device, free_vram_mb() if self.device == "cuda" else None,
                               total_ram_gb())
        self.model_name = model
        path = self._model_path(MODELS.get(model, model), on_status)
        self.beam_size = beam_size
        self.on_demand = self.device == "cpu" and unload_after_s > 0 # not loaded at startup
        self._lazy = LazyModel(lambda: self._build_model(path, cpu_threads), unload_after_s)
        tasks = [lambda: Tokenizer.from_file(str(Path(path) / "tokenizer.json")),
                 lambda: LanguageDetector(list(languages) or None)]
        if not self.on_demand: # the three pieces load side by side (3.2 s instead of 5)
            tasks.append(self._lazy.load)
        self.tokenizer, self.detector, *_ = _run_parallel(*tasks)
        if self.device == "cuda":
            self.translate("Hallo.", "eng_Latn") # first call sets up the device: pay it here

    def _pick_device(self, device: str) -> str:
        if device in ("auto", "cuda"):
            if cuda_available():
                return "cuda"
            if device == "cuda":
                log.warning("no usable NVIDIA GPU for the translation, using the CPU")
        return "cpu"

    @staticmethod
    def _model_path(repo: str, on_status: Callable[[str], None]) -> str:
        try: # cached: no network request at startup
            return snapshot_download(repo, local_files_only=True)
        except Exception:
            on_status("Downloading translation model (one-time)…")
            return snapshot_download(repo)

    def _build_model(self, path: str, cpu_threads: int) -> Model:
        compute = "int8_float16" if self.device == "cuda" else "int8"
        return ctranslate2.Translator(path, device=self.device, compute_type=compute,
                                      intra_threads=cpu_threads)

    @property
    def loaded(self) -> bool:
        """True if the CTranslate2 model is in memory (False while loading or after an unload)."""
        return self._lazy.loaded

    def prepare(self) -> None:
        """Any thread, never blocks: start loading the model if it is not loaded (the translate
        key was tapped, the load overlaps the language picker)."""
        self._lazy.start_load()

    def _tokens(self, sentence: str, source: str) -> list[str]:
        ids = self.tokenizer.encode(sentence, add_special_tokens=False).ids
        return [source] + [self.tokenizer.id_to_token(i) for i in ids] + ["</s>"]

    def _decode(self, tokens: list[str]) -> str:
        ids = [self.tokenizer.token_to_id(t) for t in tokens if t not in NLLB_CODES]
        return self.tokenizer.decode([i for i in ids if i is not None],
                                     skip_special_tokens=True).strip()

    def _sources(self, sentences: list[str], source: str | None, target: str) -> list[str]:
        if source is None:
            return source_languages(sentences, self.detector.detect, self.detector.detect_text)

        def clearly_target(sentence: str) -> bool:
            if len(sentence) < MIN_DETECT_CHARS:
                return False
            lang, p = self.detector.detect(sentence)
            return lang == target and p >= MIN_CONFIDENCE

        # picked by the user: every sentence, except those already clearly in the target
        return [target if clearly_target(s) else source for s in sentences]

    def guess(self, text: str) -> list[str]:
        """The source languages auto-detection would use, most frequent first (the picker
        shows them next to "Auto")."""
        sentences = [s for _, line, _ in segment(text) for s in line]
        sources = source_languages(sentences, self.detector.detect, self.detector.detect_text)
        return sorted(dict.fromkeys(sources), key=lambda code: -sources.count(code))

    def translate(self, text: str, target: str, source: str | None = None) -> str:
        """Translate text into `target` (NLLB code, e.g. "deu_Latn"), keeping its layout.
        source: the text's language if the user picked it, None = detect per sentence."""
        if target not in NLLB_CODES or (source is not None and source not in NLLB_CODES):
            raise ValueError(f"unknown language {target!r} / {source!r}")
        rows = segment(text)
        sentences = [s for _, line, _ in rows for s in line]
        sources = self._sources(sentences, source, target)
        todo = [i for i, src in enumerate(sources) if src != target]
        out = list(sentences)
        if todo:
            batch = [self._tokens(sentences[i], sources[i]) for i in todo]
            with self._lazy.use() as model: # waits for a running load, blocks an unload
                results = model.translate_batch(
                    batch, target_prefix=[[target]] * len(batch), beam_size=self.beam_size,
                    max_decoding_length=min(512, 2 * max(map(len, batch)) + 10))
            for i, r in zip(todo, results):
                out[i] = tidy(sentences[i], self._decode(r.hypotheses[0]), target)
        it = iter(out)
        joiner = "" if target in ("jpn_Jpan", "zho_Hans", "zho_Hant", "yue_Hant") else " "
        return "\n".join(prefix + joiner.join(next(it) for _ in line) + suffix
                         for prefix, line, suffix in rows)
