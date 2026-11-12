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
from collections.abc import Callable
from pathlib import Path

from whisperclaude.languages import NLLB_CODES, SOURCE_CODES

log = logging.getLogger(__name__)

MODELS = { # config name -> pre-converted CTranslate2 model (int8) on Hugging Face
    "nllb-600m": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
    "nllb-1.3b": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
}
LOW_VRAM_MB = 2048 # "auto" on CUDA with less free VRAM (after Whisper): the smaller model
MIN_RAM_GB = 12 # "auto" on the CPU with less RAM: the smaller model. A "16 GB" PC reports about
# 15.5 GiB (the rest is reserved), an "8 GB" one about 7.5 GiB. See docs/translation.md
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


def user_languages() -> list[str]:
    """ISO codes of the user's own languages, most likely first: the Windows display
    language, the keyboard layouts, then English."""
    import ctypes

    def iso(langid: int) -> str:
        name = ctypes.create_unicode_buffer(85)
        ctypes.windll.kernel32.LCIDToLocaleName(langid & 0xFFFF, name, 85, 0)
        return name.value.split("-")[0].lower()

    codes = [iso(ctypes.windll.kernel32.GetUserDefaultUILanguage())]
    try:
        user32 = ctypes.windll.user32
        n = user32.GetKeyboardLayoutList(0, None)
        layouts = (ctypes.c_void_p * n)()
        user32.GetKeyboardLayoutList(n, layouts)
        codes += [iso(h or 0) for h in layouts]
    except OSError:
        pass
    codes.append("en")
    return [c for i, c in enumerate(codes) if c in SOURCE_CODES and c not in codes[:i]]


@functools.lru_cache(maxsize=1)
def _identifier():
    """The py3langid model restricted to SOURCE_CODES, loaded once (~0.5 s) and shared.
    set_languages() mutates the identifier, but only here and always with the same list, so
    sharing is safe. Per-instance state (`preferred`) stays on LanguageDetector."""
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

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


def total_ram_gb() -> float | None:
    """Installed RAM in GiB (what Windows makes usable); None if unknown."""
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:
        log.info("could not read the RAM size", exc_info=True)
        return None


def auto_model(device: str, free_vram_mb: int | None, ram_gb: float | None) -> str:
    """The model for `model = "auto"`: nllb-1.3b (1.4 GB, better) unless memory is short. On
    CUDA that is less than LOW_VRAM_MB free after Whisper (unknown VRAM: still the large
    one); on the CPU less than MIN_RAM_GB of RAM (unknown RAM: the small one)."""
    if device == "cuda":
        small = free_vram_mb is not None and free_vram_mb < LOW_VRAM_MB
    else:
        small = ram_gb is None or ram_gb < MIN_RAM_GB
    return "nllb-600m" if small else "nllb-1.3b"


class Translator:
    """NLLB-200 on CTranslate2. translate() is not thread-safe (one worker thread uses it)."""

    def __init__(self, model: str = "auto", device: str = "auto", cpu_threads: int = 4,
                 beam_size: int = 2, on_status: Callable[[str], None] = lambda s: None,
                 languages: tuple[str, ...] = ()) -> None:
        """model "auto": by memory, see auto_model. languages: the user's own languages (ISO
        codes, most used first); () = Windows'."""
        import ctranslate2
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self.device = "cpu"
        if device in ("auto", "cuda"):
            from whisperclaude.transcriber import cuda_available
            if cuda_available():
                self.device = "cuda"
            elif device == "cuda":
                log.warning("no usable NVIDIA GPU for the translation, using the CPU")
        if model == "auto":
            from whisperclaude.transcriber import free_vram_mb
            model = auto_model(self.device, free_vram_mb() if self.device == "cuda" else None,
                               total_ram_gb())
        self.model_name = model
        repo = MODELS.get(self.model_name, self.model_name)
        try: # cached: no network request at startup
            path = snapshot_download(repo, local_files_only=True)
        except Exception:
            on_status("Downloading translation model (one-time)…")
            path = snapshot_download(repo)
        compute = "int8_float16" if self.device == "cuda" else "int8"
        self.model = ctranslate2.Translator(path, device=self.device, compute_type=compute,
                                            intra_threads=cpu_threads)
        self.tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self.beam_size = beam_size
        self.detector = LanguageDetector(list(languages) or None)
        self.translate("Hallo.", "eng_Latn") # first call sets up the device: pay it here

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
            results = self.model.translate_batch(
                batch, target_prefix=[[target]] * len(batch), beam_size=self.beam_size,
                max_decoding_length=min(512, 2 * max(map(len, batch)) + 10))
            for i, r in zip(todo, results):
                out[i] = tidy(sentences[i], self._decode(r.hypotheses[0]), target)
        it = iter(out)
        joiner = "" if target in ("jpn_Jpan", "zho_Hans", "zho_Hant", "yue_Hant") else " "
        return "\n".join(prefix + joiner.join(next(it) for _ in line) + suffix
                         for prefix, line, suffix in rows)
