"""Automatic vocabulary: words from the user's own texts that Whisper probably doesn't know.

The manual vocabulary list works (names wrong 18 -> 9 of 68 on the author's recordings) but
nobody can list hundreds of names and terms by hand. Instead, text files the user names in
[model] vocabulary_sources (notes, project docs) are scanned at each start for words that

- appear capitalized (names, products, German nouns) and not all in capitals,
- occur at least MIN_COUNT times (weighted: a file's count halves every HALF_LIFE_DAYS of age,
  so the list follows current work and old projects fade out), and
- are split into several pieces by Whisper's tokenizer even in lower case: a word that is one
  token is common in Whisper's training data and needs no help.

They are ranked by weighted count x extra pieces and appended to the prompt within a token
budget. Tested on the author's recordings: 26 words from the project docs and notes gave 6
instead of 9 wrong names, without curation (docs/real-recordings.md). Nothing is stored or
logged except the number of words; everything stays on the machine.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

MAX_WORDS = 40
PROMPT_TOKEN_BUDGET = 200 # Whisper's prompt holds 223 tokens; the rest is headroom
MIN_COUNT = 3.0
HALF_LIFE_DAYS = 30.0
MAX_FILES = 2000
MAX_TOTAL_BYTES = 20_000_000
MAX_FILE_BYTES = 1_000_000
SUFFIXES = {".md", ".txt"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", "site-packages", "build", "dist"}

_NOISE = re.compile(r"```.*?```|`[^`\n]*`|\]\([^)]*\)|https?://\S+", re.DOTALL) # code, link targets, URLs
_WORD = re.compile(r"(?<![\w./\\-])([A-Z][\w+-]*[A-Za-z0-9](?: [A-Z][\w+-]*[A-Za-z0-9])?)(?![\w/\\])")


def text_files(sources: Iterable[str | Path]) -> Iterator[Path]:
    """The .md / .txt files under the sources (files or folders), skipping tool and build folders."""
    for source in sources:
        root = Path(source).expanduser()
        if root.is_file():
            yield root
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in SUFFIXES and not SKIP_DIRS.intersection(path.parts) and path.is_file():
                    yield path


def weighted_counts(files: Iterable[Path], now: float | None = None) -> Counter:
    """Candidate words -> count, each file's count weighted by its age; reads at most the caps."""
    now = time.time() if now is None else now
    counts: Counter = Counter()
    total = 0
    for i, path in enumerate(files):
        if i >= MAX_FILES or total >= MAX_TOTAL_BYTES:
            break
        try:
            size, age_days = path.stat().st_size, max(now - path.stat().st_mtime, 0) / 86_400
            if size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        total += size
        weight = 0.5 ** (max(age_days - 1.0, 0.0) / HALF_LIFE_DAYS) # the last day counts fully
        for word in _WORD.findall(_NOISE.sub(" ", text)):
            if not word.replace(" ", "").isupper():
                counts[word] += weight
    return counts


def rank(counts: Counter, pieces: Callable[[str], int], max_words: int = MAX_WORDS) -> list[str]:
    """The best candidates: frequent enough, and split by the tokenizer even in lower case."""
    scored = []
    for word, count in counts.items():
        if count < MIN_COUNT or len(word) < 3:
            continue
        extra = pieces(" " + word) - 1
        if extra >= 1 and pieces(" " + word.lower()) >= 2:
            scored.append((count * extra, word))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [word for _, word in scored[:max_words]]


def fit_budget(base: str, manual: Sequence[str], auto: Sequence[str], tokens: Callable[[str], int],
               budget: int = PROMPT_TOKEN_BUDGET) -> list[str]:
    """The automatic words (best first) that still fit next to the prompt and the manual words."""
    words = [w for w in auto if w.lower() not in {m.lower() for m in manual}]
    while words and tokens(f"{base} {', '.join([*manual, *words])}.") > budget:
        words.pop()
    return words


def auto_vocabulary(sources: Sequence[str | Path], base: str, manual: Sequence[str],
                    tokens: Callable[[str], int]) -> list[str]:
    """The automatic words for the prompt (empty without sources); never raises."""
    if not sources:
        return []
    try:
        t0 = time.perf_counter()
        words = fit_budget(base, manual, rank(weighted_counts(text_files(sources)), tokens), tokens)
        log.info("automatic vocabulary: %d words in %.2f s", len(words), time.perf_counter() - t0)
        return words
    except Exception:
        log.exception("automatic vocabulary failed; using the manual list only")
        return []
