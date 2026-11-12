"""Your own recordings with the correct text, and the scoring of engines against them.

A recording is a 16 kHz WAV plus a JSON file with the reference text (what you really said) in
bench_audio/real/ (git-ignored: it is your voice). In the reference, mark names with [[double brackets]], e.g.
"Ich habe mit [[Claude]] gearbeitet": the brackets are not part of the text, and the marked words are
counted separately (names are a weak spot of every model). The output of each engine for each recording is
cached in bench_audio/real/hyp/<engine>.json, so a recording is transcribed once per engine.

scripts/compare_gui.py records and shows the results; scripts/real_eval.py runs engines over everything.
"""
from __future__ import annotations

import json
import os
import re
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = Path(os.environ.get("WC_REAL_DIR", ROOT / "bench_audio" / "real"))
SAMPLE_RATE = 16_000
NAME_MARK = re.compile(r"\[\[(.+?)\]\]")
_NUMBERS = {w: str(i) for i, ws in enumerate(
    [("null", "zero"), ("eins", "one"), ("zwei", "two"), ("drei", "three"), ("vier", "four"), ("fünf", "five"),
     ("sechs", "six"), ("sieben", "seven"), ("acht", "eight"), ("neun", "nine"), ("zehn", "ten")]) for w in ws}


@dataclass
class Recording:
    """One recording: id (also the file name), WAV path, reference text (with [[names]]) and a note."""

    id: str
    wav: Path
    reference: str = ""
    note: str = ""

    @property
    def meta(self) -> Path:
        return self.wav.with_suffix(".json")


_SENTENCE_END = r"[.!?]+[\"'”»)\]]*" # sentence-ending punctuation, optionally followed by closing quotes/brackets


def _tokens(text: str) -> list[tuple[str, str, bool]]:
    """Per word: (normalized word, original word, followed by a sentence end). Same normalization as words()."""
    text = text.replace("-", " ").replace("’", "'")
    raw = list(re.finditer(r"[\w']+", text))
    out: list[tuple[str, str, bool]] = []
    for k, m in enumerate(raw):
        gap = text[m.end(): raw[k + 1].start() if k + 1 < len(raw) else len(text)]
        ends = re.fullmatch(_SENTENCE_END + r"\s+", gap) or (k + 1 == len(raw) and re.fullmatch(_SENTENCE_END + r"\s*", gap))
        pieces = [_NUMBERS.get(p, p) for p in re.findall(r"[\w']+", m.group(0).lower())]
        for n, piece in enumerate(pieces): # lower-casing can split a word (e.g. "İ"); words() always did the same
            out.append((piece, m.group(0), bool(ends) and n == len(pieces) - 1))
    return out


def words(text: str) -> list[str]:
    """Lower-case words without punctuation; number words as digits ("two" and "2" both count)."""
    return [w for w, _, _ in _tokens(text)]


def parse_reference(text: str) -> tuple[str, list[bool]]:
    """(text without the brackets, a flag per word of it: True for words marked as names)."""
    flags: list[bool] = []
    for i, part in enumerate(NAME_MARK.split(text)): # odd parts are the marked names
        flags += [i % 2 == 1] * len(words(part))
    return NAME_MARK.sub(r"\1", text), flags


# ---------- storage ----------

def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit WAV as float32."""
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, f"{path.name}: need 16 kHz mono"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def save_recording(audio: np.ndarray, directory: Path = REAL_DIR, *,
                   when: datetime | None = None) -> Recording:
    """Writes a new recording (named by the time, to the second: `when`, default now) with an
    empty reference. Two recordings in the same second need different `when` values."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    rec = Recording(stamp, directory / f"{stamp}.wav")
    write_wav(rec.wav, audio)
    save_reference(rec)
    return rec


def _write_json(path: Path, data: dict) -> None:
    """Writes the file in one step (temp file, then replace), so a reader in another thread never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(50): # Windows refuses the swap while another thread has the file open
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.01)


def save_reference(rec: Recording, reference: str | None = None, note: str | None = None) -> None:
    """Stores the reference text (and note) next to the WAV."""
    rec.reference = rec.reference if reference is None else reference
    rec.note = rec.note if note is None else note
    _write_json(rec.meta, {"reference": rec.reference, "note": rec.note})


def load_recordings(directory: Path = REAL_DIR) -> list[Recording]:
    """All recordings in the folder, oldest first."""
    found = []
    for wav in sorted(directory.glob("*.wav")):
        rec = Recording(wav.stem, wav)
        if rec.meta.exists():
            data = json.loads(rec.meta.read_text(encoding="utf-8"))
            rec.reference, rec.note = data.get("reference", ""), data.get("note", "")
        found.append(rec)
    return found


def hyp_path(engine: str, directory: Path = REAL_DIR) -> Path:
    return directory / "hyp" / f"{engine.replace(':', '_')}.json"


def load_hyps(engine: str, directory: Path = REAL_DIR) -> dict[str, dict]:
    """{recording id: {"text", "s"}} of an engine ({} if it has not run)."""
    path = hyp_path(engine, directory)
    for attempt in range(50): # the file may be refused for a moment while another thread swaps it
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.01)
    return {}


def save_hyp(engine: str, rec_id: str, text: str, seconds: float, directory: Path = REAL_DIR) -> None:
    """Adds one engine output to the engine's cache."""
    hyps = load_hyps(engine, directory)
    hyps[rec_id] = {"text": text, "s": round(seconds, 3)}
    _write_json(hyp_path(engine, directory), hyps)


# ---------- scoring ----------

def _align(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int]]:
    """Levenshtein alignment: [(op, ref index, hyp index)], op in "=", "sub", "del", "ins" (-1: none)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            out.append(("del", i - 1, -1))
            i -= 1
        else:
            out.append(("ins", -1, j - 1))
            j -= 1
    return out[::-1]


def evaluate(reference: str, hypothesis: str) -> dict:
    """Errors of one hypothesis against a reference (which may mark names): counts by kind, name errors.

    Also counts over the CORRECT words (matched after normalization, so they are not errors):
    `casing` = matched words whose original capitalization differs (e.g. "Haus" vs "haus"); a word that starts a
    sentence in the reference is skipped, since its capital comes from the sentence. `breaks` = matched words after
    which exactly one side has a sentence end (. ! ? directly after the word, then a space or the end).
    """
    clean, name_flags = parse_reference(reference)
    ref, hyp = _tokens(clean), _tokens(hypothesis)
    ops = _align([w for w, _, _ in ref], [w for w, _, _ in hyp])
    kinds = [op for op, _, _ in ops]
    wrong_names = sum(1 for op, i, _ in ops if i >= 0 and name_flags[i] and op != "=")
    casing = breaks = 0
    for op, i, j in ops:
        if op != "=":
            continue
        (_, ref_form, ref_end), (_, hyp_form, hyp_end) = ref[i], hyp[j]
        starts_sentence = i == 0 or ref[i - 1][2]
        if ref_end != hyp_end:
            breaks += 1
        if not starts_sentence and ref_form != hyp_form and ref_form.lower() == hyp_form.lower():
            casing += 1
    return {"words": len(name_flags), "sub": kinds.count("sub"), "del": kinds.count("del"), "ins": kinds.count("ins"),
            "errors": len(kinds) - kinds.count("="), "names": sum(name_flags), "names_wrong": wrong_names,
            "casing": casing, "breaks": breaks}


def markup(reference: str, hypothesis: str) -> tuple[list[tuple[str, str]], list[str]]:
    """The hypothesis as [(text piece, "ok" | "sub" | "ins")] for highlighting, and the reference words it lacks."""
    clean, _ = parse_reference(reference)
    ref_words = words(clean)
    ops = _align(ref_words, words(hypothesis))
    status = [("ok" if op == "=" else op) for op, _, j in ops if j >= 0]
    missing = [ref_words[i] for op, i, _ in ops if op == "del"]
    pieces, k = [], 0
    for match in re.finditer(r"\s*\S+", hypothesis):
        count = len(words(match.group(0)))
        kinds = status[k:k + count] or ["ok"]
        k += count
        pieces.append((match.group(0), "ins" if "ins" in kinds else "sub" if "sub" in kinds else "ok"))
    return pieces, missing
