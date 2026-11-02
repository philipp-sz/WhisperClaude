"""The last few dictations, kept in a local file so a dictation is never lost.

Seen in real use: a 15-minute dictation was pasted while no text field had focus, and the
clipboard restore then erased the only copy. Every transcript is now saved here *before* it is
pasted; the tray menu "Recent dictations" and the pill's "Copy" button read from it.

The file lives in %LOCALAPPDATA%\\WhisperClaude (readable only by the Windows user, never in the
repository, never uploaded). `[history] keep` in config.toml sets how many are kept; 0 turns
it off and deletes the file. The log still never contains transcript text.

A translated dictation (or selection) is one entry holding both texts, so either can be
copied back.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)
FILE_NAME = "history.json"


def preview(text: str, width: int = 42) -> str:
    """The start of a text on one line: "Ich glaube, ich würde es jetzt auch…"."""
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[:width - 1].rstrip() + "…"


@dataclass(frozen=True)
class Entry:
    time: float  # Unix time
    text: str  # what was dictated (or selected)
    translation: str = ""  # its translation, if it was translated
    target: str = ""  # NLLB code of the translation's language, e.g. "fra_Latn"

    def preview(self, width: int = 42) -> str:
        return preview(self.text, width)

    def when(self) -> str:
        """"12:20", "24.10. 12:20" if not today; long ones with their size: "12:20 · 11k"."""
        when = time.strftime("%H:%M", time.localtime(self.time))
        if time.localtime(self.time)[:3] != time.localtime()[:3]:
            when = time.strftime("%d.%m. %H:%M", time.localtime(self.time))
        if len(self.text) >= 2000:
            when += f" · {len(self.text) // 1000}k"
        return when


class History:
    """Newest first. Thread-safe: the worker adds, the tray thread reads."""

    def __init__(self, folder: Path, keep: int = 5) -> None:
        self.path = folder / FILE_NAME
        self.keep = keep
        self._lock = threading.Lock()
        self._entries: list[Entry] = []
        if keep <= 0:
            self.path.unlink(missing_ok=True)  # turned off: don't leave old dictations behind
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = [Entry(float(e["time"]), str(e["text"]), str(e.get("translation", "")),
                                   str(e.get("target", ""))) for e in raw][:keep]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("history file unreadable, starting a new one")

    def entries(self) -> list[Entry]:
        with self._lock:
            return list(self._entries)

    def latest(self) -> Entry | None:
        with self._lock:
            return self._entries[0] if self._entries else None

    def add(self, text: str, translation: str = "", target: str = "") -> None:
        """Remember a dictation (and save the file) before it is pasted. With a translation:
        if the newest entry is this text without one (a dictation saved before it was
        translated), the translation is added to it, else it is a new entry."""
        if self.keep <= 0 or not text.strip():
            return
        with self._lock:
            newest = self._entries[0] if self._entries else None
            if translation and newest and newest.text == text and not newest.translation:
                self._entries[0] = Entry(newest.time, text, translation, target)
            else:
                entry = Entry(time.time(), text, translation, target if translation else "")
                self._entries = [entry] + self._entries[:self.keep - 1]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self.path.unlink(missing_ok=True)

    def _save(self) -> None:
        """Write atomically: a crash mid-write never leaves a broken file."""
        tmp = self.path.with_suffix(".tmp")
        try:
            rows = [{"time": e.time, "text": e.text}
                    | ({"translation": e.translation, "target": e.target} if e.translation else {})
                    for e in self._entries]
            tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            log.exception("could not save the dictation history")
