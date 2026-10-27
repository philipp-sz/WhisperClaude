"""The last few dictations, kept in a local file so a dictation is never lost.

Seen in real use: a 15-minute dictation was pasted while no text field had focus, and the
clipboard restore then erased the only copy. Every transcript is now saved here *before* it is
pasted; the tray menu "Recent dictations" and the pill's "Copy" button read from it.

The file lives in %LOCALAPPDATA%\\WhisperClaude (readable only by the Windows user, never in the
repository, never uploaded). `[history] keep` in config.toml sets how many are kept; 0 turns
it off and deletes the file. The log still never contains transcript text.
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


@dataclass(frozen=True)
class Entry:
    time: float  # Unix time
    text: str

    def label(self, width: int = 45) -> str:
        """Menu label: "12:20 · Ich glaube, ich würde es jetzt…  (11,209 characters)"."""
        one_line = " ".join(self.text.split())
        preview = one_line if len(one_line) <= width else one_line[:width - 1].rstrip() + "…"
        when = time.strftime("%H:%M", time.localtime(self.time))
        if time.localtime(self.time)[:3] != time.localtime()[:3]:  # not today
            when = time.strftime("%d.%m. %H:%M", time.localtime(self.time))
        size = f"  ({len(self.text):,} characters)" if len(one_line) > width else ""
        return f"{when} · {preview}{size}"


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
            self._entries = [Entry(float(e["time"]), str(e["text"])) for e in raw][:keep]
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

    def add(self, text: str) -> None:
        """Remember a dictation (and save the file) before it is pasted."""
        if self.keep <= 0 or not text.strip():
            return
        with self._lock:
            self._entries = [Entry(time.time(), text)] + self._entries[:self.keep - 1]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self.path.unlink(missing_ok=True)

    def _save(self) -> None:
        """Write atomically: a crash mid-write never leaves a broken file."""
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps([{"time": e.time, "text": e.text} for e in self._entries],
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            log.exception("could not save the dictation history")
