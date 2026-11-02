"""The translate key's parts for App (app.TranslationLike): selection capture, the language
picker, recently used languages and the translation model, which loads after Whisper."""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from whisperclaude.inserter import capture_selection
from whisperclaude.languages import BY_CODE, SOURCE_CODES
from whisperclaude.picker import Choice, LanguagePicker

if TYPE_CHECKING:  # heavy: loaded by the model loader, not on the UI path
    from whisperclaude.translator import Translator

log = logging.getLogger(__name__)
KEEP_RECENT = 5
LOAD_TIMEOUT_S = 30 # loading takes 2-5 s; the first start downloads ~0.6 GB first


class RecentLanguages:
    """Last picked target and source languages, newest first
    (%LOCALAPPDATA%\\WhisperClaude\\translate.json). Only language codes: no text is ever
    stored here."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.codes = [c for c in raw.get("recent", []) if c in BY_CODE][:KEEP_RECENT]
            self.sources = [c for c in raw.get("sources", []) if c in BY_CODE][:KEEP_RECENT]
        except (OSError, ValueError, AttributeError):
            self.codes, self.sources = [], []

    def add(self, code: str) -> None:
        self.codes = [code] + [c for c in self.codes if c != code][:KEEP_RECENT - 1]
        self._save()

    def add_source(self, code: str) -> None:
        self.sources = [code] + [c for c in self.sources if c != code][:KEEP_RECENT - 1]
        self._save()

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps({"recent": self.codes, "sources": self.sources}),
                                 encoding="utf-8")
        except OSError:
            log.warning("could not save the recent languages", exc_info=True)


class Translation:
    def __init__(self, picker: LanguagePicker, recent: RecentLanguages,
                 key_label: str = "Alt", my_languages: tuple[str, ...] = ()) -> None:
        """my_languages: [translation] my_languages (ISO codes), offered first as sources
        until the model is loaded (then its detector's list, which falls back to Windows)."""
        self.picker = picker
        self.recent = recent
        self.key_label = key_label
        self.my_languages = my_languages
        picker.recent = lambda: recent.codes
        picker.recent_sources = lambda: recent.sources
        picker.own = self.own_sources
        self._translator = None
        self._error = ""
        self._loaded = threading.Event()

    # model loader thread
    def set_translator(self, translator: Translator) -> None:
        self._translator = translator
        self._loaded.set()

    def set_failed(self, error: str) -> None:
        self._error = error or "could not load the translation model"
        self._loaded.set()

    def own_sources(self) -> list[str]:
        """NLLB codes of the user's own languages."""
        translator = self._translator
        iso = translator.detector.preferred if translator is not None else self.my_languages
        return [SOURCE_CODES[c] for c in iso if c in SOURCE_CODES]

    # app.TranslationLike
    def capture_selection(self) -> str | None:
        return capture_selection()

    def guess(self, text: str) -> list[str]:
        """Worker thread: the languages auto-detection would use; [] while loading."""
        translator = self._translator
        if translator is None:
            return []
        try:
            return translator.guess(text)
        except Exception:
            log.exception("language detection failed")
            return []

    def pick(self, on_done: Callable[[Choice | None, bool], None], dictation: bool = False,
             guess: list[str] | tuple[str, ...] = ()) -> None:
        def done(choice: Choice | None, focus_back: bool) -> None:
            if choice is not None:
                self.recent.add(choice[0])
                if choice[1] is not None:
                    self.recent.add_source(choice[1])
            on_done(choice, focus_back)

        esc = "paste original" if dictation else "cancel"
        self.picker.open(done, f"Enter or {self.key_label}: translate · Tab: from · Esc: {esc}",
                         guess)

    def set_guess(self, codes: list[str]) -> None:
        self.picker.set_guess(codes)

    def confirm_pick(self) -> None:
        self.picker.confirm()

    def close_pick(self) -> None:
        self.picker.close()

    def translate(self, text: str, target: str, source: str | None = None) -> str:
        if not self._loaded.wait(LOAD_TIMEOUT_S):
            raise RuntimeError("the translation model is still loading (the first start "
                               "downloads it), try again in a minute")
        if self._translator is None:
            raise RuntimeError(self._error)
        return self._translator.translate(text, target, source)
