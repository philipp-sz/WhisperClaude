"""Context-aware insertion: a leading space when the text before the caret needs one.

Whisper's text has no leading space, so a second dictation pasted behind the first ran into it
("works.Next sentence"). What stands before the caret is learned in two ways:

1. UI Automation (`uia.CaretReader`): the characters before the caret, exact, in apps that expose
   their text (Notepad, Chromium browsers and Electron apps, Windows Terminal, most native
   controls). Not in apps that draw their own text (Tk, games, some remote desktops).
2. Fallback, `PasteTracker`: our own last paste. If the next paste goes to the same window and no
   key (other than modifiers and our own hotkeys) and no mouse click happened since, the caret
   is still where our text ended, so that text's last character is "the text before the caret".

`adapt()` is the pure decision on top. Privacy: the context is used in memory only, never logged.
The history keeps the transcript unchanged, only the pasted text differs.
"""
from __future__ import annotations

import ctypes
import logging
import re
import time
from collections.abc import Callable

from whisperclaude.textfix import LINKING_WORDS

log = logging.getLogger(__name__)

# No space after these: the dictation continues inside a bracket, or after a path/mention mark.
_OPENERS = frozenset("([{<„¿¡/\\@#$~")
# No space before a dictation that starts with these (closing punctuation).
_CLOSERS = frozenset(".,;:!?…)]}%")
_FIRST_WORD = re.compile(r"[^\W\d_]+")
_SKIPPED_BLANKS = " \t " # not a line break: after one the next sentence starts anew

# Words that are never a noun or a name in German or English, so lowercasing their first letter
# is safe. German nouns are capitalized, "I" and names are too: those are never touched.
FUNCTION_WORDS = LINKING_WORDS | frozenset((
    "der die das den dem des ein eine einen einem einer ich du er es wir ihr nicht auch nur "
    "noch schon dann jetzt hier da wo was warum "
    "the a an this these those it he she we you they my your our their his its there here "
    "what how why also just not then now"
).split())

LAST_CHARS = 1 # characters before the caret the decision needs
CASE_CHARS = 6 # ... if the case is adapted too (the last letter before blanks)


def needs_space(before: str | None, text: str) -> bool:
    """True if `text` needs a space in front, given the characters before the caret.

    before: None = unknown, "" = the caret is at the very start of the text.
    """
    if not before or not text:
        return False
    last = before[-1]
    if last.isspace() or last in _OPENERS:
        return False
    return not (text[0].isspace() or text[0] in _CLOSERS)


def mid_sentence(before: str | None) -> bool:
    """True if the text before the caret says that the sentence goes on: the last thing before
    blanks is a letter, a comma or a semicolon, on the same line."""
    if not before:
        return False
    last = before.rstrip(_SKIPPED_BLANKS)[-1:]
    return bool(last) and (last.isalpha() or last in ",;")


def lowercase_first(text: str) -> str:
    """Lowercase the first letter if the first word is a function word ("The" -> "the", but
    never "Haus", "I", "IT" or a name)."""
    match = _FIRST_WORD.match(text)
    if not match:
        return text
    word = match.group()
    if word[0].isupper() and word[1:].islower() and word.lower() in FUNCTION_WORDS:
        return word.lower() + text[match.end():]
    return text


def adapt(text: str, before: str | None, adapt_case: bool = False) -> str:
    """The text as it should be pasted behind `before` (see needs_space, mid_sentence)."""
    if adapt_case and mid_sentence(before):
        text = lowercase_first(text)
    return " " + text if needs_space(before, text) else text


# ---------- fallback: our own last paste ----------

_MODIFIER_NAMES = ("ctrl", "shift", "alt", "cmd", "alt_gr")


def is_modifier(key: object) -> bool:
    """pynput key -> True for Ctrl/Shift/Alt/Win: pressed alone they change neither text nor
    caret, and the record/translate keys are among them."""
    return (getattr(key, "name", "") or "").split("_")[0] in _MODIFIER_NAMES


class PasteTracker:
    """Remembers our last paste, and whether the user touched keyboard or mouse since.

    note_input() is called from the hook threads: a bare counter increment, nothing else.
    """

    MAX_AGE_S = 300.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._inputs = 0
        self._window = 0
        self._last = ""
        self._inputs_at_paste = 0
        self._at = 0.0

    def note_input(self) -> None:
        self._inputs += 1

    def record(self, window: int, text: str) -> None:
        """Call right after a paste into `window` (the counter is read then, so the keys of the
        paste itself don't count)."""
        self._window, self._last = window, text[-CASE_CHARS:]
        self._inputs_at_paste, self._at = self._inputs, self.clock()

    def guess_before(self, window: int) -> str | None:
        """The text before the caret as far as we can tell, or None if we can't."""
        if (self._last and window and window == self._window
                and self._inputs == self._inputs_at_paste
                and self.clock() - self._at <= self.MAX_AGE_S):
            return self._last
        return None


def foreground_window() -> int:
    return int(ctypes.windll.user32.GetForegroundWindow() or 0)


class SmartPaster:
    """Wraps the paste function: adapts the text to what is before the caret, then pastes.

    Runs on the worker thread that pastes. Anything unexpected pastes the text unchanged.
    """

    def __init__(self, paste: Callable[[str], None],
                 read_before: Callable[[int], str | None], tracker: PasteTracker,
                 adapt_case: bool = False,
                 window: Callable[[], int] = foreground_window) -> None:
        self.paste, self.read_before, self.tracker = paste, read_before, tracker
        self.adapt_case, self.window = adapt_case, window

    def __call__(self, text: str) -> None:
        window, adapted, source = 0, text, "uia"
        try:
            window = self.window()
            before = self.read_before(CASE_CHARS if self.adapt_case else LAST_CHARS)
            if before is None:
                before = self.tracker.guess_before(window)
                source = "last paste" if before is not None else "unknown"
            adapted = adapt(text, before, self.adapt_case)
        except Exception:
            log.exception("could not adapt the text to the caret context, pasting it as is")
            adapted = text
        if adapted != text:
            log.info("insertion adapted to the context (%s)", source) # never the text itself
        self.paste(adapted)
        try:
            self.tracker.record(window, adapted)
        except Exception:
            log.exception("could not remember the paste")
