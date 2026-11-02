"""Spoken formatting commands, applied to the finished transcript.

You say a trigger word followed by a command, e.g. "command new line", and the pair is
replaced by the formatting. The trigger is what keeps this predictable: ordinary sentences
("I added a new line to the file", "the hash table") never contain trigger + command, so
nothing else is ever touched, and no guessing about punctuation or position is needed.

    command new line        line break
    command new paragraph   blank line
    command bullet          new line starting with "- " (Markdown list item)
    command separator       Markdown separator "---" with a blank line above and below
    command open quote      "   (attaches to the next word)
    command close quote     "   (attaches to the previous word)

A bullet item ends at the end of its sentence (or at the next command); text that follows
starts after a blank line, which is also what ends a list in Markdown.

Only whole words are matched, with no fuzzy matching: a mis-heard command stays visible as
text, which is easy to fix, while a false hit would silently change what you said.
"""
from __future__ import annotations

import re

COMMANDS = ("new line", "new paragraph", "bullet", "separator", "open quote", "close quote")

# Command words as Whisper might write them ("new-line", "bullet point", "closed quote", ...).
_WORDS = (
    r"(?P<paragraph>new[\s-]?paragraph)"
    r"|(?P<newline>new[\s-]?line)"
    r"|(?P<bullet>bullet(?:[\s-]?points?)?)"
    r"|(?P<separator>separator)"
    r"|(?P<open>open(?:ing)?[\s-]?quote)"
    r"|(?P<close>clos(?:e|ed|ing)[\s-]?quote)"
)
# "command" right after one of these is a noun in an ordinary sentence ("the command bullet by
# bullet", "each command new line"), not a trigger.
_NOUN_BEFORE = re.compile(
    r"(?:^|[^\w'])(?:the|a|an|this|that|these|those|each|every|any|no|some|another|your|my|our|"
    r"their|his|her|its|which|der|die|das|den|dem|ein|eine|einen|dieser|diese|dieses|jeder|"
    r"jede|jedes|mein|dein|sein|unser)[ \t]+$", re.IGNORECASE)
_AFTER_BREAK = re.compile(r"[ \t]*[,.;:]?[ \t]*") # punctuation Whisper put after the command
_AFTER_OPEN = re.compile(r"\s*[,.;:]?\s*") # "open quote." -> the period is the command's
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _pattern(trigger: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(w) for w in trigger.split())
    # Between trigger and command only plain spaces are allowed: no period ("use this command.
    # New line characters..." is a normal sentence) and no comma ("every command, bullet and
    # quote is documented" is a normal list). Zero spaces is fine, Whisper sometimes glues the
    # words together ("commandnew line"). After the command a word boundary is required, so
    # "new lines" or "bullet-proof" don't match.
    return re.compile(
        rf"(?<![\w-]){words}[ \t]*(?:{_WORDS})(?=[\s,.;:!?]|$)", re.IGNORECASE)


class _Output:
    """The text built so far, and whether it is inside a bullet item or a quote."""

    def __init__(self) -> None:
        self.text = ""
        self.in_item = False # inside a bullet item: it ends at the end of its sentence
        self.quote_open = False # between open and close quote: sentence ends there don't count

    def add(self, segment: str) -> None:
        """Dictated text. A sentence end closes a bullet item; what follows starts after a
        blank line."""
        if self.in_item and not self.quote_open:
            end = _SENTENCE_END.search(segment)
            if end:
                head, rest = segment[:end.end()], segment[end.end():]
                self.in_item = False
                self.text += head
                self.text += "\n\n" + rest.lstrip() if rest.strip() else rest
                return
        self.text += segment

    def line_break(self, kind: str, bullet: str) -> None:
        """new line, new paragraph, bullet or separator."""
        out = self.text.rstrip(" \t")
        if out.endswith((",", ";")): # the pause before the command, not real punctuation
            out = out[:-1].rstrip(" \t")
        self.in_item = False
        if kind == "bullet":
            if out and not out.endswith("\n"):
                out += "\n"
            out += bullet
            self.in_item = True
        elif kind == "paragraph":
            out = out.rstrip("\n") + "\n\n" if out.strip("\n") else "\n\n"
        elif kind == "separator":
            out = out.rstrip("\n") + "\n\n---\n\n" if out.strip("\n") else "---\n\n"
        else:
            out += "\n"
        self.text = out

    def open_quote(self) -> None:
        if self.text and not self.text[-1].isspace() and self.text[-1] not in "([{":
            self.text += " "
        self.text += '"'
        self.quote_open = True

    def close_quote(self) -> None:
        out = self.text.rstrip(" \t")
        if out.endswith(","):
            out = out[:-1]
        self.text = out + '"'
        self.quote_open = False


def apply_commands(text: str, trigger: str = "command", bullet: str = "- ") -> str:
    """Replace "<trigger> <command>" pairs in the transcript. Text without any is returned as is.

    Line breaks produced by commands at the very start or end are kept (dictating just
    "command new line" pastes a line break).
    """
    pattern = _pattern(trigger)
    if not pattern.search(text):
        return text

    out, pos = _Output(), 0
    for m in pattern.finditer(text):
        if _NOUN_BEFORE.search(text[:m.start()]):
            continue # ordinary sentence about a command: leave the words alone
        if m.start() >= pos:
            out.add(text[pos:m.start()])
        kind, pos = m.lastgroup, m.end()
        if kind in ("newline", "paragraph", "bullet", "separator"):
            out.line_break(kind, bullet)
            pos = _AFTER_BREAK.match(text, pos).end() # "new line." -> the period is the command's
        elif kind == "open":
            out.open_quote()
            pos = _AFTER_OPEN.match(text, pos).end() # attach to the next word
        else: # close
            out.close_quote()
            if out.text[-2:-1] in (".", "!", "?") and text[pos:pos + 1] == ".":
                pos += 1 # the command's own period would double the one inside the quote
    out.add(text[pos:])
    return _without_item_periods(out.text, bullet)


def _without_item_periods(text: str, bullet: str) -> str:
    """List items conventionally have no final period."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(bullet):
            line = line.rstrip()
            if line.endswith(".") and not line.endswith(".."):
                line = line[:-1]
            lines[i] = line
    return "\n".join(lines)
