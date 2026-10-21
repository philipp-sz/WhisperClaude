"""Small, predictable fixes to Whisper's text, applied where chunk transcripts are joined.

- A chunk starts where the VAD cut at a pause. If you paused mid-sentence, Whisper still starts
  the new chunk with a capital ("...in verschiedenen Sprachen aufnehmen | Und wir werden...").
  When the previous chunk doesn't end a sentence and the new one starts with a linking word
  (und, oder, um, and, but, ...), that word is lowercased. Only linking words: they never need
  a capital mid-sentence, while nouns (German) and "I" (English) do. Giving Whisper the
  previous chunk as context fixed this too, but made it translate English after German.
- The English pronoun "i" (seen in real dictations: "because i had") becomes "I".
"""
from __future__ import annotations

import re
from collections.abc import Iterable

LINKING_WORDS = frozenset((
    # German conjunctions and prepositions
    "und oder aber sondern denn dass weil wenn als ob um damit sodass also bis während obwohl "
    "falls nachdem bevor sowie wie zu mit für von auf in an im am zum zur bei nach über unter "
    "durch gegen ohne sowohl beziehungsweise "
    # English conjunctions and prepositions ("I" is never one)
    "and or but so because that which who whom whose when where while if than then though "
    "although unless until whether to for with of in on at as by from about into"
).split())

# Chunk text that ends a sentence (or a spoken formatting command, which starts a new line).
_ENDS_SENTENCE = re.compile(
    r"(?:[.!?…:;\"»“”)\]]|new[\s-]?line|new[\s-]?paragraph|bullet(?:[\s-]?points?)?|separator)"
    r"\s*$", re.IGNORECASE)
_FIRST_WORD = re.compile(r"[^\W\d_]+")
_PRONOUN_I = re.compile(r"(?<![\w.'’-])i(?=(?:['’](?:m|ve|ll|d))?(?![\w.'’-]))")


def fix_pronoun_i(text: str) -> str:
    """'i' standing alone (also i'm, i've, i'll, i'd) -> 'I'. Not i.e., i-Punkt, iPhone."""
    return _PRONOUN_I.sub("I", text)


def join_chunks(texts: Iterable[str]) -> str:
    """Join chunk transcripts with spaces, applying the fixes above."""
    out = ""
    for text in texts:
        text = fix_pronoun_i(text.strip())
        if not text:
            continue
        if out and not _ENDS_SENTENCE.search(out):
            first = _FIRST_WORD.match(text)
            if first and first.group().lower() in LINKING_WORDS and text[0].isupper():
                text = text[0].lower() + text[1:]
        out = f"{out} {text}" if out else text
    return out
