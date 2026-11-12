"""Small, predictable fixes to Whisper's text, applied where chunk transcripts are joined.

- A chunk starts where the VAD cut at a pause. If you paused mid-sentence, Whisper still starts
  the new chunk with a capital ("...in verschiedenen Sprachen aufnehmen | Und wir werden...").
  When the previous chunk doesn't end a sentence and the new one starts with a linking word
  (und, oder, um, and, but, ...), that word is lowercased. Only linking words: they never need
  a capital mid-sentence, while nouns (German) and "I" (English) do. Giving Whisper the
  previous chunk as context fixed this too, but made it translate English after German.
- The English pronoun "i" (seen in real dictations: "because i had") becomes "I".
- German "Model" / "Chatmodel" become "Modell" / "Chatmodell", but only in sentences that are German
  (Whisper writes the German noun in English spelling; English "model" is never touched).
- English "a" before a vowel sound becomes "an" ("a app" -> "an app"). "a one", "a European" and
  capitalized words (acronyms, "I") are left alone.
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

# Sentence-level language guess for "Model": a few function words per language are enough.
_DE_WORDS = frozenset(
    "der die das den dem des und ist sind nicht ich wir ein eine einen einem mit für auf zu ja auch "
    "man dass aber wenn haben hat noch oder bei nach sich es sie wie wird werden kann können muss "
    "müssen habe bin ob weil".split())
_EN_WORDS = frozenset(
    "the and is are of it that this with but be have you we they what would could should which am "
    "not i'm it's i've don't that's there".split())
_SENTENCE = re.compile(r"[^.!?;:…\n]+")
_MODEL = re.compile(r"\b(?:Chatmodel|chatmodel|Model)\b")

# "a" before a vowel letter. "one", "once" and "eu..." start with a consonant sound; "hour", "honest"
# and friends have a silent h, so they take "an".
_A_BEFORE_VOWEL = re.compile(
    r"\b([Aa])(\s+)([aeioAEIO]\w*|hour\w*|honest\w*|honor\w*|honour\w*|heir\w*)\b")
_A_EXCEPTIONS = ("one", "once", "eu", "ewe")
_A_NEVER_ARTICLE = frozenset(("and", "or", "are", "is")) # "vitamin a and b", "plan a or b"
# Function words of the languages where a standalone "a" is a preposition or a verb ("voy a
# escribir", "il a entendu", "vado a ordinare"): such sentences are left alone.
_NOT_EN_WORDS = frozenset(
    "un una el la los las que por para con del pero es muy le les des est je nous vous il elle pas "
    "dans avec pour mais che di sono non per gli della voy vado hay".split())


def fix_pronoun_i(text: str) -> str:
    """'i' standing alone (also i'm, i've, i'll, i'd) -> 'I'. Not i.e., i-Punkt, iPhone."""
    return _PRONOUN_I.sub("I", text)


def _is_german(sentence: str) -> bool:
    """True if the sentence has more German than English function words (and at least one)."""
    words = re.findall(r"[\w'’]+", sentence.lower())
    de = sum(w in _DE_WORDS for w in words)
    return de > 0 and de > sum(w in _EN_WORDS for w in words)


def fix_model_de(text: str) -> str:
    """'Model' / 'Chatmodel' -> 'Modell' / 'Chatmodell', in German sentences only."""
    def fix_sentence(match: re.Match[str]) -> str:
        sentence = match.group()
        if not _is_german(sentence):
            return sentence
        return _MODEL.sub(lambda m: m.group().replace("Model", "Modell").replace("model", "modell"), sentence)
    return _SENTENCE.sub(fix_sentence, text)


def fix_article_a(text: str) -> str:
    """'a' -> 'an' before a vowel sound. Capitalized words are skipped (acronyms, "I")."""
    def fix(match: re.Match[str]) -> str:
        word = match.group(3)
        if (word[0].isupper() or word.lower().startswith(_A_EXCEPTIONS)
                or word.lower() in _A_NEVER_ARTICLE):
            return match.group(0)
        article = "An" if match.group(1) == "A" else "an"
        return article + match.group(2) + word

    def fix_sentence(sentence: re.Match[str]) -> str:
        words = re.findall(r"[\w'’]+", sentence.group().lower())
        other = sum(w in _NOT_EN_WORDS for w in words) + sum(w in _DE_WORDS for w in words)
        if other > sum(w in _EN_WORDS for w in words):
            return sentence.group() # not English: "a" may be a preposition or a verb
        return _A_BEFORE_VOWEL.sub(fix, sentence.group())
    return _SENTENCE.sub(fix_sentence, text)


def join_chunks(texts: Iterable[str]) -> str:
    """Join chunk transcripts with spaces, applying the fixes above."""
    out = ""
    for text in texts:
        text = fix_article_a(fix_model_de(fix_pronoun_i(text.strip())))
        if not text:
            continue
        if out and not _ENDS_SENTENCE.search(out):
            first = _FIRST_WORD.match(text)
            if first and first.group().lower() in LINKING_WORDS and text[0].isupper():
                text = text[0].lower() + text[1:]
        out = f"{out} {text}" if out else text
    return out
