"""Formal ("Sie", "vous", "usted") or informal ("du", "tu", "tú") address in translations.

NLLB-200 has no switch for this and mostly writes the formal form for English "you" (NLLB-1.3B:
"Sie" in 8 of 11 en->de sentences). A cue in front of the source sentence steers it: "Hey buddy, "
makes it write the informal form. The model then also translates the cue ("Hey Kumpel,", "Hé, mec,",
"Oye amigo,", "Amigo,"), which is cut off the output again. If a cue cannot be cut off safely, the
sentence is translated again without it. Light module (no model): the picker uses it too.
"""
from __future__ import annotations

import re
import unicodedata

INFORMAL, FORMAL = "informal", "formal"
DEFAULT = INFORMAL

# Target languages that have a T-V distinction in the picker (a subset of what NLLB can do).
TV_TARGETS = frozenset({"deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "nld_Latn",
                        "pol_Latn"})

CUE_INFORMAL = "Hey buddy, "
CUE_FORMAL = "Dear Sir, " # measured against "Dear Sir or Madam, " and plain NLLB, see docs/translation.md


def has_formality(target: str) -> bool:
    return target in TV_TARGETS


def cue(formality: str) -> str:
    return CUE_INFORMAL if formality == INFORMAL else CUE_FORMAL


def cue_for(formality: str, sentence: str) -> str:
    """The cue for this sentence: none if it starts with a greeting itself."""
    return "" if source_has_greeting(sentence) else cue(formality)


def strip_cue(formality: str, source: str, text: str) -> str | None:
    """See strip_informal_cue; the formal cue is cut by strip_formal_cue."""
    return strip_informal_cue(source, text) if formality == INFORMAL else strip_formal_cue(source, text)


def _fold(word: str) -> str:
    return unicodedata.normalize("NFKD", word.casefold()).encode("ascii", "ignore").decode()


# What the model makes of "Hey buddy, " in the target languages (greetings, then a name for the buddy).
_GREETINGS = {"hey", "hej", "he", "hé", "ey", "hi", "hallo", "hello", "hola", "oye", "ehi", "ei", "ola",
              "olá", "salut", "hoi", "cześć", "czesc", "ciao", "bonjour", "buenas", "hallöchen"}
_BUDDIES = {"kumpel", "kumpels", "freund", "buddy", "mate", "bro", "mec", "pote", "copain", "amigo",
            "amigos", "amico", "amici", "colega", "compadre", "maatje", "maat", "vriend", "vriendje",
            "kolego", "kolega", "stary", "przyjacielu", "brother", "dude", "kumpelchen", "digger"}
_POSSESSIVE = {"mon", "meu", "mi", "mio"} # "mon pote", "meu amigo": counts only before a buddy word
# Unambiguous cue words: a leak anywhere in the output counts (not just at its start).
_STRICT = re.compile(r"\b(?:kumpels?|mec|pote|amigos?|amico|maatje|kolego|buddy)\b", re.I)
_SOURCE_HAS_FRIEND = ("friend", "buddy", "mate", "pal", "freund", "kumpel", "ami", "amigo", "amico",
                      "vriend", "przyjaci", "kolega", "mec", "copain")
# A source that starts like this already has a greeting: no cue, nothing is cut off.
_SOURCE_GREETING = _GREETINGS | _BUDDIES | {"dear", "liebe", "lieber", "sehr", "guten", "good", "moin",
                                            "servus", "gruss", "gruß", "grüß", "hello", "yo", "dude",
                                            "chers", "cher", "chère", "estimado", "estimada", "caro",
                                            "mon", "meu", "mi"}
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.U)
_DELIM = re.compile(r"\s*([,!:.])\s*|\s+")


def _head_words(text: str) -> list[str]:
    """The first words, lowercase and without accents ("Hé" -> "he")."""
    return [_fold(m.group()) for m in _WORD.finditer(text[:60])][:4]


def source_has_greeting(source: str) -> bool:
    words = _head_words(source)
    return bool(words) and words[0] in {_fold(w) for w in _SOURCE_GREETING}


def _recapitalise(rest: str, source: str) -> str:
    """The cue was the start of the sentence: the next word takes its capital (also after "¿")."""
    if not source[:1].isupper():
        return rest
    for i, ch in enumerate(rest[:3]):
        if ch.isalpha():
            return rest[:i] + ch.upper() + rest[i + 1:]
    return rest


_GREETING_FOLDED = {_fold(w) for w in _GREETINGS}
_BUDDY_FOLDED = {_fold(w) for w in _BUDDIES}


def strip_informal_cue(source: str, text: str) -> str | None:
    """The translation without the translated cue; text unchanged if the model dropped the cue
    ("Tu peux ..."); None if a cue word is left that cannot be cut off safely (the caller
    translates without the cue then). Only the start of the sentence is cut, at most 3 words
    (greeting and buddy words, ended by "," "!" ":" or "."); the rest must not be empty."""
    if source_has_greeting(source):
        return text
    pos, taken, ended, first = 0, 0, False, ""
    while taken < 3:
        m = _WORD.match(text, pos)
        if m is None:
            break
        word = _fold(m.group())
        after = _WORD.match(text, m.end() + 1) if text[m.end():m.end() + 1] == " " else None
        possessive = word in _POSSESSIVE and after is not None and _fold(after.group()) in _BUDDY_FOLDED
        if word not in _GREETING_FOLDED and word not in _BUDDY_FOLDED and not possessive:
            break
        taken += 1
        first = first or word
        pos = m.end()
        d = _DELIM.match(text, pos)
        ended = bool(d and d.group(1))
        if d is not None:
            pos = d.end()
    if taken:
        rest = text[pos:]
        if not ended:
            # "Hey Kumpel kannst du": a leaked cue without a comma; "Freund hat ..." may be real
            return None if first in _GREETING_FOLDED or _STRICT.match(text) else text
        if not rest.strip() or not _WORD.match(rest.lstrip("¿¡ \"'„“«")):
            return None
        return _recapitalise(rest, source)
    if _STRICT.search(text) and not any(f in source.lower() for f in _SOURCE_HAS_FRIEND):
        return None
    return text


# What the model makes of "Dear Sir, ": "Lieber Herr,", "Lieber Sir,", "Sehr geehrter Herr,",
# "Cher monsieur,", "Chers messieurs,", "Querido señor,", also with "oder / ou / o" and a name.
_SALUTE_START = {"sehr", "geehrter", "geehrte", "lieber", "liebe", "cher", "chers", "chere", "cheres",
                 "querido", "querida", "estimado", "estimada", "estimados", "caro", "cara", "egregio",
                 "gentile", "gentilissimo", "prezado", "prezada", "caros", "beste", "geachte", "waarde",
                 "lieve", "szanowny", "szanowna", "szanowni", "drogi", "droga", "kochany", "amado", "ilustre"}
_SALUTE_REST = {"herr", "herrn", "frau", "sir", "mr", "m", "mme", "monsieur", "messieurs", "madame",
                "mesdames", "senor", "senora", "senores", "sr", "sra", "signore", "signor", "signora",
                "senhor", "senhora", "meneer", "mevrouw", "pan", "pani", "panie", "pani", "oder", "ou",
                "o", "y", "e", "und", "of", "i", "damen", "herren", "madam", "miller", "geehrter",
                "geehrte"}


def strip_formal_cue(source: str, text: str) -> str | None:
    """Like strip_informal_cue for the salutation the formal cue turns into. A salutation that
    cannot be cut off safely (no comma, or too long) gives None."""
    if source_has_greeting(source):
        return text
    words = list(_WORD.finditer(text[:80]))
    if not words or _fold(words[0].group()) not in _SALUTE_START:
        return text
    end = None
    for k, m in enumerate(words[:7]):
        word = _fold(m.group())
        if k and word not in _SALUTE_REST and word not in _SALUTE_START:
            break
        nxt = text[m.end():m.end() + 1] if m.end() < len(text) else ""
        if nxt == "." and word in ("mr", "sr", "sra", "m", "mme"): # "Mr. Miller"
            continue
        if nxt in (",", "!", ":"):
            end = m.end() + 1
            break
    if end is None:
        return None
    rest = text[end:].lstrip()
    if not rest or not _WORD.match(rest.lstrip("¿¡ \"'„“«")):
        return None
    return _recapitalise(rest, source)
