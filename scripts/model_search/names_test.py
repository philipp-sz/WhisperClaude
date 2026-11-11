"""Does Granite 2B's keyword list fix names? English FLEURS, no keywords / the sentence's own names / one shared list.

    python names_test.py [n]      (in the PyTorch environment; see README.md)
Names = capitalized words inside a sentence (not the first word, not "I"). Prints WER and wrong names per condition.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import eval_models as em
from scripts.eval_fleurs import read_wav, sentences
from scripts.eval_quality import align, cased, score, words

STOP = {"I", "I'm", "I'll", "I've", "I'd"}


def names_of(reference: str) -> list[str]:
    """The capitalized words inside the sentence, without punctuation."""
    return [w for k, w in enumerate(cased(reference)) if k and w[:1].isupper() and w not in STOP and len(w) >= 3]


def wrong_names(reference: str, hypothesis: str) -> tuple[int, int]:
    """(names wrong, names) in the hypothesis against the reference."""
    ref_words, ref_cased = words(reference), cased(reference)
    flags = {k for k in range(1, len(ref_cased)) if ref_cased[k][:1].isupper() and ref_cased[k] not in STOP and len(ref_cased[k]) >= 3}
    wrong = total = k = 0
    for op, _, _ in align(ref_words, words(hypothesis)):
        if op == "ins":
            continue
        if k in flags:
            total, wrong = total + 1, wrong + (op != "=")
        k += 1
    return wrong, total


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    torch.set_num_threads(4)
    model = em.Granite2B(torch.float32)
    data = [(read_wav(Path(p)), ref) for p, ref in sentences(em.WORK / "fleurs" / "en_us", n)]
    shared = sorted({w for _, ref in data for w in names_of(ref)})
    print("shared list:", shared, flush=True)
    results = {}
    for label, keywords in (("none", lambda ref: []), ("own names", names_of), ("shared list", lambda ref: shared)):
        errors = words_n = wrong = total = 0
        for audio, ref in data:
            model.keywords = keywords(ref)
            text = model(audio)
            s = score(ref, text)
            w, t = wrong_names(ref, text)
            errors, words_n, wrong, total = errors + s["errors"], words_n + s["ref_words"], wrong + w, total + t
        results[label] = {"wer": 100 * errors / words_n, "errors": errors, "names_wrong": wrong, "names": total}
        print(label, json.dumps(results[label]), flush=True)
    (em.WORK / "names_test_granite2b.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
