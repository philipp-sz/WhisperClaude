"""The prompt sentence only for its own languages (PromptByLanguage), without models."""
from types import SimpleNamespace

import numpy as np

from whisperclaude import transcriber as tr

PROMPT = "Hallo, das ist ein Test. Okay, so let's see."
CHUNK = np.zeros(5 * tr.SAMPLE_RATE, np.float32)


class FakePipeline:
    """Whisper hearing `spoken`: with the German+English prompt it answers in English."""

    def __init__(self, spoken):
        self.spoken, self.calls = spoken, []

    def get_generation_config(self):
        return SimpleNamespace()

    def generate(self, chunk, cfg):
        forced = getattr(cfg, "language", None)
        self.calls.append((cfg.initial_prompt, forced))
        translated = cfg.initial_prompt and cfg.initial_prompt.startswith("Hallo")
        text = "English text." if translated and self.spoken == "fr" else f"{self.spoken} text."
        return SimpleNamespace(texts=[text], language=forced or f"<|{self.spoken}|>")


def engine(spoken, language=None):
    e = tr.OpenVinoTranscriber.__new__(tr.OpenVinoTranscriber)
    e.pipeline, e.language = FakePipeline(spoken), language
    e._set_prompts(PROMPT, ["Claude"], ("de", "en"))
    return e


def test_prompt_languages_are_decoded_once_with_the_prompt():
    """Test that German costs no second decode and keeps the full prompt."""
    e = engine("de")
    assert e._generate([CHUNK]) == ("de text.", "de")
    assert e.pipeline.calls == [(f"{PROMPT} Claude.", None)]


def test_other_language_is_decoded_again_without_the_sentence():
    """Test that a French chunk is decoded again in French with only the vocabulary as prompt."""
    e = engine("fr")
    assert e._generate([CHUNK]) == ("fr text.", "fr")
    assert e.pipeline.calls == [(f"{PROMPT} Claude.", None), ("Claude.", "<|fr|>")]


def test_fixed_other_language_skips_the_sentence_at_once():
    """Test that a configured language outside prompt_languages never gets the sentence."""
    e = engine("fr", language="fr")
    assert e._generate([CHUNK])[0] == "fr text."
    assert e.pipeline.calls == [("Claude.", "<|fr|>")]


def test_next_chunk_starts_with_the_last_language_prompt():
    """Test that French after French costs one decode, and German after French gets the
    sentence back."""
    e = engine("fr")
    e._generate([CHUNK])
    e.pipeline.calls.clear()
    assert e._generate([CHUNK])[0] == "fr text."
    assert e.pipeline.calls == [("Claude.", None)]
    e.pipeline.spoken = "de"
    e.pipeline.calls.clear()
    assert e._generate([CHUNK])[0] == "de text."
    assert e.pipeline.calls == [("Claude.", None), (f"{PROMPT} Claude.", None)]


def test_unknown_language_keeps_the_prompt_sentence():
    """Test that an empty detected language counts as unknown: full prompt, no second decode."""
    e = engine("")
    e._generate([CHUNK])
    assert e.pipeline.calls == [(f"{PROMPT} Claude.", None)]


SHORT = np.zeros(int((tr.LEAD_IN_S + 2) * tr.SAMPLE_RATE), np.float32)


def test_short_chunk_in_another_language_keeps_the_sentence():
    """Test that a 2 s chunk heard as Russian after German is not decoded again: short German
    was often heard as Russian or Korean."""
    e = engine("de")
    e._generate([CHUNK])
    e.pipeline.spoken = "ru"
    e.pipeline.calls.clear()
    assert e._generate([SHORT])[1] == "ru"
    assert e.pipeline.calls == [(f"{PROMPT} Claude.", None)]
    assert e._heard is None # not trusted: the next chunk starts with the sentence


def test_short_chunk_in_the_previous_language_drops_the_sentence():
    """Test that a 2 s French chunk after a French chunk is trusted: no sentence, no redo."""
    e = engine("fr")
    e._generate([CHUNK])
    e.pipeline.calls.clear()
    assert e._generate([SHORT])[0] == "fr text."
    assert e.pipeline.calls == [("Claude.", None)]


def test_every_generate_call_stays_within_30_s_on_a_full_chunk_with_a_redo():
    """Test that the language redo and the detection pass never get audio over 30 s."""
    lengths = []
    e = engine("fr")
    inner = e.pipeline.generate
    e.pipeline.generate = lambda a, cfg: lengths.append(len(a)) or (
        inner(a, cfg) if hasattr(cfg, "initial_prompt") else SimpleNamespace(language="<|fr|>"))
    speech = np.full(int(29.8 * tr.SAMPLE_RATE), 1, np.float32)
    chunk = tr._join(speech, [{"start": 0, "end": len(speech)}])
    e._generate([chunk])
    e.detect_language(np.concatenate([speech, speech]))
    assert len(lengths) == 3 and max(lengths) <= tr.CHUNK_S * tr.SAMPLE_RATE
