"""Language guard: a chunk heard as German but written in English is decoded again without the
prompt sentence (OpenVinoTranscriber and CudaTranscriber), without models."""
from types import SimpleNamespace

import numpy as np

from whisperclaude import transcriber as tr

PROMPT = "Hallo, das ist ein Test. Okay, so let's see."
CHUNK = np.zeros(5 * tr.SAMPLE_RATE, np.float32)
DE = "Das ist ein ganz normaler deutscher Satz über das Wetter."
EN = "This is a perfectly normal English sentence about the weather."


class FakePipeline:
    """Hears `heard`; writes `with_sentence` if the prompt starts with the sentence, else `without`."""

    def __init__(self, heard, with_sentence, without):
        self.heard, self.texts, self.calls = heard, (with_sentence, without), []

    def get_generation_config(self):
        return SimpleNamespace()

    def generate(self, chunk, cfg):
        forced = getattr(cfg, "language", None)
        self.calls.append((cfg.initial_prompt, forced))
        text = self.texts[0 if cfg.initial_prompt.startswith("Hallo") else 1]
        return SimpleNamespace(texts=[text], language=forced or f"<|{self.heard}|>")


def engine(heard, with_sentence, without=DE, language=None):
    e = tr.OpenVinoTranscriber.__new__(tr.OpenVinoTranscriber)
    e.pipeline, e.language = FakePipeline(heard, with_sentence, without), language
    e._set_prompts(PROMPT, ["Claude"], ("de", "en"))
    return e


def test_german_written_in_english_is_decoded_again_without_the_sentence():
    e = engine("de", EN)
    assert e._generate([CHUNK])[0] == DE
    (first, _), (second, forced) = e.pipeline.calls
    assert first.startswith("Hallo") and second == "Claude." and forced == "<|de|>"


def test_agreement_is_decoded_once():
    e = engine("de", DE)
    assert e._generate([CHUNK])[0] == DE
    assert len(e.pipeline.calls) == 1


def test_real_language_switch_is_kept():
    e = engine("en", EN)
    assert e._generate([CHUNK])[0] == EN
    assert len(e.pipeline.calls) == 1


def test_short_text_is_not_checked():
    e = engine("de", "This is English.")
    assert e._generate([CHUNK])[0] == "This is English."
    assert len(e.pipeline.calls) == 1


def test_other_language_is_not_checked():
    e = engine("fr", EN)
    e._generate([CHUNK])
    assert all(forced == "<|fr|>" for _, forced in e.pipeline.calls[1:])
    assert len(e.pipeline.calls) == 2 # the usual redo without the sentence, no guard retry


def test_fixed_language_is_not_checked():
    e = engine("de", EN, language="de")
    assert e._generate([CHUNK])[0] == EN
    assert len(e.pipeline.calls) == 1


def test_first_text_is_kept_if_the_retry_is_still_wrong():
    e = engine("de", EN, without="Another English sentence that is also wrong here.")
    assert e._generate([CHUNK])[0] == EN
    assert len(e.pipeline.calls) == 2


def test_text_language():
    assert tr.text_language(DE, ("de", "en")) == "de"
    assert tr.text_language(EN, ("de", "en")) == "en"
    assert tr.text_language(EN, ("de",)) is None # one language: nothing to tell apart
    assert tr.text_language("Das Pull Request Review im Backend.", ("de", "en")) is None # short


class FakeModel:
    def __init__(self, heard, with_sentence, without):
        self.heard, self.texts, self.calls = heard, (with_sentence, without), []

    def transcribe(self, chunk, language, initial_prompt, **kwargs):
        self.calls.append((initial_prompt, language))
        text = self.texts[0 if initial_prompt.startswith("Hallo") else 1]
        return [SimpleNamespace(text=text)], SimpleNamespace(language=language or self.heard)


def test_cuda_path_has_the_same_guard():
    e = tr.CudaTranscriber.__new__(tr.CudaTranscriber)
    e.language, e.beam_size = None, 1
    e.model = FakeModel("de", EN, DE)
    e._set_prompts(PROMPT, ["Claude"], ("de", "en"))
    assert e._generate([CHUNK])[0] == DE
    assert e.model.calls[-1] == ("Claude.", "de")
