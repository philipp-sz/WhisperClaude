"""Formal / informal address: cue cutting, which targets have it, the picker toggle, persistence."""
import contextlib
import tkinter as tk

import pytest

import whisperclaude.translator as tl
from whisperclaude import formality as fm
from whisperclaude.languages import LANGUAGES
from whisperclaude.translation import RecentLanguages


@pytest.mark.parametrize("text, expected", [
    ("Hey Kumpel, kannst du mir helfen?", "Kannst du mir helfen?"),
    ("Hey, Kumpel, willst du was trinken?", "Willst du was trinken?"),
    ("Hé, mon pote, j'ai mis le fichier.", "J'ai mis le fichier."),
    ("Salut, mon pote, comment vas-tu ?", "Comment vas-tu ?"),
    ("Oye amigo, ¿puedes venir?", "¿Puedes venir?"),
    ("Amigo, ¿podrías enviarme el archivo?", "¿Podrías enviarme el archivo?"),
    ("Ehi, amico, hai un minuto?", "Hai un minuto?"),
    ("Hé maatje, kan je helpen?", "Kan je helpen?"),
    ("Hej, kolego, możesz pomóc?", "Możesz pomóc?"),
    ("Tu peux m'aider ?", "Tu peux m'aider ?"), # the model dropped the cue: unchanged
    ("Mi hermano llegó tarde.", "Mi hermano llegó tarde."), # not a cue
    ("Das ist ein Mann, der schläft.", "Das ist ein Mann, der schläft."),
])
def test_informal_cue_is_cut_only_at_the_start(text, expected):
    assert fm.strip_informal_cue("Can you help me?", text) == expected


@pytest.mark.parametrize("text", [
    "Hey Kumpel kannst du mir helfen?", # no comma: not safe to cut
    "Hey Kumpel,", # nothing left
    "Ich nenne dich Kumpel.", # a cue word inside the sentence
    "Hey, Amigo",
])
def test_informal_cue_leaks_are_reported_not_guessed(text):
    assert fm.strip_informal_cue("Can you help me?", text) is None


def test_a_source_greeting_is_kept_and_gets_no_cue():
    assert fm.cue_for(fm.INFORMAL, "Hey, can you help?") == ""
    assert fm.strip_informal_cue("Hey, can you help?", "Hey, kannst du helfen?") == "Hey, kannst du helfen?"
    assert fm.cue_for(fm.INFORMAL, "Dear Anna, thanks.") == ""
    assert fm.cue_for(fm.INFORMAL, "Can you help?") == fm.CUE_INFORMAL
    # a friend in the source may come back as "amigo": not a leak
    assert fm.strip_informal_cue("Tell my friend hello.", "Saluda a mi amigo.") == "Saluda a mi amigo."


@pytest.mark.parametrize("text, expected", [
    ("Lieber Herr oder Frau, können Sie helfen?", "Können Sie helfen?"),
    ("Sehr geehrter Herr, vielen Dank.", "Vielen Dank."),
    ("Chers Monsieur ou Madame, comment allez-vous ?", "Comment allez-vous ?"),
    ("Cher monsieur, pouvez-vous venir ?", "Pouvez-vous venir ?"),
    ("Querido Sr. Miller, ¿puede venir?", "¿Puede venir?"),
    ("Lieve meneer, kunt u mij helpen?", "Kunt u mij helpen?"),
    ("Können Sie mir helfen?", "Können Sie mir helfen?"),
])
def test_formal_cue_is_cut(text, expected):
    assert fm.strip_formal_cue("Can you help?", text) == expected


def test_formal_cue_without_comma_is_a_leak():
    assert fm.strip_formal_cue("Can you help?", "Lieber Herr können Sie") is None


def test_only_languages_with_a_t_v_distinction_get_a_cue():
    codes = {lang.code for lang in LANGUAGES}
    assert fm.TV_TARGETS <= codes
    for code in ("deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "nld_Latn", "pol_Latn"):
        assert fm.has_formality(code)
    for code in ("eng_Latn", "jpn_Jpan", "zho_Hans"):
        assert not fm.has_formality(code)


class FakeModel:
    """translate_batch echoes the sentence (cue included); a "translation" is its tokens."""
    def __init__(self):
        self.batches = []

    def translate_batch(self, batch, **kw):
        self.batches.append(batch)

        class R:
            def __init__(self, text):
                self.hypotheses, self.scores = [text], [0.0]
        return [R(" ".join(t[1:-1])) for t in batch]


def make_translator(fake):
    t = tl.Translator.__new__(tl.Translator)
    t.beam_size = 2
    t._tokens = lambda sentence, source: [source] + sentence.split() + ["</s>"]
    t._decode = lambda text: text
    t._sources = lambda sentences, source, target: [source or "eng_Latn"] * len(sentences)

    class Lazy:
        @contextlib.contextmanager
        def use(self):
            yield fake
    t._lazy = Lazy()
    return t


def sent(batch, i=0):
    return " ".join(batch[i][1:-1])


def test_translate_uses_the_cue_per_target_and_strips_it():
    fake = FakeModel()
    t = make_translator(fake)
    out = t.translate("Can you help me? Thanks a lot.", "deu_Latn", "eng_Latn", "informal")
    assert sent(fake.batches[0]).startswith("Hey buddy,") and len(fake.batches[0]) == 2
    assert out == "Can you help me? Thanks a lot." # the echoed cue "Hey buddy, " was cut off again
    fake.batches.clear()
    t.translate("Can you help me?", "jpn_Jpan", "eng_Latn", "informal") # no distinction: no cue
    assert sent(fake.batches[0]) == "Can you help me?"
    fake.batches.clear()
    t.translate("Can you help me?", "fra_Latn", "eng_Latn", "formal")
    assert sent(fake.batches[0]).startswith("Dear Sir,")


def test_a_leaked_cue_is_translated_again_without_it(monkeypatch):
    fake = FakeModel()
    t = make_translator(fake)
    monkeypatch.setattr(fm, "strip_cue", lambda mode, source, text: None if "buddy" in text else text)
    out = t.translate("Can you help me?", "deu_Latn", "eng_Latn", "informal")
    assert out == "Can you help me?" and len(fake.batches) == 2
    assert sent(fake.batches[1]) == "Can you help me?" # second pass without the cue


def test_formality_is_remembered_like_the_recent_languages(tmp_path):
    path = tmp_path / "translate.json"
    recent = RecentLanguages(path)
    assert recent.formality == "informal" # default
    recent.set_formality("formal")
    recent.add("deu_Latn")
    assert RecentLanguages(path).formality == "formal"
    recent.set_formality("bogus")
    assert RecentLanguages(path).formality == "formal"
    path.write_text('{"formality": "bogus"}', encoding="utf-8")
    assert RecentLanguages(path).formality == "informal"


def test_picker_toggle_follows_the_target_language_and_the_stored_choice():
    from whisperclaude.picker import LanguagePicker

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    chosen = ["informal"]
    picker = LanguagePicker(root, LANGUAGES, formality=lambda: chosen[0],
                            set_formality=lambda m: chosen.__setitem__(0, m))
    try:
        picker._target = "deu_Latn"
        picker._show_state()
        assert picker.formality_enabled
        picker._pick_formality("formal") # click on "Formal"
        assert chosen == ["formal"]
        picker._toggle_formality() # Ctrl+F
        assert chosen == ["informal"]
        picker._target = "jpn_Jpan" # no T-V distinction: inert
        picker._show_state()
        assert not picker.formality_enabled
        picker._pick_formality("formal")
        assert chosen == ["informal"]
    finally:
        root.destroy()
