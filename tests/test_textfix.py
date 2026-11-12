import pytest

from whisperclaude.textfix import fix_article_a, fix_model_de, fix_pronoun_i, join_chunks


@pytest.mark.parametrize("chunks, expected", [
    # mid-sentence cut: linking word lowercased (cases seen in real chunk borders)
    (["in verschiedenen Sprachen aufnehmen", "Und wir testen."],
     "in verschiedenen Sprachen aufnehmen und wir testen."),
    (["Sprachen aufnehmen,", "Und wir testen."], "Sprachen aufnehmen, und wir testen."),
    (["we tested it", "And it works."], "we tested it and it works."),
    # sentence ended: capital stays
    (["zunächst Englisch und jetzt Deutsch.", "Um zu sehen, wie."],
     "zunächst Englisch und jetzt Deutsch. Um zu sehen, wie."),
    (["Is it done?", "And then?"], "Is it done? And then?"),
    # not a linking word: nouns and "I" keep their capital
    (["wir haben die", "Aufnahme gestartet."], "wir haben die Aufnahme gestartet."),
    (["the problem is that", "Every time I pause"], "the problem is that Every time I pause"),
    (["and then", "I paused."], "and then I paused."),
    # a spoken command ends the line: the next chunk keeps its capital
    (["Erster Punkt command new paragraph", "Und weiter."],
     "Erster Punkt command new paragraph Und weiter."),
    # empty chunks are skipped
    (["", "Hallo.", " ", "Und tschüss."], "Hallo. Und tschüss."),
    ([], ""),
])
def test_join_chunks(chunks, expected):
    assert join_chunks(chunks) == expected


@pytest.mark.parametrize("text, expected", [
    ("because i had something", "because I had something"),
    ("i think i'm right, i've seen it and i'll say i'd do it", "I think I'm right, I've seen it and I'll say I'd do it"),
    ("so i, too", "so I, too"),
    # untouched
    ("i.e. this", "i.e. this"),
    ("das i-Tüpfelchen", "das i-Tüpfelchen"),
    ("my iPhone and iGPU", "my iPhone and iGPU"),
    ("Wiki in Hi", "Wiki in Hi"),
])
def test_fix_pronoun_i(text, expected):
    assert fix_pronoun_i(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Ein Speech to Text Model. Aber", "Ein Speech to Text Modell. Aber"),
    ("mehr als ein Chatmodel für einfache Fragen sind sie", "mehr als ein Chatmodell für einfache Fragen sind sie"),
    ("Das Speech-to-Text-Model, das wir nehmen", "Das Speech-to-Text-Modell, das wir nehmen"),
    # English sentences keep "model"
    ("the model is fast and it works.", "the model is fast and it works."),
    ("I use a Model for that. It is good.", "I use a Model for that. It is good."),
    ("Chatmodels are cool", "Chatmodels are cool"),
    ("Mein Model ist gut", "Mein Modell ist gut"),
])
def test_fix_model_de(text, expected):
    assert fix_model_de(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("a app that", "an app that"),
    ("A idea", "An idea"),
    ("a hour later", "an hour later"),
    ("an honest try", "an honest try"),
    # consonant sound or capitalized: untouched
    ("a user", "a user"),
    ("a one-time fee", "a one-time fee"),
    ("a European bank", "a European bank"),
    ("a Windows machine", "a Windows machine"),
    ("a couple of times", "a couple of times"),
    ("an alternative", "an alternative"),
    ("a I x", "a I x"),
])
def test_fix_article_a(text, expected):
    assert fix_article_a(text) == expected


@pytest.mark.parametrize("text", [
    "voy a escribir un correo",      # Spanish preposition
    "Il a entendu le bruit",         # French verb
    "Vado a ordinare per la cena",   # Italian preposition
    "Plan a or plan b",              # a label, not an article
    "vitamin a and b",
])
def test_fix_article_a_leaves_other_languages_and_labels(text):
    assert fix_article_a(text) == text
    assert join_chunks([text]) == text
