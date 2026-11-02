"""Translation helpers (no model): sentence splitting, layout, source languages, search."""
import pytest

from whisperclaude.languages import BY_CODE, COMMON, LANGUAGES, NLLB_CODES, SOURCE_CODES, search
from whisperclaude.translation import RecentLanguages
from whisperclaude.translator import segment, source_languages, split_sentences, tidy


@pytest.mark.parametrize("text, expected", [
    ("Hallo Welt. Wie geht's? Gut!", ["Hallo Welt.", "Wie geht's?", "Gut!"]),
    ("Das ist z. B. so. Und e.g. this one.", ["Das ist z. B. so.", "Und e.g. this one."]),
    ('Er sagte: „Ja.“ Dann ging er.', ['Er sagte: „Ja.“', "Dann ging er."]),
    ("Version 2.5 is out. 3 people came.", ["Version 2.5 is out.", "3 people came."]),
    ("你好。今天很好！", ["你好。", "今天很好！"]),
    ("no end", ["no end"]),
    ("", []),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_segment_keeps_layout():
    rows = segment("Intro. Second.\n\n- Item one\n  2. Item two\n> quoted")
    assert rows == [("", ["Intro.", "Second."], ""), ("", [], ""), ("- ", ["Item one"], ""),
                    ("  2. ", ["Item two"], ""), ("> ", ["quoted"], "")]


def test_source_languages_short_sentences_follow_the_text():
    def detect(s):
        if "Englisch" in s or len(s) > 60:
            return "deu_Latn", 0.99
        return ("eng_Latn", 0.95) if "English" in s else ("fra_Latn", 0.5)

    sentences = ["Ok.", "Das ist ein langer Satz auf Deutsch, nicht Englisch.",
                 "This sentence is clearly English."]
    assert source_languages(sentences, detect) == ["deu_Latn", "deu_Latn", "eng_Latn"]
    assert source_languages([], detect) == []


def test_unsure_detection_falls_back_to_the_text():
    detect = lambda s: ("deu_Latn", 0.99) if len(s) > 80 else ("ita_Latn", 0.6)
    long = "Ein Satz, der lang genug für die Erkennung ist, aber unsicher."
    assert source_languages([long, long], detect) == ["deu_Latn", "deu_Latn"]


def test_language_tables():
    assert len(LANGUAGES) > 190 and len({l.code for l in LANGUAGES}) == len(LANGUAGES)
    assert set(COMMON) <= NLLB_CODES and set(SOURCE_CODES.values()) <= NLLB_CODES
    assert BY_CODE["deu_Latn"].native == "Deutsch"
    assert SOURCE_CODES["de"] == "deu_Latn" and SOURCE_CODES["fa"] == "pes_Arab"


@pytest.mark.parametrize("query, first", [
    ("", "fra_Latn"), ("de", "deu_Latn"), ("deutsch", "deu_Latn"), ("GERM", "deu_Latn"),
    ("francais", "fra_Latn"), ("zh", "zho_Hans"), ("traditional", "zho_Hant"),
    ("nor", "nob_Latn"), ("persian", "pes_Arab"),
])
def test_search(query, first):
    assert search(LANGUAGES, query, ["fra_Latn"])[0].code == first


def test_search_empty_query_lists_everything():
    result = search(LANGUAGES, "", ["jpn_Jpan"])
    assert [l.code for l in result[:3]] == ["jpn_Jpan", "eng_Latn", "deu_Latn"]
    assert len(result) == len(LANGUAGES)
    assert search(LANGUAGES, "xyzzy") == []


def test_recent_languages(tmp_path):
    path = tmp_path / "translate.json"
    recent = RecentLanguages(path)
    assert recent.codes == []
    for code in ("eng_Latn", "fra_Latn", "eng_Latn", "bogus", "deu_Latn"):
        recent.add(code)
    assert recent.codes == ["deu_Latn", "bogus", "eng_Latn", "fra_Latn"]
    reloaded = RecentLanguages(path)
    assert reloaded.codes == ["deu_Latn", "eng_Latn", "fra_Latn"] # unknown codes dropped
    path.write_text("not json", encoding="utf-8")
    assert RecentLanguages(path).codes == []


@pytest.mark.parametrize("source, text, target, expected", [
    ("ok", "Okay , that 's all right .", "eng_Latn", "Okay, that's all right."),
    ("Ja", "- Yes , I did .", "eng_Latn", "Yes, I did."),
    ("Nein", "I do n't know .", "eng_Latn", "I don't know."),
    ("Danke", "Merci !", "fra_Latn", "Merci !"), # French typography stays
    ("Danke", "Gracias , amigo !", "spa_Latn", "Gracias, amigo!"),
    ("- Punkt eins", "- Point one", "eng_Latn", "- Point one"),
    ("Pi ist 3,14.", "Pi is 3 .14.", "eng_Latn", "Pi is 3 .14."), # decimals untouched
    ("Ein normaler Satz.", "A normal sentence, isn't it?", "eng_Latn",
     "A normal sentence, isn't it?"),
])
def test_tidy(source, text, target, expected):
    assert tidy(source, text, target) == expected


def test_detect_text_prefers_the_users_languages():
    from whisperclaude.translator import LanguageDetector
    d = LanguageDetector(preferred=["de", "en"])
    assert d.detect_text("ok") == "deu_Latn" # no signal: the first preferred
    assert d.detect_text("Meeting") == "eng_Latn"
    assert d.detect_text("Merci beaucoup") == "fra_Latn" # clear enough on its own
    assert d.detect_text("Ich komme später nach Hause.") == "deu_Latn"


def test_translation_waits_for_the_model(monkeypatch):
    import threading

    from whisperclaude import translation as tr

    class Fake:
        def translate(self, text, target, source=None):
            return f"{target}:{text}"

    t = tr.Translation.__new__(tr.Translation) # no picker / Tk needed for this part
    t._translator, t._error, t._loaded = None, "", threading.Event()
    monkeypatch.setattr(tr, "LOAD_TIMEOUT_S", 0.05)
    with pytest.raises(RuntimeError, match="still loading"):
        t.translate("x", "eng_Latn")
    t.set_failed("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        t.translate("x", "eng_Latn")
    t._translator = Fake()
    assert t.translate("x", "eng_Latn") == "eng_Latn:x"


def test_own_languages_from_the_config_beat_windows(monkeypatch):
    """A German speaker with English Windows and a US keyboard: Windows only knows English."""
    import whisperclaude.translator as translator

    monkeypatch.setattr(translator, "user_languages", lambda: ["en"])
    assert translator.LanguageDetector().preferred == ["en"] # nothing configured: Windows
    assert translator.LanguageDetector().detect_text("Bis morgen") == "eng_Latn"
    configured = translator.LanguageDetector(["de", "en"])
    assert configured.detect_text("Bis morgen") == "deu_Latn"
    assert configured.detect_text("See you tomorrow") == "eng_Latn"


def test_picked_source_is_used_except_for_sentences_already_in_the_target():
    from whisperclaude.translator import LanguageDetector, Translator

    t = Translator.__new__(Translator) # no model needed for choosing the source languages
    t.detector = LanguageDetector(["en"])
    sentences = ["Danke", "Kein Problem", "This sentence is clearly written in English."]
    assert t._sources(sentences, None, "eng_Latn")[:2] == ["eng_Latn", "eng_Latn"] # wrong
    assert t._sources(sentences, "deu_Latn", "eng_Latn") == ["deu_Latn", "deu_Latn", "eng_Latn"]
    text = ("Das ist ein ganz normaler deutscher Satz. Noch ein deutscher Satz mit Inhalt. "
            "This one is an English sentence for sure.")
    assert t.guess(text) == ["deu_Latn", "eng_Latn"] # most sentences first


def test_recent_sources_and_own_languages(tmp_path):
    from whisperclaude import translation as tr

    recent = RecentLanguages(tmp_path / "translate.json")
    recent.add("fra_Latn")
    recent.add_source("deu_Latn")
    again = RecentLanguages(tmp_path / "translate.json")
    assert again.codes == ["fra_Latn"] and again.sources == ["deu_Latn"]
    t = tr.Translation.__new__(tr.Translation)
    t._translator, t.my_languages = None, ("es", "en")
    assert t.own_sources() == ["spa_Latn", "eng_Latn"] # from the config until loaded
    assert t.guess("Hola") == [] # nothing detected while the model loads
