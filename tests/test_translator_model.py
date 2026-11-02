"""The real translation model (downloads ~0.6 GB on first run): pytest -m slow."""
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def translator():
    from whisperclaude.translator import Translator
    return Translator("nllb-600m", device="cpu")


def test_translates_and_keeps_layout(translator):
    text = ("Hallo zusammen, das Meeting morgen fällt leider aus.\n\n"
            "- Bitte schickt mir eure Folien.\n"
            "- Wir verschieben die Demo auf Freitag.")
    out = translator.translate(text, "eng_Latn")
    lines = out.split("\n")
    assert len(lines) == 4 and lines[1] == ""
    assert lines[2].startswith("- ") and lines[3].startswith("- ")
    assert "meeting" in out.lower() and "friday" in out.lower()


def test_mixed_text_and_target_language_sentences(translator):
    out = translator.translate("This sentence is already in English, nothing to do here. "
                               "Aber dieser Satz hier ist auf Deutsch geschrieben.", "eng_Latn")
    assert out.startswith("This sentence is already in English, nothing to do here. ")
    assert "german" in out.lower() and "Satz" not in out


def test_other_scripts(translator):
    assert any("぀" <= c <= "ヿ" or "一" <= c <= "鿿"
               for c in translator.translate("Guten Morgen! Wie geht es dir?", "jpn_Jpan"))
    assert "Bonjour" in translator.translate("Good morning!", "fra_Latn")


def test_unknown_target(translator):
    with pytest.raises(ValueError):
        translator.translate("Hallo", "xxx_Latn")


def test_picked_source_fixes_short_texts(translator):
    """With only English as the user's language, "Kein Problem" is detected as English and
    NLLB wrote "Le problème de Kein". Picking German as the source fixes it."""
    preferred, translator.detector.preferred = translator.detector.preferred, ["en"]
    try:
        assert "kein" in translator.translate("Kein Problem", "fra_Latn").lower()
        fixed = translator.translate("Kein Problem", "fra_Latn", source="deu_Latn")
        assert fixed.lower().startswith("pas de problème")
    finally:
        translator.detector.preferred = preferred
