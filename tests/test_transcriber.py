"""Real models on your own short clips. Opt-in: pytest -m slow.

The clips aren't in the repo (no voices are committed). Record them once:
  .venv\\Scripts\\python.exe scripts\\record_clip.py speech_de --dir tests/data --say "<SENTENCES de>"
  .venv\\Scripts\\python.exe scripts\\record_clip.py speech_en --dir tests/data --say "<SENTENCES en>"
"""
import wave
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
SENTENCES = {
    "de": "Guten Morgen. Die Brötchen und der Käse sind sehr lecker.",
    "en": "Good morning. This is a short test of the dictation app.",
}
KEYWORDS = {"de": ["morgen", "brötchen", "käse"], "en": ["morning", "test", "dictation"]}


def load(lang):
    path = DATA / f"speech_{lang}.wav"
    if not path.exists():
        pytest.skip(f"{path.name} missing; record it: scripts\\record_clip.py speech_{lang} "
                    f'--dir tests/data --say "{SENTENCES[lang]}"')
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16_000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def transcriber(request):
    from whisperclaude import transcriber as t
    if request.param == "cpu":
        return t.Transcriber()
    if not t.gpu_available():
        pytest.skip("no OpenVINO GPU")
    return t.OpenVinoTranscriber()


@pytest.mark.slow
@pytest.mark.parametrize("lang", ["de", "en"])
def test_transcribes_recorded_clip(transcriber, lang):
    text, detected = transcriber.transcribe_detailed(load(lang))
    assert detected == lang
    for word in KEYWORDS[lang]:
        assert word in text.lower(), text


@pytest.mark.slow
def test_silence_gives_empty_text(transcriber):
    assert transcriber.transcribe(np.zeros(3 * 16_000, np.float32)) == ""
