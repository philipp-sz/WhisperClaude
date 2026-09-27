"""Real model on bundled clips (Windows text-to-speech, no personal voice). Opt-in: -m slow."""
import wave
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


def load(name):
    with wave.open(str(DATA / name), "rb") as w:
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
@pytest.mark.parametrize("clip, lang, words", [
    ("speech_de.wav", "de", ["morgen", "brötchen", "käse"]),
    ("speech_en.wav", "en", ["morning", "test", "dictation"]),
])
def test_transcribes_bundled_clip(transcriber, clip, lang, words):
    text, detected = transcriber.transcribe_detailed(load(clip))
    assert detected == lang
    for word in words:
        assert word in text.lower(), text


@pytest.mark.slow
def test_silence_gives_empty_text(transcriber):
    assert transcriber.transcribe(np.zeros(3 * 16_000, np.float32)) == ""
