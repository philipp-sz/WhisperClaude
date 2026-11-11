"""Loop guard, VAD input level and the GPU probe of transcriber.py, without models."""
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from whisperclaude import transcriber as tr

SPEECH = ("Der Bericht ist fertig, aber die Zahlen im zweiten Abschnitt muss ich noch einmal prüfen. Danach schicke "
          "ich ihn an das ganze Team, damit alle bis Freitag Rückmeldung geben können, falls etwas fehlt.")
LOOP = "Das Integral von x² " + "dx " * 368 + "ist gleich ein Drittel."


@pytest.mark.parametrize("text, seconds, looped", [
    (SPEECH, 12.0, False),
    ("Ja, ja, ja, ja.", 1.0, False), # short repeats are speech
    (LOOP, 19.0, True), # repetition compresses far below normal text
    (" ".join(f"wort{i}" for i in range(60)), 5.0, True), # 12 words/s: more than anyone speaks
    ("", 5.0, False),
])
def test_is_loop(text, seconds, looped):
    """Test that loops are told apart from normal and repetitive speech."""
    assert tr.is_loop(text, seconds) is looped


@pytest.mark.parametrize("text, expected", [
    ("x² dx dx dx dx dx dx.", "x² dx."),
    ("ich weiß nicht ich weiß nicht ich weiß nicht ich weiß nicht ob", "ich weiß nicht ob"),
    ("ganz, ganz stark gesunken", "ganz, ganz stark gesunken"), # a spoken double stays
])
def test_collapse_repeats(text, expected):
    """Test that runs of 4+ repeats shrink to one and normal text is untouched."""
    assert tr.collapse_repeats(text) == expected


class FakePipeline:
    """Answers generate() with a fixed text per repetition penalty."""

    def __init__(self, texts):
        self.texts, self.penalties = texts, []

    def get_generation_config(self):
        return SimpleNamespace()

    def generate(self, chunk, cfg):
        self.penalties.append(cfg.repetition_penalty)
        return SimpleNamespace(texts=[self.texts[cfg.repetition_penalty]], language="<|de|>")


def fake_openvino(texts):
    engine = tr.OpenVinoTranscriber.__new__(tr.OpenVinoTranscriber)
    engine.pipeline, engine.initial_prompt, engine.language = FakePipeline(texts), None, None
    return engine


def test_loop_is_decoded_again_with_penalty():
    """Test that a looped chunk is decoded again with the repetition penalty and the retry is kept."""
    engine = fake_openvino({1.0: LOOP, tr.RETRY_PENALTY: "Das Integral von x² dx ist gleich ein Drittel."})
    text, _ = engine._generate([np.zeros(19 * tr.SAMPLE_RATE, np.float32)])
    assert text == "Das Integral von x² dx ist gleich ein Drittel."
    assert engine.pipeline.penalties == [1.0, tr.RETRY_PENALTY]


def test_loop_in_retry_is_collapsed():
    """Test that a retry that loops too is collapsed instead of dropped."""
    engine = fake_openvino({1.0: LOOP, tr.RETRY_PENALTY: LOOP})
    text, _ = engine._generate([np.zeros(19 * tr.SAMPLE_RATE, np.float32)])
    assert text == "Das Integral von x² dx ist gleich ein Drittel."


def test_normal_chunk_is_decoded_once():
    """Test that normal chunks cost no retry."""
    engine = fake_openvino({1.0: SPEECH})
    assert engine._generate([np.zeros(12 * tr.SAMPLE_RATE, np.float32)])[0] == SPEECH
    assert engine.pipeline.penalties == [1.0]


def level_db(audio):
    return 20 * np.log10(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


@pytest.mark.parametrize("db, expected_db", [(-41.0, -26.0), (-60.0, -40.0), (-20.0, -20.0)])
def test_vad_input_raises_quiet_audio_only(db, expected_db):
    """Test that quiet audio is raised to the target (at most by the max gain) and loud audio is left alone."""
    audio = (np.random.default_rng(0).standard_normal(16_000) * 10 ** (db / 20)).astype(np.float32)
    assert level_db(tr.vad_input(audio)) == pytest.approx(expected_db, abs=0.2)


def test_vad_input_keeps_silence():
    """Test that digital silence and empty audio pass through."""
    assert not tr.vad_input(np.zeros(100, np.float32)).any()
    assert len(tr.vad_input(np.zeros(0, np.float32))) == 0


def test_quiet_speech_is_found(monkeypatch):
    """Test that _speech_groups hands the raised audio to the VAD."""
    seen = []
    monkeypatch.setattr(tr, "get_speech_timestamps", lambda audio, opts: seen.append(level_db(audio)) or [])
    tr._speech_groups((np.random.default_rng(1).standard_normal(16_000) * 0.01).astype(np.float32))
    assert seen == [pytest.approx(-26.0, abs=0.2)]


@pytest.mark.parametrize("returncode, found", [(0, True), (1, False)])
def test_gpu_present(monkeypatch, returncode, found):
    """Test that the probe's exit code says whether the GPU is there."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=returncode))
    assert tr.gpu_present("igpu") is found


def test_gpu_present_survives_a_hanging_probe(monkeypatch):
    """Test that a probe that times out counts as no GPU instead of raising."""
    def hang(*a, **kw):
        raise subprocess.TimeoutExpired("python", 60)
    monkeypatch.setattr(subprocess, "run", hang)
    assert tr.gpu_present("cuda") is False
