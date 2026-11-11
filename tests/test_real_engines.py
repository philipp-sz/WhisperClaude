"""The comparison session of scripts/real_engines.py with fake engines: caching, missing engines, fallbacks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import real_data as rd # noqa: E402
from scripts import real_engines as re_ # noqa: E402


class Fake:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text


@pytest.fixture
def session(tmp_path):
    s = re_.Session(tmp_path, python=None)
    s.rec = rd.save_recording(np.zeros(1600, np.float32), tmp_path)
    return s


def test_an_engine_runs_once_per_recording_and_is_then_cached(session):
    session.engines["whisper"] = fake = Fake("Hallo Welt")
    first = session.transcribe(session.rec, "whisper")
    again = session.transcribe(session.rec, "whisper")
    assert first["text"] == again["text"] == "Hallo Welt" and fake.calls == 1
    session.transcribe(session.rec, "whisper", force=True)
    assert fake.calls == 2


def test_the_cache_survives_a_new_session(session, tmp_path):
    session.engines["parakeet"] = Fake("aus dem Cache")
    session.transcribe(session.rec, "parakeet")
    assert re_.Session(tmp_path).transcribe(session.rec, "parakeet")["text"] == "aus dem Cache"


def test_parakeet_is_offered_only_with_a_torch_environment(tmp_path):
    assert re_.Session(tmp_path, python=None).available() == ["whisper"]
    assert re_.Session(tmp_path, python="C:/x/python.exe").available() == ["whisper", "parakeet"]


def test_outputs_collect_every_cached_engine(session):
    """Test that the window gets the cached output of each engine that has one."""
    rd.save_hyp("whisper", session.rec.id, "a", 0.1, session.directory)
    rd.save_hyp("granite", session.rec.id, "b", 0.1, session.directory)
    assert set(session.outputs(session.rec)) == {"whisper", "granite"}


@pytest.mark.parametrize("device, last, refused", [("CPU", "igpu", True), ("CPU", None, False), ("GPU", "igpu", False)])
def test_make_local_refuses_the_silent_cpu_fallback(monkeypatch, tmp_path, device, last, refused):
    """Test that a comparison never runs Whisper small labelled as the GPU model."""
    from whisperclaude import recovery, transcriber
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    (tmp_path / "WhisperClaude").mkdir()
    if last:
        recovery.remember_device(tmp_path / "WhisperClaude", last)
    monkeypatch.setattr(transcriber, "create_transcriber", lambda **kw: type("T", (), {"device": device})())
    if refused:
        with pytest.raises(RuntimeError, match="fell back to the CPU"):
            re_.make_local("whisper")
    else:
        assert re_.make_local("whisper").device == device


def test_a_missing_torch_environment_is_explained(monkeypatch):
    monkeypatch.delenv("WC_TORCH_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="WC_TORCH_PYTHON"):
        re_.torch_python(None)
