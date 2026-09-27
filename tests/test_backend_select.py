"""create_transcriber picks iGPU or CPU and falls back to CPU; fakes, no real models."""
import pytest

from whisperclaude import transcriber as t


class FakeCPU:
    device = "CPU"

    def __init__(self, model, **kwargs):
        self.model = model


class FakeGPU:
    device = "GPU"

    def __init__(self, model, device, on_status=None, **kwargs):
        self.model = model


class BrokenGPU:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("GPU driver too old")


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(t, "Transcriber", FakeCPU)
    monkeypatch.setattr(t, "OpenVinoTranscriber", FakeGPU)
    return monkeypatch


@pytest.mark.parametrize("device, has_gpu, expected", [
    ("auto", True, "GPU"),
    ("auto", False, "CPU"),
    ("gpu", True, "GPU"),
    ("gpu", False, "CPU"),
    ("cpu", True, "CPU"),
])
def test_device_selection(fakes, device, has_gpu, expected):
    fakes.setattr(t, "gpu_available", lambda: has_gpu)
    assert t.create_transcriber(device=device).device == expected


def test_gpu_failure_falls_back_to_cpu(fakes):
    fakes.setattr(t, "gpu_available", lambda: True)
    fakes.setattr(t, "OpenVinoTranscriber", BrokenGPU)
    engine = t.create_transcriber(device="auto", cpu_model="small")
    assert engine.device == "CPU" and engine.model == "small"


def test_ov_repo_name():
    assert t.ov_repo("large-v3-turbo") == "OpenVINO/whisper-large-v3-turbo-int8-ov"
