"""create_transcriber picks CUDA, iGPU or CPU and falls back; fakes, no real models."""
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


class FakeCUDA:
    device = "CUDA"

    def __init__(self, model, on_status=None, **kwargs):
        self.model = model


class BrokenGPU:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("GPU driver too old")


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(t, "Transcriber", FakeCPU)
    monkeypatch.setattr(t, "OpenVinoTranscriber", FakeGPU)
    monkeypatch.setattr(t, "CudaTranscriber", FakeCUDA)
    monkeypatch.setattr(t, "cuda_available", lambda: False)
    return monkeypatch


@pytest.mark.parametrize("device, has_gpu, expected", [
    ("auto", True, "GPU"),
    ("auto", False, "CPU"),
    ("gpu", True, "GPU"),
    ("gpu", False, "CPU"),
    ("igpu", True, "GPU"),
    ("cuda", True, "CPU"),  # no NVIDIA GPU: CPU, not the iGPU
    ("cpu", True, "CPU"),
])
def test_device_selection(fakes, device, has_gpu, expected):
    fakes.setattr(t, "intel_gpu", lambda: "GPU" if has_gpu else None)
    assert t.create_transcriber(device=device).device == expected


@pytest.mark.parametrize("device, expected", [
    ("auto", "CUDA"), ("gpu", "CUDA"), ("cuda", "CUDA"), ("igpu", "GPU"), ("cpu", "CPU"),
])
def test_nvidia_gpu_comes_first(fakes, device, expected):
    fakes.setattr(t, "intel_gpu", lambda: "GPU")
    fakes.setattr(t, "cuda_available", lambda: True)
    assert t.create_transcriber(device=device).device == expected


def test_cuda_failure_falls_back_to_igpu(fakes):
    fakes.setattr(t, "intel_gpu", lambda: "GPU")
    fakes.setattr(t, "cuda_available", lambda: True)
    fakes.setattr(t, "CudaTranscriber", BrokenGPU)
    assert t.create_transcriber(device="auto").device == "GPU"


def test_gpu_failure_falls_back_to_cpu(fakes):
    fakes.setattr(t, "intel_gpu", lambda: "GPU")
    fakes.setattr(t, "OpenVinoTranscriber", BrokenGPU)
    engine = t.create_transcriber(device="auto", cpu_model="small")
    assert engine.device == "CPU" and engine.model == "small"


class FakeCore:
    available_devices = ["CPU", "GPU.0", "GPU.1"]
    names = {"CPU": "Intel Core", "GPU.0": "NVIDIA GeForce RTX 4070 (dGPU)",
             "GPU.1": "Intel(R) Arc(TM) 140V GPU (iGPU)"}

    def get_property(self, device, key):
        return self.names[device]


def test_intel_gpu_skips_nvidia_cards(monkeypatch):
    import openvino
    monkeypatch.setattr(openvino, "Core", FakeCore)
    assert t.intel_gpu() == "GPU.1"
    monkeypatch.setattr(FakeCore, "available_devices", ["CPU", "GPU"])
    monkeypatch.setattr(FakeCore, "names", {"GPU": "NVIDIA GeForce RTX 4070 (dGPU)"})
    assert t.intel_gpu() is None and not t.gpu_available()


def test_ov_repo_name():
    assert t.ov_repo("large-v3-turbo") == "OpenVINO/whisper-large-v3-turbo-int8-ov"


def test_auto_beam_size(fakes):
    seen = {}

    class Recording(FakeCPU):
        def __init__(self, model, **kwargs):
            seen["cpu"] = kwargs["beam_size"]

    fakes.setattr(t, "Transcriber", Recording)
    t.create_transcriber(device="cpu")  # beam_size 0 = auto
    assert seen["cpu"] == 1
    t.create_transcriber(device="cpu", beam_size=4)
    assert seen["cpu"] == 4


@pytest.mark.parametrize("free_mb, expected", [
    (10661, "float16"), (4096, "float16"), (3000, "int8_float16"), (None, "float16"),
])
def test_cuda_precision_follows_free_vram(free_mb, expected):
    assert t.cuda_compute_type(free_mb) == expected


@pytest.mark.parametrize("model, free_mb, expected", [
    ("auto", 10661, ("large-v3", "float16")),       # RTX 4070: the larger model
    ("auto", 5000, ("large-v3", "int8_float16")),   # 2.1 GB, still better than the turbo
    ("auto", 3000, ("large-v3-turbo", "int8_float16")),
    ("auto", None, ("large-v3-turbo", "float16")),  # unknown: the safe choice
    ("large-v3-turbo", 10661, ("large-v3-turbo", "float16")),  # set in config.toml: kept
])
def test_cuda_model_follows_free_vram(model, free_mb, expected):
    assert t.cuda_choice(model, free_mb) == expected


def test_igpu_gets_the_turbo_model_for_auto(fakes):
    seen = []

    class Recording(FakeGPU):
        def __init__(self, model, device, **kwargs):
            seen.append(model)

    fakes.setattr(t, "OpenVinoTranscriber", Recording)
    fakes.setattr(t, "intel_gpu", lambda: "GPU")
    t.create_transcriber(device="igpu")
    assert seen == ["large-v3-turbo"]
