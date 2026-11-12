"""scripts/setup.py: hardware -> plan decisions (fake hardware), constants kept in sync with the app,
the HF cache check, and that --dry-run runs nothing."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "whisperclaude"


def load_setup():
    spec = importlib.util.spec_from_file_location("wc_setup", ROOT / "scripts" / "setup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it executes
    spec.loader.exec_module(module)
    return module


setup = load_setup()


def hw(**kwargs):
    base = dict(nvidia_name=None, nvidia_free_mb=None, intel_gpu_name=None, gpu_names=(),
                ram_gb=31.5, cpu_name="cpu", cpu_count=8)
    base.update(kwargs)
    return setup.Hardware(**base)


RTX_4070 = dict(nvidia_name="NVIDIA GeForce RTX 4070", nvidia_free_mb=11000)
ARC_140V = dict(intel_gpu_name="Intel(R) Arc(TM) 140V GPU (16GB)")


# ---- plan decisions ----------------------------------------------------------------------------

@pytest.mark.parametrize("hardware, expected", [
    # desktop: RTX 4070 with plenty of VRAM -> large-v3 float16 on CUDA, NLLB-1.3B on CUDA
    (hw(**RTX_4070), ("cuda", "large-v3", "float16", "cuda", "nllb-1.3b", ("cuda",))),
    # 6 GB free: large-v3 int8 (2.1 GB), 2.9 GB left for NLLB -> 1.3B
    (hw(nvidia_name="x", nvidia_free_mb=6000), ("cuda", "large-v3", "int8_float16", "cuda", "nllb-1.3b", ("cuda",))),
    # 3 GB free: turbo int8 (1.0 GB), 2.0 GB left after it -> NLLB-600M (less than 2048 MB)
    (hw(nvidia_name="x", nvidia_free_mb=3000), ("cuda", "large-v3-turbo", "int8_float16", "cuda", "nllb-600m", ("cuda",))),
    # 3.5 GB free: turbo int8 leaves 2.5 GB -> NLLB-1.3B still fits
    (hw(nvidia_name="x", nvidia_free_mb=3500), ("cuda", "large-v3-turbo", "int8_float16", "cuda", "nllb-1.3b", ("cuda",))),
    # VRAM unknown (NVML failed): turbo float16, NLLB-1.3B (unknown counts as big)
    (hw(nvidia_name="x", nvidia_free_mb=None), ("cuda", "large-v3-turbo", "float16", "cuda", "nllb-1.3b", ("cuda",))),
    # laptop: Intel Arc iGPU, 31.5 GiB RAM -> turbo on OpenVINO, NLLB-1.3B on the CPU
    (hw(**ARC_140V), ("igpu", "large-v3-turbo", "int8", "cpu", "nllb-1.3b", ())),
    # weak PC: no GPU, 8 GB RAM -> small on the CPU, NLLB-600M
    (hw(ram_gb=7.5), ("cpu", "small", "int8", "cpu", "nllb-600m", ())),
    # 16 GB PC reports about 15.5 GiB -> NLLB-1.3B
    (hw(ram_gb=15.5), ("cpu", "small", "int8", "cpu", "nllb-1.3b", ())),
    # RAM unknown -> the smaller translation model
    (hw(ram_gb=None), ("cpu", "small", "int8", "cpu", "nllb-600m", ())),
    # NVIDIA and Intel: CUDA wins (same order as create_transcriber)
    (hw(intel_gpu_name="Intel(R) UHD Graphics", **RTX_4070), ("cuda", "large-v3", "float16", "cuda", "nllb-1.3b", ("cuda",))),
    # NVIDIA card only listed by the driver (NVML failed): CUDA with unknown VRAM
    (hw(nvidia_name="NVIDIA GeForce GTX 1650", nvidia_free_mb=None, ram_gb=7.5),
     ("cuda", "large-v3-turbo", "float16", "cuda", "nllb-1.3b", ("cuda",))),
    # weak laptop with iGPU: Whisper on the iGPU, translation on the CPU with the small model
    (hw(ram_gb=7.5, **ARC_140V), ("igpu", "large-v3-turbo", "int8", "cpu", "nllb-600m", ())),
])
def test_plan_for_hardware(hardware, expected):
    plan = setup.plan_for(hardware)
    device, whisper_model, precision, tdevice, tmodel, extras = expected
    assert (plan.device, plan.whisper_model, plan.whisper_precision) == (device, whisper_model, precision)
    assert (plan.translation_device, plan.translation_model, plan.extras) == (tdevice, tmodel, extras)
    assert plan.reasons


def test_repos_follow_the_plan():
    plan = setup.plan_for(hw(**RTX_4070))
    assert plan.whisper_repo == "Systran/faster-whisper-large-v3"
    assert plan.translation_repo == "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
    assert setup.plan_for(hw(**ARC_140V)).whisper_repo == "OpenVINO/whisper-large-v3-turbo-int8-ov"
    assert setup.plan_for(hw(ram_gb=7.5)).repos == [
        "Systran/faster-whisper-small", "JustFrederik/nllb-200-distilled-600M-ct2-int8"]


def test_every_plan_repo_has_a_size_estimate():
    for hardware in (hw(**RTX_4070), hw(nvidia_name="x", nvidia_free_mb=3000), hw(**ARC_140V), hw(ram_gb=7.5)):
        for repo in setup.plan_for(hardware).repos:
            assert repo in setup.FALLBACK_SIZE_GB


def test_intel_gpu_name_and_video_parsing():
    names = setup.parse_video_names("NVIDIA GeForce RTX 4070\n\n  Intel(R) UHD Graphics 630 \n")
    assert names == ["NVIDIA GeForce RTX 4070", "Intel(R) UHD Graphics 630"]
    assert setup.intel_gpu_name(names) == "Intel(R) UHD Graphics 630"
    assert setup.intel_gpu_name(["NVIDIA GeForce RTX 4070"]) is None


# ---- constants mirrored from the app: fail when the app changes ----------------------------------

def read_const(relpath, pattern):
    match = re.search(pattern, (PKG / relpath).read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{pattern} not found in {relpath}"
    return match.group(1)


def test_thresholds_match_the_app():
    assert setup.LOW_VRAM_MB == int(read_const("transcriber.py", r"^LOW_VRAM_MB = (\d+)"))
    assert setup.LARGE_VRAM_MB == int(read_const("transcriber.py", r"^LARGE_VRAM_MB = (\d+)"))
    assert setup.GPU_MODEL == read_const("transcriber.py", r'^GPU_MODEL = "([^"]+)"')
    assert setup.TRANSLATOR_LOW_VRAM_MB == int(read_const("translator.py", r"^LOW_VRAM_MB = (\d+)"))
    assert setup.MIN_RAM_GB == int(read_const("system.py", r"^MIN_RAM_GB = (\d+)"))
    assert setup.CPU_MODEL == read_const("config.py", r'cpu_model: str = "([^"]+)"')


def test_translation_repos_match_the_app():
    source = (PKG / "translator.py").read_text(encoding="utf-8")
    for name, repo in setup.TRANSLATION_REPOS.items():
        assert f'"{name}": "{repo}"' in source


def test_openvino_repo_matches_the_app():
    source = (PKG / "transcriber.py").read_text(encoding="utf-8")
    assert 'f"OpenVINO/whisper-{model}-int8-ov"' in source
    assert setup.WHISPER_REPOS[("igpu", setup.GPU_MODEL)] == f"OpenVINO/whisper-{setup.GPU_MODEL}-int8-ov"


# ---- cache, detection helpers ------------------------------------------------------------------

def test_is_cached_reads_the_hf_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = "Systran/faster-whisper-small"
    assert not setup.is_cached(repo)
    (tmp_path / "models--Systran--faster-whisper-small" / "snapshots").mkdir(parents=True)
    assert not setup.is_cached(repo)  # empty snapshots folder is not a download
    (tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "abc").mkdir()
    assert setup.is_cached(repo)


def test_cuda_choice_thresholds():
    assert setup.cuda_choice(7000) == ("large-v3", "float16")
    assert setup.cuda_choice(6999) == ("large-v3", "int8_float16")
    assert setup.cuda_choice(4096) == ("large-v3", "int8_float16")
    assert setup.cuda_choice(4095) == ("large-v3-turbo", "int8_float16")
    assert setup.cuda_choice(None) == ("large-v3-turbo", "float16")


# ---- main() ------------------------------------------------------------------------------------

def test_non_windows_exits_2(monkeypatch):
    monkeypatch.setattr(setup, "is_windows", lambda: False)
    assert setup.main(["--dry-run"]) == 2


def test_dry_run_runs_nothing(monkeypatch, tmp_path, capsys):
    """--dry-run must not create a venv, install, download or run anything."""
    monkeypatch.setattr(setup, "is_windows", lambda: True)
    monkeypatch.setattr(setup, "detect_hardware", lambda: hw(**RTX_4070))
    monkeypatch.setattr(setup, "hub_size_gb", lambda repo: None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    def no_subprocess(*args, **kwargs):
        raise AssertionError(f"dry run started a process: {args}")

    monkeypatch.setattr(setup.subprocess, "run", no_subprocess)
    monkeypatch.setattr(setup, "VENV", tmp_path / ".venv")
    monkeypatch.setattr(setup, "VENV_PY", tmp_path / ".venv" / "Scripts" / "python.exe")
    assert setup.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "large-v3" in out and "nllb-1.3b" in out
    assert "-m venv" in out and "pip install -e" in out and "[cuda]" in out
    assert "dry-run, not run" in out
    assert not (tmp_path / ".venv").exists()


def test_confirm_without_answer_is_no(monkeypatch, capsys):
    class Terminal:
        def isatty(self):
            return True

    def closed(prompt=""):
        raise EOFError

    monkeypatch.setattr(setup.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", closed)
    assert setup.confirm("Download now?") is False
    assert capsys.readouterr().out == "\n"  # input() is faked here, so only the newline is printed


def test_no_models_skips_step_3(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(setup, "is_windows", lambda: True)
    monkeypatch.setattr(setup, "detect_hardware", lambda: hw(**ARC_140V))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(setup, "VENV_PY", tmp_path / "python.exe")
    assert setup.main(["--dry-run", "--no-models"]) == 0
    assert "Skipped (--no-models)" in capsys.readouterr().out
