"""The engines the comparison tools can run, behind one interface: engine.transcribe(audio) -> text.

- "whisper": the app's own engine (create_transcriber with your config.toml, the manual vocabulary only: the
  automatic one would read docs that quote the references), in this process.
- "parakeet", "granite", "granite2b": PyTorch models of scripts/model_search, run in the separate environment
  named by the environment variable WC_TORCH_PYTHON (or --torch-python) through transcribe_cli.py; the model is
  loaded for every call, so these are slow to start.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.real_data import REAL_DIR, Recording, load_hyps, read_wav, save_hyp, write_wav # noqa: E402

LOCAL = ("whisper",)
TORCH = ("parakeet", "granite", "granite2b")
BRIDGE = ROOT / "scripts" / "model_search" / "transcribe_cli.py"


def make_local(name: str):
    """The app's Whisper as configured in config.toml; never the silent CPU fallback on a GPU machine.

    On 2026-11-10 the GPU vanished and a day of comparisons ran Whisper `small` on the CPU while
    being labelled "whisper"; now that is an error.
    """
    from whisperclaude import recovery
    from whisperclaude.config import load_config
    from whisperclaude.transcriber import create_transcriber

    m = load_config().model
    engine = create_transcriber(
        device=m.device, gpu_model=m.gpu_model, cpu_model=m.cpu_model, language=m.language or None,
        initial_prompt=m.initial_prompt or None, vocabulary=m.vocabulary, compute_type=m.compute_type,
        beam_size=m.beam_size, cpu_threads=m.cpu_threads, batch_size=m.batch_size)
    data_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude"
    expected = recovery.last_device(data_dir)
    if recovery.device_kind(engine.device) == "cpu" and expected in recovery.GPU_KINDS:
        raise RuntimeError(f"{name}: the {expected} is missing and Whisper fell back to the CPU model; "
                           "results would not be comparable (restart Windows, see docs/real-recordings.md)")
    return engine


def torch_python(given: str | None = None) -> str:
    """The Python of the PyTorch environment (WC_TORCH_PYTHON), or an error that says how to set it."""
    path = given or os.environ.get("WC_TORCH_PYTHON")
    if not path or not Path(path).exists():
        raise RuntimeError("set WC_TORCH_PYTHON (or --torch-python) to the python.exe of the environment described in "
                           "scripts/model_search/README.md")
    return path


def run_bridge(model: str, wavs: list[Path], out: Path, python: str, keywords: str = "") -> dict[str, dict]:
    """Transcribes WAV files with a PyTorch model through transcribe_cli.py; {file stem: {"text", "s"}}."""
    out.unlink(missing_ok=True)
    command = [python, str(BRIDGE), model, str(out), *(["--keywords", keywords] if keywords else []),
               *map(str, wavs)]
    subprocess.run(command, check=True, env={**os.environ, "PYTHONUTF8": "1"}, cwd=BRIDGE.parent)
    return json.loads(out.read_text(encoding="utf-8"))


class TorchEngine:
    """A PyTorch model as an engine with transcribe(audio); one subprocess (and model load) per call."""

    def __init__(self, model: str, python: str | None = None) -> None:
        self.model, self.python = model, torch_python(python)

    def transcribe(self, audio: np.ndarray) -> str:
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / "audio.wav"
            write_wav(wav, audio)
            return run_bridge(self.model, [wav], Path(folder) / "out.json", self.python)["audio"]["text"]


def fill_cache(name: str, recordings: list[Recording], force: bool = False, python: str | None = None,
               directory: Path = REAL_DIR) -> None:
    """Transcribes every recording the engine has no cached output for, and caches the outputs."""
    cached = load_hyps(name, directory)
    todo = [r for r in recordings if force or r.id not in cached]
    if not todo:
        return
    if name in TORCH:
        with tempfile.TemporaryDirectory() as folder:
            got = run_bridge(name, [r.wav for r in todo], Path(folder) / "out.json", torch_python(python))
        for rec in todo:
            save_hyp(name, rec.id, got[rec.id]["text"], got[rec.id]["s"], directory)
        return
    engine = make_local(name)
    for rec in todo:
        audio = read_wav(rec.wav)
        start = time.perf_counter()
        text = engine.transcribe(audio)
        save_hyp(name, rec.id, text, time.perf_counter() - start, directory)


class Session:
    """The engines and caches behind the comparison window (no UI): transcribe a recording with an engine once.

    `engines` maps names to objects with transcribe(audio); load() fills it with the real engines, tests
    put fakes in. Outputs are cached per engine and recording (scripts/real_data.py).
    """

    def __init__(self, directory: Path = REAL_DIR, python: str | None = None) -> None:
        self.directory = directory
        self.python = python or os.environ.get("WC_TORCH_PYTHON")
        self.engines: dict[str, object] = {}

    def available(self) -> list[str]:
        """Engine names the window offers: Whisper, and Parakeet if the PyTorch environment is set."""
        return [*LOCAL, *(["parakeet"] if self.python else [])]

    def load(self, name: str) -> None:
        """Loads an engine (slow: Whisper compiles kernels, the PyTorch models start a process per call)."""
        self.engines[name] = TorchEngine(name, self.python) if name in TORCH else make_local(name)

    def transcribe(self, rec: Recording, name: str, force: bool = False) -> dict:
        """{"text", "s"} of the engine for the recording: from the cache, else by running the loaded engine."""
        cached = load_hyps(name, self.directory).get(rec.id)
        if cached and not force:
            return cached
        audio = read_wav(rec.wav)
        start = time.perf_counter()
        text = self.engines[name].transcribe(audio)
        seconds = time.perf_counter() - start
        save_hyp(name, rec.id, text, seconds, self.directory)
        return {"text": text, "s": round(seconds, 3)}

    def outputs(self, rec: Recording) -> dict[str, dict]:
        """The cached outputs of all engines for the recording."""
        return {n: h[rec.id] for n in (*LOCAL, *TORCH) if (h := load_hyps(n, self.directory)).get(rec.id)}
