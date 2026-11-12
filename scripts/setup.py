"""Set up WhisperClaude on this PC: detect the hardware, create .venv, download the models, check.

    py scripts\\setup.py                 interactive (asks before downloads and autostart)
    py scripts\\setup.py --dry-run       print every step, change nothing
    py scripts\\setup.py --yes           no questions (models are downloaded; autostart only with --autostart)
    py scripts\\setup.py --no-models     skip the model download (step 3)
    py scripts\\setup.py --autostart     also register the logon and revive tasks (step 5)

Steps: 1 detect the hardware and print what the app will use, 2 create .venv and
`pip install -e .` (with the [cuda] extra on an NVIDIA GPU), 3 download the Whisper and
translation models (sizes first), 4 run scripts\\check_env.py, 5 offer install_autostart.py.

Stdlib only: it runs with any Python >= 3.12 before the venv exists. The decision logic
(Hardware -> Plan) is plain Python and mirrors transcriber.cuda_choice, transcriber.GPU_MODEL,
translator.auto_model and system.MIN_RAM_GB; tests/test_setup.py checks the constants against
the source, so the two cannot drift apart silently.

Exit codes: 0 done (or nothing to do), 1 a step failed, 2 unsupported system (not Windows,
Python < 3.12, Python in .venv too old), 130 interrupted.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
MIN_PYTHON = (3, 12)

# Mirrors of the app's constants (tests/test_setup.py compares them with the source files).
LOW_VRAM_MB = 4096  # transcriber.LOW_VRAM_MB
LARGE_VRAM_MB = 7000  # transcriber.LARGE_VRAM_MB
GPU_MODEL = "large-v3-turbo"  # transcriber.GPU_MODEL: the iGPU model, and the CUDA model when VRAM is unknown
CPU_MODEL = "small"  # config.ModelConfig.cpu_model
TRANSLATOR_LOW_VRAM_MB = 2048  # translator.LOW_VRAM_MB
MIN_RAM_GB = 12  # system.MIN_RAM_GB

# Hugging Face repos the app downloads (faster-whisper's own name mapping, the OpenVINO
# repo of transcriber.ov_repo, and translator.MODELS).
WHISPER_REPOS = {
    ("cuda", "large-v3"): "Systran/faster-whisper-large-v3",
    ("cuda", "large-v3-turbo"): "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    ("igpu", "large-v3-turbo"): "OpenVINO/whisper-large-v3-turbo-int8-ov",
    ("cpu", "small"): "Systran/faster-whisper-small",
}
TRANSLATION_REPOS = {
    "nllb-600m": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
    "nllb-1.3b": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
}
# Rough download sizes (GB), used when the Hub size query fails. Not measured for this PC.
FALLBACK_SIZE_GB = {
    "Systran/faster-whisper-large-v3": 3.1,
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": 1.6,
    "OpenVINO/whisper-large-v3-turbo-int8-ov": 0.9,
    "Systran/faster-whisper-small": 0.5,
    "JustFrederik/nllb-200-distilled-600M-ct2-int8": 0.6,
    "OpenNMT/nllb-200-distilled-1.3B-ct2-int8": 1.4,
}
# GPU memory of a Whisper model after loading (for NLLB's "free VRAM after Whisper"), from docs/transcription.md.
WHISPER_VRAM_MB = {
    ("large-v3", "float16"): 3900,
    ("large-v3", "int8_float16"): 2100,
    ("large-v3-turbo", "float16"): 2200,
    ("large-v3-turbo", "int8_float16"): 1000,
}


@dataclass(frozen=True)
class Hardware:
    """What step 1 found. None = unknown."""
    nvidia_name: str | None = None
    nvidia_free_mb: int | None = None
    intel_gpu_name: str | None = None
    gpu_names: tuple[str, ...] = ()
    ram_gb: float | None = None
    cpu_name: str = ""
    cpu_count: int = 0


@dataclass(frozen=True)
class Plan:
    """What the app will use on this PC (mirrors create_transcriber and translator.auto_model)."""
    device: str  # Whisper: "cuda", "igpu" or "cpu"
    whisper_model: str
    whisper_precision: str
    translation_device: str  # "cuda" or "cpu"
    translation_model: str  # "nllb-1.3b" or "nllb-600m"
    extras: tuple[str, ...]  # pip extras of the project: ("cuda",) on NVIDIA
    reasons: tuple[str, ...]

    @property
    def whisper_repo(self) -> str:
        return WHISPER_REPOS[(self.device, self.whisper_model)]

    @property
    def translation_repo(self) -> str:
        return TRANSLATION_REPOS[self.translation_model]

    @property
    def repos(self) -> list[str]:
        return [self.whisper_repo, self.translation_repo]


# ---- decision logic (no I/O) ---------------------------------------------------------------

def cuda_choice(free_mb: int | None) -> tuple[str, str]:
    """(model, precision) on CUDA for model "auto" (transcriber.cuda_choice)."""
    if free_mb is None:
        return GPU_MODEL, "float16"
    if free_mb >= LARGE_VRAM_MB:
        return "large-v3", "float16"
    if free_mb >= LOW_VRAM_MB:
        return "large-v3", "int8_float16"
    return GPU_MODEL, "int8_float16"


def auto_model(device: str, free_vram_mb: int | None, ram_gb: float | None) -> str:
    """translator.auto_model: NLLB-1.3B unless memory is short."""
    if device == "cuda":
        small = free_vram_mb is not None and free_vram_mb < TRANSLATOR_LOW_VRAM_MB
    else:
        small = ram_gb is None or ram_gb < MIN_RAM_GB
    return "nllb-600m" if small else "nllb-1.3b"


def plan_for(hw: Hardware) -> Plan:
    """Hardware -> what the app will use. Same order as create_transcriber: NVIDIA (CUDA),
    then Intel GPU (OpenVINO), then the CPU. Translation runs on CUDA or the CPU only."""
    if hw.nvidia_name:
        model, precision = cuda_choice(hw.nvidia_free_mb)
        after_whisper = None
        if hw.nvidia_free_mb is not None:
            after_whisper = hw.nvidia_free_mb - WHISPER_VRAM_MB[(model, precision)]
        free = "unknown" if hw.nvidia_free_mb is None else f"{hw.nvidia_free_mb} MB"
        reasons = (f"NVIDIA GPU {hw.nvidia_name} with {free} VRAM free: Whisper on CUDA",)
        return Plan("cuda", model, precision, "cuda",
                    auto_model("cuda", after_whisper, hw.ram_gb), ("cuda",), reasons)
    if hw.intel_gpu_name:
        reasons = (f"Intel GPU {hw.intel_gpu_name}: Whisper on the GPU via OpenVINO",)
        return Plan("igpu", GPU_MODEL, "int8", "cpu",
                    auto_model("cpu", None, hw.ram_gb), (), reasons)
    reasons = ("no NVIDIA or Intel GPU found: Whisper on the CPU",)
    return Plan("cpu", CPU_MODEL, "int8", "cpu", auto_model("cpu", None, hw.ram_gb), (), reasons)


def ram_note(ram_gb: float | None) -> str:
    if ram_gb is None:
        return "RAM unknown: the smaller translation model"
    if ram_gb < MIN_RAM_GB:
        return f"{ram_gb:.1f} GiB RAM (< {MIN_RAM_GB}): smaller translation model, CTranslate2 without MKL"
    return f"{ram_gb:.1f} GiB RAM"


# ---- detection (Windows APIs, PowerShell) ----------------------------------------------------

def is_windows() -> bool:
    return sys.platform == "win32"


def parse_video_names(output: str) -> list[str]:
    """Names from `Get-CimInstance Win32_VideoController`, one per line."""
    return [line.strip() for line in output.splitlines() if line.strip()]


def intel_gpu_name(names: list[str]) -> str | None:
    """An Intel GPU, as OpenVINO would list it (the app checks the full device name for 'intel')."""
    return next((n for n in names if "intel" in n.lower()), None)


def detect_ram_gb() -> float | None:
    """Installed RAM in GiB (GlobalMemoryStatusEx), or None."""

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong)]

    try:
        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.total_phys / 2**30
    except (OSError, AttributeError):
        return None


def detect_nvidia() -> tuple[str | None, int | None]:
    """(name, free VRAM MB) of the first NVIDIA GPU via NVML (nvml.dll ships with the driver),
    like system.free_vram_mb; (None, None) if there is none."""

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    try:
        nvml = ctypes.WinDLL("nvml.dll")
        if nvml.nvmlInit_v2() != 0:
            return None, None
        try:
            handle, memory = ctypes.c_void_p(), Memory()
            if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return None, None
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                free_mb = None
            else:
                free_mb = memory.free // 2**20
            name_buffer = ctypes.create_string_buffer(96)
            name = None
            if nvml.nvmlDeviceGetName(handle, name_buffer, 96) == 0:
                name = name_buffer.value.decode("ascii", "replace")
            return name, free_mb
        finally:
            nvml.nvmlShutdown()
    except (OSError, AttributeError):
        return None, None


def detect_video_names() -> list[str]:
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return parse_video_names(done.stdout) if done.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


def detect_cpu_name() -> str:
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return done.stdout.strip().splitlines()[0] if done.returncode == 0 and done.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_hardware() -> Hardware:
    names = detect_video_names()
    nvidia_name, free_mb = detect_nvidia()
    if nvidia_name is None:  # NVML unavailable (free VRAM unknown) but the driver lists the card
        nvidia_name = next((n for n in names if "nvidia" in n.lower()), None)
    return Hardware(nvidia_name=nvidia_name, nvidia_free_mb=free_mb,
                    intel_gpu_name=intel_gpu_name(names), gpu_names=tuple(names),
                    ram_gb=detect_ram_gb(), cpu_name=detect_cpu_name(),
                    cpu_count=os.cpu_count() or 0)


# ---- model sizes and cache -------------------------------------------------------------------

def hub_size_gb(repo: str) -> float | None:
    """Total size of all files in the repo (what snapshot_download fetches), from the Hub API.
    Read-only; None if the Hub can't be reached."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true",
                                    timeout=10) as response:
            info = json.load(response)
        return sum(f.get("size") or 0 for f in info.get("siblings", [])) / 1e9
    except (OSError, ValueError):
        return None


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def is_cached(repo: str) -> bool:
    """True if the repo has a snapshot in the Hugging Face cache (same location as the app uses)."""
    snapshots = hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


# ---- output and running commands -------------------------------------------------------------

class Runner:
    """Runs (or, in dry-run, only prints) the commands of the steps."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(self, cmd: list[str], cwd: Path = ROOT, shown: str | None = None) -> bool:
        """Run cmd; `shown` replaces the command line in the output (for long inline scripts)."""
        text = shown or subprocess.list2cmdline(cmd)
        print("  $ " + text + ("   (dry-run, not run)" if self.dry_run else ""))
        if self.dry_run:
            return True
        return subprocess.run(cmd, cwd=cwd).returncode == 0


def heading(number: int, title: str) -> None:
    print(f"\n[{number}/5] {title}")


def print_hardware(hw: Hardware, plan: Plan) -> None:
    print(f"  CPU:       {hw.cpu_name or 'unknown'} ({hw.cpu_count} logical cores)")
    print(f"  RAM:       {ram_note(hw.ram_gb)}")
    print(f"  GPUs:      {', '.join(hw.gpu_names) or 'none listed'}")
    if hw.nvidia_name:
        free = "unknown" if hw.nvidia_free_mb is None else f"{hw.nvidia_free_mb} MB free"
        print(f"  NVIDIA:    {hw.nvidia_name}, {free}")
    print("  What the app will use:")
    for reason in plan.reasons:
        print(f"    - {reason}")
    precision = {"float16": "float16", "int8_float16": "int8 weights, float16 compute",
                 "int8": "int8"}[plan.whisper_precision]
    print(f"    Whisper:      {plan.whisper_model} ({precision}) on {plan.device}")
    print(f"    Translation:  {plan.translation_model} on {plan.translation_device}")
    print(f"    pip extras:   {', '.join(plan.extras) or 'none'}")


def step_detect() -> Plan | None:
    heading(1, "Detect the hardware")
    if not is_windows():
        print("  WhisperClaude runs on Windows only.")
        return None
    hw = detect_hardware()
    plan = plan_for(hw)
    print_hardware(hw, plan)
    return plan


def python_ok(interpreter: Path | str) -> bool:
    """True if the interpreter is >= MIN_PYTHON (checked in a subprocess)."""
    code = f"import sys; sys.exit(0 if sys.version_info[:2] >= {MIN_PYTHON} else 1)"
    return subprocess.run([str(interpreter), "-c", code], capture_output=True).returncode == 0


def step_venv(runner: Runner, plan: Plan) -> bool:
    heading(2, "Create .venv and install WhisperClaude")
    if VENV_PY.exists():
        if not runner.dry_run and not python_ok(VENV_PY):
            print(f"  {VENV} uses a Python older than {MIN_PYTHON[0]}.{MIN_PYTHON[1]}: delete the folder and run again.")
            return False
        print(f"  {VENV} exists, reusing it.")
    else:
        if not runner.run([sys.executable, "-m", "venv", str(VENV)]):
            return False
    target = str(ROOT) + (f"[{','.join(plan.extras)}]" if plan.extras else "")
    return runner.run([str(VENV_PY), "-m", "pip", "install", "-e", target])


def confirm(question: str) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        print(f"  {question} [y/N] no terminal: answering no (rerun with --yes)")
        return False
    try:
        return input(f"  {question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        print()  # no answer (input closed): end the prompt's line
        return False


DOWNLOAD_CODE = (
    "import sys\n"
    "from huggingface_hub import snapshot_download\n"
    "for repo in sys.argv[1:]:\n"
    "    print('->', repo, flush=True)\n"
    "    try:\n"
    "        snapshot_download(repo, local_files_only=True)\n"
    "    except Exception:\n"
    "        snapshot_download(repo)\n"
)


def step_models(runner: Runner, plan: Plan, assume_yes: bool) -> bool:
    heading(3, "Download the models")
    sizes = {}
    for repo in plan.repos:
        size = hub_size_gb(repo)
        sizes[repo] = size if size is not None else FALLBACK_SIZE_GB[repo]
        size_text = f"{sizes[repo]:.1f} GB" if size is not None else f"about {sizes[repo]} GB (estimate)"
        state = "cached" if is_cached(repo) else "missing"
        print(f"  {repo}: {size_text}, {state}")
    missing = [repo for repo in plan.repos if not is_cached(repo)]
    if not missing:
        print("  All models are already in the Hugging Face cache.")
        return True
    print(f"  To download: about {sum(sizes[r] for r in missing):.1f} GB.")
    if not assume_yes and not runner.dry_run and not confirm("Download now?"):
        print("  Skipped. The app downloads missing models on its first start.")
        return True
    shown = f"python -c <download script> {' '.join(missing)}"
    return runner.run([str(VENV_PY), "-I", "-c", DOWNLOAD_CODE, *missing], shown=shown)


def step_check(runner: Runner) -> bool:
    heading(4, "Check the environment")
    return runner.run([str(VENV_PY), str(ROOT / "scripts" / "check_env.py")])


def step_autostart(runner: Runner, args: argparse.Namespace) -> bool:
    heading(5, "Start at login (optional)")
    script = ROOT / "scripts" / "install_autostart.py"
    if args.autostart or (not args.yes and not runner.dry_run and confirm("Start WhisperClaude at Windows login?")):
        return runner.run([str(VENV_PY), str(script)])
    print(f"  Later: {VENV_PY} {script}")
    return True


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up WhisperClaude on this PC.")
    parser.add_argument("--dry-run", action="store_true", help="print the steps, change nothing")
    parser.add_argument("--yes", action="store_true", help="no questions: download the models")
    parser.add_argument("--no-models", action="store_true", help="skip step 3 (model download)")
    parser.add_argument("--autostart", action="store_true", help="register autostart in step 5")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not is_windows():
        print("WhisperClaude runs on Windows only.")
        return 2
    if sys.version_info[:2] < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed (this one is {sys.version.split()[0]}).")
        return 2
    runner = Runner(args.dry_run)
    try:
        if args.dry_run:
            print("Dry run: nothing is installed or downloaded.")
        plan = step_detect()
        if plan is None:
            return 2
        if not step_venv(runner, plan):
            return 1
        if args.no_models:
            heading(3, "Download the models")
            print("  Skipped (--no-models).")
        elif not step_models(runner, plan, args.yes):
            return 1
        if not step_check(runner):
            return 1
        if not step_autostart(runner, args):
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print("\nDone. Start the app with:")
    print(f"  {VENV / 'Scripts' / 'pythonw.exe'} -m whisperclaude   (no console)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
