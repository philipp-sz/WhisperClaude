"""Light probes of the machine: RAM, NVIDIA GPU, the user's languages.

Imported at the top of the UI path (`__main__`) and of the heavy modules (`transcriber`,
`translator`), so it must stay cheap: nothing from the Whisper / translation stack at module level.
The one deferred import, ctranslate2 in cuda_available(), is explained there.
"""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import psutil

from whisperclaude.languages import SOURCE_CODES

log = logging.getLogger(__name__)

MIN_RAM_GB = 12 # "auto" on the CPU with less RAM: the smaller translation model, and CTranslate2
# without MKL (see __main__.lean_cpu_math). A "16 GB" PC reports about 15.5 GiB (the rest is
# reserved), an "8 GB" one about 7.5 GiB. See docs/translation.md


def total_ram_gb() -> float | None:
    """Installed RAM in GiB (what Windows makes usable); None if unknown."""
    try:
        return psutil.virtual_memory().total / 2**30
    except Exception:
        log.info("could not read the RAM size", exc_info=True)
        return None


def user_languages() -> list[str]:
    """ISO codes of the user's own languages, most likely first: the Windows display
    language, the keyboard layouts, then English. Only codes the translation knows."""

    def iso(langid: int) -> str:
        name = ctypes.create_unicode_buffer(85)
        ctypes.windll.kernel32.LCIDToLocaleName(langid & 0xFFFF, name, 85, 0)
        return name.value.split("-")[0].lower()

    codes = [iso(ctypes.windll.kernel32.GetUserDefaultUILanguage())]
    try:
        user32 = ctypes.windll.user32
        n = user32.GetKeyboardLayoutList(0, None)
        layouts = (ctypes.c_void_p * n)()
        user32.GetKeyboardLayoutList(n, layouts)
        codes += [iso(h or 0) for h in layouts]
    except OSError:
        pass
    codes.append("en")
    return [c for i, c in enumerate(codes) if c in SOURCE_CODES and c not in codes[:i]]


class _NvmlMemory(ctypes.Structure):
    """nvmlMemory_t: sizes in bytes."""

    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


def free_vram_mb() -> int | None:
    """Free memory of the first NVIDIA GPU in MB, from NVML (installed with the driver);
    None if unknown."""
    try:
        return _nvml_free_mb()
    except (OSError, AttributeError): # no NVML (no NVIDIA driver, or an old one)
        return None


def _nvml_free_mb() -> int | None:
    """Free memory of GPU 0 in MB via NVML; None if NVML reports an error. Raises OSError or
    AttributeError if nvml.dll or one of its functions is missing."""
    nvml = ctypes.WinDLL("nvml.dll")
    if nvml.nvmlInit_v2() != 0:
        return None
    try:
        handle, memory = ctypes.c_void_p(), _NvmlMemory()
        if (nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0
                or nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0):
            return None
        return memory.free // 2**20
    finally:
        nvml.nvmlShutdown()


def load_cuda_libraries() -> bool:
    """Load cuBLAS and cuDNN from the pip packages of the [cuda] extra (nvidia-cublas-cu12,
    nvidia-cudnn-cu12). CTranslate2 loads them by name only when the GPU is first used, and
    Windows doesn't search the packages' folders: a DLL that is already loaded is found.
    Returns False if they aren't installed."""
    loaded = False
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            spec = None
        for folder in (spec.submodule_search_locations or []) if spec else []:
            for dll in sorted(Path(folder, "bin").glob("*.dll")):
                ctypes.WinDLL(str(dll))
                loaded = True
    return loaded


def cuda_available() -> bool:
    """True if there is an NVIDIA GPU and the CUDA libraries of the [cuda] extra are installed."""
    try:
        # Deferred: ctranslate2 costs ~65 ms and must not be imported before main() has set
        # CT2_USE_MKL (lean_cpu_math); callers are on the loader thread anyway.
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0 and load_cuda_libraries()
    except Exception:
        log.info("CUDA check failed", exc_info=True)
        return False


_PROBES = {
    "igpu": ("import openvino as ov, sys; c = ov.Core(); sys.exit(0 if any(d.startswith('GPU') and 'intel' in "
             "str(c.get_property(d, 'FULL_DEVICE_NAME')).lower() for d in c.available_devices) else 1)"),
    "cuda": "import ctranslate2, sys; sys.exit(0 if ctranslate2.get_cuda_device_count() > 0 else 1)",
}


def gpu_present(kind: str, timeout_s: float = 60.0) -> bool:
    """Whether a GPU of `kind` ("igpu" / "cuda") can be seen now, checked in a fresh process.

    In the running process a vanished device may stay vanished (the device list is read once
    per plugin), so a short-lived subprocess asks instead (0.4 s measured), only run while
    the app is on its CPU fallback (see recovery.py).
    """
    try:
        done = subprocess.run([sys.executable, "-c", _PROBES[kind]], timeout=timeout_s, capture_output=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return done.returncode == 0
    except Exception:
        log.info("GPU check (%s) failed", kind, exc_info=True)
        return False


def foreground_window() -> int:
    """Handle of the window that has the keyboard focus (0 if none, e.g. during a switch)."""
    return int(ctypes.windll.user32.GetForegroundWindow() or 0)
