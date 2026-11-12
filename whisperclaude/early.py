"""Cheap duplicate check that runs before the heavy imports of __main__.

Only ctypes (and collections.abc) are imported here. A logon or unlock task that starts the app while it
already runs exits from this check, without loading numpy, tkinter, PIL, pynput, pystray or
psutil. The authoritative check stays claim_instance() in __main__ (race-free).
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from collections.abc import Callable

MUTEX_NAME = "Local\\WhisperClaude"
SILENT_FLAGS = ("--autostart", "--revive") # duplicates of these exit silently
SYNCHRONIZE = 0x00100000


def single_thread_blas(environ: dict = os.environ) -> None:
    """Default numpy's OpenBLAS to one thread. Must run before numpy is first imported.

    OpenBLAS allocates a buffer per thread at import: with 8 threads numpy alone added 234 MB of
    private memory (9 MB with one thread). Whisper turbo on the iGPU then idled 225 MB less
    (1603 -> 1379 MB private after the warm-up), with identical transcripts (de and en clips),
    and a load time within noise (+0.07 s median, measured 2026-11-12). An explicit
    OPENBLAS_NUM_THREADS in the environment wins.
    """
    environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _open_mutex(name: str) -> int:
    """1 if a mutex with this name exists, else 0. Only opens (and closes) it, never creates it."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE # pointer-sized: the default c_int truncates
    kernel32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, name)
    if not handle:
        return 0
    kernel32.CloseHandle(wintypes.HANDLE(handle))
    return 1


def quiet_duplicate(argv: list[str], open_mutex: Callable[[str], int] = _open_mutex) -> bool:
    """True if this is a silent (autostart / revive) launch and the app already runs."""
    if not any(flag in argv for flag in SILENT_FLAGS):
        return False
    return open_mutex(MUTEX_NAME) != 0
