"""The duplicate check before the heavy imports (whisperclaude.early)."""
import ctypes
import subprocess
import sys
import uuid

import pytest

from whisperclaude.early import MUTEX_NAME, _open_mutex, quiet_duplicate, single_thread_blas


@pytest.mark.parametrize("env, expected", [
    ({}, "1"), # default: one OpenBLAS thread (less private memory)
    ({"OPENBLAS_NUM_THREADS": "4"}, "4"), # an explicit setting wins
])
def test_single_thread_blas(env, expected):
    environ = dict(env)
    single_thread_blas(environ)
    assert environ["OPENBLAS_NUM_THREADS"] == expected


@pytest.mark.parametrize("argv, exists, expected", [
    (["m", "--autostart"], True, True),
    (["m", "--revive"], True, True),
    (["m", "--autostart"], False, False),
    (["m", "--revive"], False, False),
    (["m"], True, False), # manual launch: the message box must still show
    (["m", "--restarted"], True, False), # Restart waits for the old instance instead
])
def test_quiet_duplicate(argv, exists, expected):
    seen = []

    def fake_open(name: str) -> int:
        seen.append(name)
        return 1234 if exists else 0

    assert quiet_duplicate(argv, open_mutex=fake_open) is expected
    if any(f in argv for f in ("--autostart", "--revive")):
        assert seen == [MUTEX_NAME]


def test_open_mutex_sees_only_existing_mutexes():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    name = f"Local\\WhisperClaudeTest-{uuid.uuid4().hex}"

    assert _open_mutex(name) == 0 # does not exist, and must not be created by the probe
    created = kernel32.CreateMutexW(None, False, name)
    assert created
    try:
        assert _open_mutex(name) != 0
    finally:
        kernel32.CloseHandle(created)


def test_early_import_loads_no_heavy_modules():
    code = ("import sys, whisperclaude.early; "
            "heavy = [m for m in ('numpy', 'tkinter', 'PIL', 'pynput', 'pystray', 'psutil') "
            "if m in sys.modules]; print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
