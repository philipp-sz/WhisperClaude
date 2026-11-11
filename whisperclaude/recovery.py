"""Self-healing when the GPU goes missing: notice it, look for it again, switch back, warn late.

On 2026-11-10 the laptop's Intel GPU vanished from OpenVINO after a wake: the app started on
the CPU (Whisper `small`: twice the errors, 4.5 s instead of 0.05 s after the stop) and stayed
there for a day without saying so; only a reboot brought the GPU back. Now:

- The device of the last successful load is remembered (`device.txt` in the data folder). A CPU
  load on a machine that last ran on a GPU is a degradation; on a machine without a GPU the
  CPU is normal and nothing happens.
- While degraded, the GPU is looked for again (gpu_present(), a fresh subprocess) a minute after
  the start, at every resume, and every PROBE_EVERY_S. When it is back, the model is loaded on
  it in the background and replaces the CPU model as soon as the app is idle.
- Only after NOTICE_AFTER failed checks does the user get one notice (restarting Windows is
  what helped).

GpuWatch is the pure scheduling logic (a clock is injected), so it can be tested directly.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

FIRST_PROBE_S = 60.0 # the GPU driver may still be starting right after a wake or logon
PROBE_EVERY_S = 300.0
NOTICE_AFTER = 2 # failed checks before the user is told
GPU_KINDS = ("cuda", "igpu")
DEVICE_FILE = "device.txt"


def device_kind(device: str) -> str:
    """"cuda", "igpu" or "cpu" for a transcriber's device ("CUDA", "GPU", "GPU.1", "CPU")."""
    return "cuda" if device == "CUDA" else "igpu" if device.startswith("GPU") else "cpu"


def last_device(data_dir: Path) -> str | None:
    """The device kind of the last successful model load, or None if unknown."""
    try:
        kind = (data_dir / DEVICE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return kind if kind in (*GPU_KINDS, "cpu") else None


def remember_device(data_dir: Path, kind: str) -> None:
    """Stores the device kind of a successful load (best effort)."""
    try:
        (data_dir / DEVICE_FILE).write_text(kind, encoding="utf-8")
    except OSError:
        pass


class GpuWatch:
    """Decides when to look for a missing GPU again and when to tell the user."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.missing: str | None = None # the GPU kind we should be on, while we are on the CPU
        self.probing = False
        self.failures = 0
        self.noticed = False
        self.next_probe = 0.0

    def loaded(self, kind: str, expected: str | None) -> None:
        """A model was loaded on `kind`; `expected` is the kind of the last good load."""
        degraded = kind == "cpu" and expected in GPU_KINDS
        self.missing = expected if degraded else None
        self.probing, self.failures, self.noticed = False, 0, False
        self.next_probe = self.clock() + FIRST_PROBE_S

    def due(self, resumed: bool = False) -> bool:
        """True if a check should start now (the caller runs it and reports probe_done)."""
        if self.missing is None or self.probing:
            return False
        if resumed or self.clock() >= self.next_probe:
            self.probing = True
            return True
        return False

    def probe_done(self, found: bool) -> bool:
        """Records a check's result; True if the user should be told now (once)."""
        self.probing = False
        if found:
            self.missing = None # loading on the GPU starts; loaded() follows
            return False
        self.failures += 1
        self.next_probe = self.clock() + PROBE_EVERY_S
        if self.failures >= NOTICE_AFTER and not self.noticed:
            self.noticed = True
            return True
        return False
