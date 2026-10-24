"""Measuring helpers for scripts: battery power (system-wide) and this process's GPU time."""
import ctypes
import os
import subprocess
import threading
import time
from ctypes import wintypes


class _BatteryState(ctypes.Structure):
    _fields_ = [("AcOnLine", ctypes.c_ubyte), ("BatteryPresent", ctypes.c_ubyte),
                ("Charging", ctypes.c_ubyte), ("Discharging", ctypes.c_ubyte),
                ("Spare1", ctypes.c_ubyte * 3), ("Tag", ctypes.c_ubyte),
                ("MaxCapacity", wintypes.DWORD), ("RemainingCapacity", wintypes.DWORD),
                ("Rate", ctypes.c_long), ("EstimatedTime", wintypes.DWORD),
                ("DefaultAlert1", wintypes.DWORD), ("DefaultAlert2", wintypes.DWORD)]


def battery_watts() -> float | None:
    """Current battery discharge in W (None on AC power: then nothing can be measured)."""
    s = _BatteryState()
    ctypes.windll.powrprof.CallNtPowerInformation(5, None, 0, ctypes.byref(s), ctypes.sizeof(s))
    if s.AcOnLine or not s.Discharging:
        return None
    return -s.Rate / 1000 if s.Rate < 0 else s.Rate / 1000


class PowerSampler:
    """Samples battery power every 0.5 s in a thread; .stop() returns (joules, seconds, mean W)."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            w = battery_watts()
            if w is not None:
                self.samples.append((time.perf_counter(), w))
            self._stop.wait(0.5)

    def stop(self) -> tuple[float, float, float]:
        self._stop.set()
        self._t.join()
        if len(self.samples) < 2:
            return float("nan"), 0.0, float("nan")
        secs = self.samples[-1][0] - self.samples[0][0]
        mean = sum(w for _, w in self.samples) / len(self.samples)
        return mean * secs, secs, mean


def gpu_seconds(pid: int | None = None) -> float:
    """Total GPU engine busy time of a process so far (all engines), in seconds."""
    pid = pid or os.getpid()
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         rf'(Get-Counter "\GPU Engine(pid_{pid}_*)\Running Time" -ErrorAction SilentlyContinue)'
         ".CounterSamples | Measure-Object CookedValue -Sum | % Sum"],
        capture_output=True, text=True).stdout.strip()
    return float(out or 0) / 1e7  # 100 ns units
