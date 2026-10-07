"""Marker files that tell a new instance how the previous one ended, and whether to start at all.

Windows can end the app without a trace: a shutdown attempt that fails afterwards (seen: lid or
power button at the moment the laptop goes to standby) still closes user programs, but the
Windows session survives, so the "at logon" task never fires again and the app stays dead until
the next login. A second task (unlock / wake from sleep) therefore starts the app with
"--revive". Two small files in the data folder make that safe and diagnosable:

- "running": created at start, touched about once a minute, deleted on a clean exit. A start that
  finds it with a dead pid knows the previous instance was killed, and roughly when.
- "stopped-by-user": written when the user chooses Quit in the tray. A "--revive" start then does
  nothing, so Quit stays quit until the next login (a normal start deletes it).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import psutil

RUNNING = "running"
STOPPED = "stopped-by-user"


def mark_running(folder: Path) -> None:
    (folder / RUNNING).write_text(str(os.getpid()), encoding="utf-8")


def heartbeat(folder: Path) -> None:
    """Touch the marker so its modification time shows when the app was last alive."""
    try:
        os.utime(folder / RUNNING)
    except OSError:
        mark_running(folder)


def mark_clean_exit(folder: Path) -> None:
    (folder / RUNNING).unlink(missing_ok=True)


def previous_instance_died(folder: Path, now: float | None = None) -> str | None:
    """Describe the previous instance if it ended without a clean exit, else None."""
    marker = folder / RUNNING
    try:
        pid = int(marker.read_text(encoding="utf-8").strip())
        last_seen = marker.stat().st_mtime
    except (OSError, ValueError):
        return None  # no marker (clean exit or first run) or unreadable
    if pid != os.getpid() and psutil.pid_exists(pid):
        try:
            if "python" in psutil.Process(pid).name().lower():
                return None  # still running (the mutex decides what happens next)
        except psutil.Error:
            pass
    ago = (now if now is not None else time.time()) - last_seen
    return (f"previous instance (pid {pid}) did not shut down cleanly; last seen "
            f"{time.strftime('%H:%M:%S', time.localtime(last_seen))} ({ago / 60:.0f} min ago)")


def mark_user_quit(folder: Path) -> None:
    (folder / STOPPED).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def clear_user_quit(folder: Path) -> None:
    (folder / STOPPED).unlink(missing_ok=True)


def user_quit(folder: Path) -> bool:
    return (folder / STOPPED).exists()
