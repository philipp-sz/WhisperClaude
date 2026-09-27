"""Entry point: python -m whisperclaude (console) or pythonw -m whisperclaude (no console).

Main thread: Tk (overlay) sleeps until the event queue wakes it. pynput threads: hotkey.
Worker: transcription + paste. pystray: tray icon in its own thread.
"""
import ctypes
import functools
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Callable

from whisperclaude.app import QUIT, RESTART, App, State, WakeQueue
from whisperclaude.config import CONFIG_PATH, Config, ConfigError, load_config
from whisperclaude.hotkey import key_label, parse_key, start_hotkey_listener
from whisperclaude.inserter import paste_text
from whisperclaude.overlay import Overlay, enable_dpi_awareness
from whisperclaude.recorder import Recorder
from whisperclaude.transcriber import Transcriber
from whisperclaude.tray import Tray

log = logging.getLogger("whisperclaude")
BACKUP_POLL_MS = 1000  # safety net only; normally the queue wakes Tk via a virtual event
WAKE_EVENT = "<<WhisperClaudeWake>>"
RESTARTED_FLAG = "--restarted"  # passed by Restart: wait for the old instance to exit


def setup_logging() -> Path:
    """Log to %LOCALAPPDATA%\\WhisperClaude\\log.txt, plus the console if there is one."""
    log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=1,
                                             encoding="utf-8")
    ]
    if sys.stderr is not None:  # None under pythonw
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    # Without a console, uncaught errors would vanish: send them to the log.
    sys.excepthook = lambda *exc: log.error("uncaught exception", exc_info=exc)
    threading.excepthook = lambda a: log.error(
        "uncaught exception in %s", a.thread, exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    return log_path


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def single_instance(wait_s: float = 0.0) -> int | None:
    """Named mutex; returns None if WhisperClaude is already running (two hotkey listeners
    would toggle twice). wait_s: keep trying (a restarting instance may still be exiting).
    Keep the returned handle until exit, or pass it to release_instance()."""
    deadline = time.monotonic() + wait_s
    while True:
        handle = _kernel32.CreateMutexW(None, False, "Local\\WhisperClaude")
        if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
            return handle
        _kernel32.CloseHandle(handle)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.2)


def release_instance(handle: int) -> None:
    _kernel32.CloseHandle(handle)


def spawn_new_instance() -> None:
    """Start a fresh, console-less, detached copy (used by Restart)."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(pythonw if pythonw.exists() else exe), "-m", "whisperclaude",
                      RESTARTED_FLAG], cwd=CONFIG_PATH.parent, creationflags=flags, close_fds=True)


def load_in_background(root: tk.Tk, load: Callable[[], Transcriber]) -> Transcriber:
    """Load the model on a thread while Tk keeps running, so the start animation plays.

    Re-raises the loader's exception."""
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["value"] = load()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while thread.is_alive():  # only during startup (~2 s)
        root.update()
        time.sleep(0.015)
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["value"]  # type: ignore[return-value]


def main() -> None:
    log_path = setup_logging()
    mutex = single_instance(wait_s=5 if RESTARTED_FLAG in sys.argv else 0)
    if mutex is None:
        log.warning("already running, exiting")
        return

    config_error = ""
    try:
        cfg = load_config()
    except ConfigError as e:
        log.error("config error, using defaults: %s", e)
        cfg, config_error = Config(), "Config error – using defaults (see log)"

    enable_dpi_awareness()
    root = tk.Tk()
    root.report_callback_exception = lambda *exc: log.error("error in Tk callback", exc_info=exc)
    overlay = Overlay(root)
    events = WakeQueue()

    def request_quit() -> None:  # thread-safe: tray thread, signal handler
        events.put((QUIT, None))

    def request_restart() -> None:
        events.put((RESTART, None))

    hotkey = key_label(cfg.hotkey.key)
    tray = Tray(request_quit, request_restart, log_path, CONFIG_PATH, hotkey)
    tray.start()

    overlay.show_starting("Loading model…")
    m = cfg.model
    try:
        transcriber = load_in_background(root, lambda: Transcriber(
            model=m.name, compute_type=m.compute_type, beam_size=m.beam_size,
            language=m.language or None, cpu_threads=m.cpu_threads,
            initial_prompt=m.initial_prompt or None, batch_size=m.batch_size,
            vocabulary=m.vocabulary,
        ))
    except Exception:
        log.exception("could not load model %r", m.name)
        overlay.show_message("Could not load model – see log", seconds=None)
        root.after(5000, root.destroy)
        root.mainloop()
        tray.stop()
        return

    def on_state(state: State, msg: str = "") -> None:
        overlay.set_state(state, msg)
        tray.set_state(state, msg)

    paste = functools.partial(paste_text, restore_delay=cfg.paste.restore_delay_s)
    recorder = Recorder()
    overlay.level_fn = lambda: recorder.level
    app = App(recorder, transcriber, paste, events, on_state)
    listener = start_hotkey_listener(events, parse_key(cfg.hotkey.key), cfg.hotkey.max_hold_s)
    signal.signal(signal.SIGINT, lambda *_: request_quit())  # Ctrl+C in the console
    tray.set_state(State.IDLE)
    if config_error:
        overlay.show_message(config_error, seconds=6)
    else:
        overlay.show_ready(f"Ready – tap {hotkey}")
    log.info("ready (model %s, hotkey %s)", m.name, cfg.hotkey.key)

    stopped = False

    def pump() -> None:
        nonlocal stopped
        if stopped or app.process_pending():
            return
        stopped = True
        restart = app.exit_reason == RESTART
        log.info("restarting" if restart else "quitting")
        events.wake = lambda: None
        listener.stop()
        tray.stop()

        def finish() -> None:
            if restart:
                release_instance(mutex)  # let the new instance take over
                spawn_new_instance()
            root.destroy()

        overlay.show_goodbye("Restarting…" if restart else "WhisperClaude stopped", finish)

    def backup_poll() -> None:
        pump()
        if not stopped:
            root.after(BACKUP_POLL_MS, backup_poll)

    # Other threads put events -> generate a Tk virtual event -> pump runs on the Tk thread.
    root.bind(WAKE_EVENT, lambda e: pump())
    events.wake = lambda: root.event_generate(WAKE_EVENT, when="tail")
    root.after(BACKUP_POLL_MS, backup_poll)
    root.mainloop()


if __name__ == "__main__":
    main()
