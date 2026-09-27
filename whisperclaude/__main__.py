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
import sys
import threading
import tkinter as tk
from pathlib import Path

from whisperclaude.app import QUIT, App, State, WakeQueue
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


def single_instance() -> object | None:
    """Named mutex; returns None if WhisperClaude is already running (two hotkey listeners
    would toggle twice). Keep the returned handle alive for the process lifetime."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.CreateMutexW(None, False, "Local\\WhisperClaude")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return handle


def main() -> None:
    log_path = setup_logging()
    mutex = single_instance()
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

    hotkey = key_label(cfg.hotkey.key)
    tray = Tray(request_quit, log_path, CONFIG_PATH, hotkey)
    tray.start()

    overlay.show_message("Loading model…", seconds=None)
    root.update()
    m = cfg.model
    try:
        transcriber = Transcriber(
            model=m.name, compute_type=m.compute_type, beam_size=m.beam_size,
            language=m.language or None, cpu_threads=m.cpu_threads,
            initial_prompt=m.initial_prompt or None, batch_size=m.batch_size,
        )
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
    app = App(Recorder(), transcriber, paste, events, on_state)
    listener = start_hotkey_listener(events, parse_key(cfg.hotkey.key), cfg.hotkey.max_hold_s)
    signal.signal(signal.SIGINT, lambda *_: request_quit())  # Ctrl+C in the console
    tray.set_state(State.IDLE)
    overlay.show_message(config_error or f"Ready – tap {hotkey}", seconds=6 if config_error else 2.5)
    log.info("ready (model %s, hotkey %s)", m.name, cfg.hotkey.key)

    stopped = False

    def pump() -> None:
        nonlocal stopped
        if stopped or app.process_pending():
            return
        stopped = True
        log.info("quitting")
        events.wake = lambda: None
        listener.stop()
        tray.stop()
        root.destroy()

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
