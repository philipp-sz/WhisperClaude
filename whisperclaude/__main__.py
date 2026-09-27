"""Entry point: python -m whisperclaude (console) or pythonw -m whisperclaude (no console).

Main thread: Tk (overlay) polls the event queue. pynput threads: hotkey. Worker: transcription.
pystray: tray icon in its own thread.
"""
import ctypes
import logging
import logging.handlers
import os
import queue
import signal
import sys
import threading
import tkinter as tk
from pathlib import Path

from whisperclaude.app import QUIT, App, State
from whisperclaude.hotkey import start_hotkey_listener
from whisperclaude.inserter import paste_text
from whisperclaude.overlay import Overlay, enable_dpi_awareness
from whisperclaude.recorder import Recorder
from whisperclaude.transcriber import Transcriber
from whisperclaude.tray import Tray

log = logging.getLogger("whisperclaude")
POLL_MS = 50


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

    enable_dpi_awareness()
    root = tk.Tk()
    root.report_callback_exception = lambda *exc: log.error("error in Tk callback", exc_info=exc)
    overlay = Overlay(root)
    events: queue.Queue = queue.Queue()

    def request_quit() -> None:  # thread-safe: tray thread, signal handler
        events.put((QUIT, None))

    tray = Tray(on_quit=request_quit, log_path=log_path)
    tray.start()

    overlay.show_message("Loading model…", seconds=None)
    root.update()
    try:
        transcriber = Transcriber()
    except Exception:
        log.exception("could not load model")
        tray.stop()
        root.destroy()
        return

    def on_state(state: State, msg: str = "") -> None:
        overlay.set_state(state, msg)
        tray.set_state(state, msg)

    app = App(Recorder(), transcriber, paste_text, events, on_state)
    listener = start_hotkey_listener(events)
    signal.signal(signal.SIGINT, lambda *_: request_quit())  # Ctrl+C in the console
    tray.set_state(State.IDLE)
    overlay.show_message("Ready – tap Left Ctrl")
    log.info("ready")

    def pump() -> None:
        if app.process_pending():
            root.after(POLL_MS, pump)
            return
        log.info("quitting")
        listener.stop()
        tray.stop()
        root.destroy()

    root.after(POLL_MS, pump)
    root.mainloop()


if __name__ == "__main__":
    main()
