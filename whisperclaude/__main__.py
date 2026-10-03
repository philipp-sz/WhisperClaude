"""Entry point: python -m whisperclaude (console) or pythonw -m whisperclaude (no console).

Startup order matters. After a lid close Windows ends the user session and starts us again at
the next logon, in the middle of the logon storm, where loading the model can take 30+ s. So the
UI, tray, hotkey and event loop come up first; the model loads on a background thread and reports
back through the event queue. Taps and tray clicks work (and say "still loading") from the
first second.

Main thread: Tk (overlay) sleeps until the event queue wakes it. pynput threads: hotkey.
Worker: transcription + paste. pystray: tray icon in its own thread. Model loader: short-lived.
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

import psutil

from whisperclaude.app import QUIT, RESTART, App, State, WakeQueue
from whisperclaude.commands import apply_commands, prompt_examples
from whisperclaude.config import (CONFIG_PATH, CommandsConfig, Config, ConfigError, ModelConfig,
                                  load_config)
from whisperclaude.hotkey import GapDetector, key_label, parse_key, start_hotkey_listener
from whisperclaude.inserter import paste_text
from whisperclaude.overlay import Overlay, enable_dpi_awareness
from whisperclaude.recorder import Recorder
from whisperclaude.tray import Tray
# whisperclaude.transcriber is imported by the loader thread: it pulls in the heavy Whisper
# stack (ctranslate2, onnxruntime, OpenVINO), which must not delay the UI.

log = logging.getLogger("whisperclaude")
BACKUP_POLL_MS = 1000  # safety net + resume detection; normally the queue wakes Tk itself
WAKE_EVENT = "<<WhisperClaudeWake>>"
RESTARTED_FLAG = "--restarted"  # passed by Restart: wait for the old instance to exit

# Events from the model-loader thread (handled via App.handlers)
LOADED = "loaded"            # payload: transcriber
LOAD_FAILED = "load_failed"  # payload: error text
LOAD_STATUS = "load_status"  # payload: text for the start pill ("Preparing GPU…")


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


def start_model_loading(events: WakeQueue, m: ModelConfig, c: CommandsConfig) -> None:
    """Load the model on a thread; the outcome arrives as LOADED / LOAD_FAILED events."""
    prompt = m.initial_prompt
    if c.enabled and prompt:  # priming: stops Whisper dropping/gluing "command bullet" mid-sentence
        prompt += prompt_examples(c.trigger)

    def run() -> None:
        try:
            from whisperclaude.transcriber import create_transcriber

            t0 = time.perf_counter()
            transcriber = create_transcriber(
                device=m.device, gpu_model=m.gpu_model, cpu_model=m.cpu_model,
                on_status=lambda text: events.put((LOAD_STATUS, text)),
                language=m.language or None, initial_prompt=prompt or None,
                vocabulary=m.vocabulary, compute_type=m.compute_type, beam_size=m.beam_size,
                cpu_threads=m.cpu_threads, batch_size=m.batch_size,
            )
            log.info("model loaded in %.1f s", time.perf_counter() - t0)
            events.put((LOADED, transcriber))
        except Exception as e:
            log.exception("could not load a model")
            events.put((LOAD_FAILED, str(e)))

    threading.Thread(target=run, daemon=True, name="model-loader").start()


def session_id() -> int:
    sid = ctypes.c_ulong(0)
    _kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sid))
    return sid.value


def tell_already_running() -> None:
    """A second launch has nothing to show. Without a console (pythonw) that looks like
    "nothing happens", so say so."""
    log.warning("already running, exiting")
    if sys.stderr is None and RESTARTED_FLAG not in sys.argv:
        ctypes.windll.user32.MessageBoxW(
            0, "WhisperClaude is already running.\n\nLook for the microphone icon in the tray "
               "(under the ^ arrow next to the clock). Right-click it for Restart / Quit.",
            "WhisperClaude", 0x40 | 0x40000)  # info icon, topmost


def main() -> None:
    log_path = setup_logging()
    mutex = single_instance(wait_s=5 if RESTARTED_FLAG in sys.argv else 0)
    if mutex is None:
        tell_already_running()
        return
    log.info("starting (pid %d, windows session %d, %.1f s after process start)",
             os.getpid(), session_id(), time.time() - psutil.Process().create_time())

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

    signal.signal(signal.SIGINT, lambda *_: request_quit())  # Ctrl+C in the console
    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break; also an external "quit" trigger for tests
        signal.signal(signal.SIGBREAK, lambda *_: request_quit())

    hotkey = key_label(cfg.hotkey.key)
    tray = Tray(request_quit, request_restart, log_path, CONFIG_PATH, hotkey)
    tray.start()

    def on_state(state: State, msg: str = "") -> None:
        overlay.set_state(state, msg)
        tray.set_state(state, msg)

    paste = functools.partial(paste_text, restore_delay=cfg.paste.restore_delay_s)
    recorder = Recorder()
    overlay.level_fn = lambda: recorder.level
    cmd = cfg.commands
    postprocess = ((lambda text: apply_commands(text, cmd.trigger, cmd.bullet))
                   if cmd.enabled else None)
    app = App(recorder, None, paste, events, on_state, postprocess)  # starts in State.LOADING
    listener = start_hotkey_listener(events, parse_key(cfg.hotkey.key), cfg.hotkey.max_hold_s)
    overlay.show_starting("Loading model…")
    log.info("UI, tray and hotkey up %.1f s after process start; loading the model",
             time.time() - psutil.Process().create_time())

    def on_loaded(transcriber) -> None:
        app.set_transcriber(transcriber)
        tray.engine = "iGPU" if transcriber.device == "GPU" else "CPU"
        tray.set_state(State.IDLE)
        if config_error:
            overlay.show_message(config_error, seconds=6)
        else:
            overlay.show_ready(f"Ready – tap {hotkey}")
        log.info("ready (%s on %s, hotkey %s)", type(transcriber).__name__, transcriber.device,
                 cfg.hotkey.key)

    def on_load_failed(message) -> None:
        overlay.show_message("Could not load model – see log", seconds=None)
        root.after(5000, request_quit)

    app.handlers = {
        LOADED: on_loaded,
        LOAD_FAILED: on_load_failed,
        LOAD_STATUS: lambda text: overlay.set_starting_text(str(text)),
    }

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

    gap = GapDetector()
    last_poll = [time.monotonic()]

    def backup_poll() -> None:
        now = time.monotonic()
        stall, last_poll[0] = now - last_poll[0] - BACKUP_POLL_MS / 1000, now
        if 2.0 < stall < GapDetector().threshold_s:
            # The UI thread didn't get to run for a while (e.g. cold-disk DLL loading holds the
            # GIL during the model load at logon): the pill can't animate in that time.
            log.warning("UI thread was blocked for %.1f s", stall)
        slept = gap.check()
        if slept:
            # Waking from sleep (session kept alive): Windows may have dropped our keyboard
            # hook and PortAudio's device list may be stale. Both are cheap to renew.
            log.info("resume detected (%.0f s gap): re-arming hotkey, refreshing audio", slept)
            listener.restart()
            recorder.refresh_devices()
        pump()
        if not stopped:
            root.after(BACKUP_POLL_MS, backup_poll)

    # Other threads put events -> generate a Tk virtual event -> pump runs on the Tk thread.
    root.bind(WAKE_EVENT, lambda e: pump())
    events.wake = lambda: root.event_generate(WAKE_EVENT, when="tail")
    root.after(BACKUP_POLL_MS, backup_poll)
    start_model_loading(events, cfg.model, cfg.commands)
    root.mainloop()


if __name__ == "__main__":
    main()
    # OpenVINO's native worker threads can keep the process alive after the window is gone
    # (seen after Restart: a zombie instance holding ~1.2 GB). Everything is shut down at
    # this point, so flush the log and exit hard.
    logging.shutdown()
    os._exit(0)
