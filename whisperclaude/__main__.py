"""Entry point: python -m whisperclaude (console) or pythonw -m whisperclaude (no console).

Startup order matters. After a lid close Windows ends the user session and starts us again at
the next logon, in the middle of the logon storm, where loading the model can take 30+ s. So the
UI, tray, hotkey and event loop come up first; the model loads on a background thread and reports
back through the event queue. Taps and tray clicks work (and say "still loading") from the
first second.

Main thread: Tk (overlay, language picker) sleeps until the event queue wakes it. pynput
threads: hotkeys. Worker: transcription, translation + paste. pystray: tray icon in its own
thread. Model loader: short-lived, loads Whisper and then the translation model.
"""
from __future__ import annotations

import sys
from whisperclaude.early import quiet_duplicate
if __name__ == "__main__" and quiet_duplicate(sys.argv):
    sys.exit(0) # already running: exit before the heavy imports (no log line, no threads yet)

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
from typing import TYPE_CHECKING

import psutil

from whisperclaude.app import QUIT, RELOADED, RESTART, App, State, WakeQueue
from whisperclaude.commands import apply_commands
from whisperclaude.config import (CONFIG_PATH, Config, ConfigError, ModelConfig,
                                  TranslationConfig, load_config)
from whisperclaude import lifecycle, recovery
from whisperclaude.hotkey import GapDetector, key_label, parse_key, start_hotkey_listener
from whisperclaude.history import History
from whisperclaude.inserter import ClipboardError, copy_text, paste_text
from whisperclaude.languages import LANGUAGES
from whisperclaude.overlay import KEPT_S, Overlay, enable_dpi_awareness
from whisperclaude.picker import LanguagePicker
from whisperclaude.recorder import Recorder, load_sounddevice
from whisperclaude.system import MIN_RAM_GB, gpu_present, total_ram_gb
from whisperclaude.translation import RecentLanguages, Translation
from whisperclaude.tray import Tray
# whisperclaude.transcriber and .translator are imported by the loader thread (deferred imports
# in create_from_config and _load_translator): they pull in the heavy Whisper stack
# (ctranslate2, onnxruntime, OpenVINO), which must not delay the UI.
if TYPE_CHECKING:
    from whisperclaude.transcriber import OpenVinoTranscriber, Transcriber

log = logging.getLogger("whisperclaude")
BACKUP_POLL_MS = 1000 # safety net + resume detection; normally the queue wakes Tk itself
WAKE_EVENT = "<<WhisperClaudeWake>>"
RESTARTED_FLAG = "--restarted" # passed by Restart: wait for the old instance to exit
AUTOSTART_FLAG = "--autostart" # passed by the logon task: a duplicate start exits silently
REVIVE_FLAG = "--revive" # passed by the unlock / wake-from-sleep task: like autostart, but
                         # it also does nothing after a Quit from the tray
HEARTBEAT_S = 60

# Events from the model-loader thread (handled via App.handlers)
LOADED = "loaded" # payload: transcriber
LOAD_FAILED = "load_failed" # payload: error text
LOAD_STATUS = "load_status" # payload: text for the start pill ("Preparing GPU…")
COPIED = "copied" # payload: feedback after "Copy" (pill button or tray menu)
PROBED = "probed" # payload: True if the missing GPU was found again (recovery.py)


def setup_logging() -> Path:
    """Log to %LOCALAPPDATA%\\WhisperClaude\\log.txt, plus the console if there is one."""
    log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=1,
                                             encoding="utf-8")
    ]
    if sys.stderr is not None: # None under pythonw
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
        if ctypes.get_last_error() != 183: # ERROR_ALREADY_EXISTS
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


def create_from_config(m: ModelConfig, on_status=lambda text: None):
    """The transcriber for the [model] settings (heavy import, so only on worker threads)."""
    from whisperclaude.transcriber import create_transcriber # deferred: ~180 ms + the model stack, loader thread only

    return create_transcriber(
        device=m.device, gpu_model=m.gpu_model, cpu_model=m.cpu_model, on_status=on_status,
        language=m.language or None, initial_prompt=m.initial_prompt or None,
        vocabulary=m.vocabulary, compute_type=m.compute_type, beam_size=m.beam_size,
        cpu_threads=m.cpu_threads, batch_size=m.batch_size,
    )


def _load_whisper(events: WakeQueue, m: ModelConfig, quiet: bool = False) -> bool:
    """Loader thread: Whisper. The outcome arrives as a LOADED / LOAD_FAILED event; `quiet`
    (a reload in the background) shows no status texts."""
    try:
        t0 = time.perf_counter()
        status = (lambda text: None) if quiet else (lambda text: events.put((LOAD_STATUS, text)))
        transcriber = create_from_config(m, status)
        log.info("model loaded in %.1f s", time.perf_counter() - t0)
        events.put((LOADED, transcriber))
        return True
    except Exception as e:
        log.exception("could not load a model")
        if not quiet:
            events.put((LOAD_FAILED, str(e)))
        return False


def _load_translator(t: TranslationConfig, translation: Translation) -> None:
    """Loader thread: the translation model, handed to `translation` directly."""
    try:
        from whisperclaude.translator import Translator # deferred: ctranslate2, tokenizers, py3langid; loader thread only

        t0 = time.perf_counter()
        translator = Translator(t.model, t.device, t.cpu_threads, languages=t.my_languages,
                                unload_after_s=t.unload_after_min * 60)
        log.info("translation %s on %s ready in %.1f s (%s); own languages %s (%s)",
                 translator.model_name, translator.device, time.perf_counter() - t0,
                 "model loads on demand" if translator.on_demand else "model loaded",
                 ", ".join(translator.detector.preferred),
                 "config" if t.my_languages else "Windows")
        translation.set_translator(translator)
    except Exception as e:
        log.exception("could not load the translation model")
        translation.set_failed(f"could not load the translation model: {e}")


def start_model_loading(events: WakeQueue, m: ModelConfig, t: TranslationConfig,
                        translation: Translation | None) -> None:
    """Load the model on a thread; the outcome arrives as LOADED / LOAD_FAILED events. Then
    the translation model, handed to `translation` directly (a dictation can already run).

    The prompt has no "Command new line. Command bullet. ..." examples any more: on short,
    unclear pieces of speech Whisper copied them ("Command. Command. Command." instead of what
    was said; reproduced 3/66 with, 0/66 without). Commands still work without them.
    """
    def run() -> None:
        load_sounddevice() # ~100 ms (PortAudio init): here, not in the first start() on the Tk thread
        if _load_whisper(events, m) and translation is not None:
            _load_translator(t, translation)

    threading.Thread(target=run, daemon=True, name="model-loader").start()


def session_id() -> int:
    sid = ctypes.c_ulong(0)
    _kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sid))
    return sid.value


def tell_already_running() -> None:
    """A second launch has nothing to show. Without a console (pythonw) that looks like
    "nothing happens", so say so."""
    log.warning("already running, exiting")
    silent = {RESTARTED_FLAG, AUTOSTART_FLAG, REVIVE_FLAG}
    if sys.stderr is None and not silent.intersection(sys.argv): # only a manual 2nd launch
        ctypes.windll.user32.MessageBoxW(
            0, "WhisperClaude is already running.\n\nLook for the microphone icon in the tray "
               "(under the ^ arrow next to the clock). Right-click it for Restart / Quit.",
            "WhisperClaude", 0x40 | 0x40000) # info icon, topmost


def claim_instance(data_dir: Path) -> int | None:
    """The single-instance mutex and the lifecycle markers. None = don't start: already
    running, or a revive after a Quit from the tray."""
    revive = REVIVE_FLAG in sys.argv
    if revive and lifecycle.user_quit(data_dir):
        log.info("revive request ignored: the app was quit from the tray")
        return None
    mutex = single_instance(wait_s=5 if RESTARTED_FLAG in sys.argv else 0)
    if mutex is None:
        tell_already_running()
        return None
    log.info("starting (pid %d, windows session %d, %.1f s after process start%s)",
             os.getpid(), session_id(), time.time() - psutil.Process().create_time(),
             ", revived after wake/unlock" if revive else "")
    died = lifecycle.previous_instance_died(data_dir)
    if died:
        log.warning("%s", died)
    lifecycle.clear_user_quit(data_dir) # a normal start (login, Restart, by hand) ends a Quit
    lifecycle.mark_running(data_dir)
    return mutex


def load_config_or_defaults() -> tuple[Config, str]:
    """config.toml, or the defaults and a message for the pill if it is invalid."""
    try:
        return load_config(), ""
    except ConfigError as e:
        log.error("config error, using defaults: %s", e)
        return Config(), "Config error, using defaults (see log)"


class Session:
    """One run of the app: the pill, tray, hotkeys and App, driven by the Tk event loop.

    The UI comes up before the models load (see the module docstring); its methods are the
    callbacks the other parts call (tray menu, App, model loader, signals).
    """

    def __init__(self, log_path: Path, mutex: int) -> None:
        self.log_path, self.data_dir, self.mutex = log_path, log_path.parent, mutex
        self.cfg, self.config_error = load_config_or_defaults()
        self.events = WakeQueue()
        self.stopped = False
        self.gap = GapDetector()
        self.watch = recovery.GpuWatch()
        self.upgrade = None # a model loaded on the returned GPU, waiting for the app to be idle
        self.last_poll = self.last_beat = time.monotonic()
        self._build_ui()
        self._build_app()
        hk = self.cfg.hotkey
        self.listener = start_hotkey_listener(
            self.events, parse_key(hk.key), hk.max_hold_s,
            parse_key(hk.cancel_key) if hk.cancel_key else None,
            parse_key(self.cfg.translation.key) if self.translation else None)
        self.overlay.show_starting("Loading model…")
        log.info("UI, tray and hotkey up %.1f s after process start; loading the model",
                 time.time() - psutil.Process().create_time())

    def _build_ui(self) -> None:
        """The Tk window with the pill, quit signals, the history and the tray icon."""
        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.report_callback_exception = (
            lambda *exc: log.error("error in Tk callback", exc_info=exc))
        self.overlay = Overlay(self.root, self.cfg.pill.style)
        signal.signal(signal.SIGINT, self._on_signal) # Ctrl+C in the console
        if hasattr(signal, "SIGBREAK"): # Ctrl+Break; also an external "quit" trigger for tests
            signal.signal(signal.SIGBREAK, self._on_signal)
        self.history = History(self.data_dir, self.cfg.history.keep)
        self.hotkey = key_label(self.cfg.hotkey.key)
        self.tray = Tray(self.request_quit, self.request_restart, self.log_path, CONFIG_PATH,
                         self.hotkey, self.history, self.copy_to_clipboard,
                         style=self.cfg.tray.style, level_fn=lambda: self.overlay.meter.unit)
        self.tray.start()

    def _build_app(self) -> None:
        """The translate feature, the recorder and the App (it starts in State.LOADING)."""
        cfg, self.translation = self.cfg, None
        if cfg.translation.enabled:
            self.translation = Translation(LanguagePicker(self.root, LANGUAGES),
                                           RecentLanguages(self.data_dir / "translate.json"),
                                           key_label(cfg.translation.key),
                                           cfg.translation.my_languages)
        paste = functools.partial(paste_text, restore_delay=cfg.paste.restore_delay_s)
        self.recorder = Recorder()
        self.overlay.level_fn = lambda: self.recorder.level
        cmd = cfg.commands
        postprocess = ((lambda text: apply_commands(text, cmd.trigger, cmd.bullet))
                       if cmd.enabled else None)
        self.app = App(self.recorder, None, paste, self.events, self.on_state, postprocess,
                       stop_after_silence_s=cfg.recording.stop_after_silence_s,
                       remember=self.remember, on_pasted=self.on_pasted, on_kept=self.on_kept,
                       translation=self.translation,
                       reload=lambda: create_from_config(self.cfg.model))
        self.app.handlers = {
            LOADED: self.on_loaded,
            RELOADED: self.on_reloaded,
            PROBED: self.on_probed,
            LOAD_FAILED: self.on_load_failed,
            LOAD_STATUS: lambda text: self.overlay.set_starting_text(str(text)),
            COPIED: self.on_copied,
        }

    def run(self) -> None:
        """Wire the event queue to Tk, start loading the models, run the Tk loop."""
        # Other threads put events -> generate a Tk virtual event -> pump runs on the Tk thread.
        self.root.bind(WAKE_EVENT, lambda e: self.pump())
        self.events.wake = lambda: self.root.event_generate(WAKE_EVENT, when="tail")
        self.root.after(BACKUP_POLL_MS, self.backup_poll)
        start_model_loading(self.events, self.cfg.model, self.cfg.translation, self.translation)
        self.root.mainloop()

    # ---------- callbacks from the tray, App, loader and signals ----------

    def request_quit(self) -> None: # thread-safe: tray thread, signal handler
        lifecycle.mark_user_quit(self.data_dir) # stays quit across wake/unlock until next login
        self.events.put((QUIT, None))

    def request_restart(self) -> None:
        self.events.put((RESTART, None))

    def _on_signal(self, signum: int, _frame: object) -> None:
        log.info("quit requested by signal %s", signal.Signals(signum).name)
        self.request_quit()

    def copy_to_clipboard(self, text: str) -> None: # thread-safe: pill (Tk) or tray thread
        try:
            copy_text(text)
            log.info("copied a dictation from the history (%d characters)", len(text))
            self.events.put((COPIED, "Copied: paste with Ctrl+V"))
        except ClipboardError:
            log.warning("copy failed", exc_info=True)
            self.events.put((COPIED, "Copy failed: clipboard busy, try again"))

    def on_copied(self, msg: object) -> None:
        if self.app.state is State.IDLE: # never replace the recording / transcribing pill
            self.overlay.show_message(str(msg))

    def remember(self, text: str, translation: str = "", target: str = "", note: str = "",
                 update: bool = False) -> None: # worker thread
        self.history.add(text, translation, target, note, update)
        self.tray.refresh()

    def on_state(self, state: State, msg: str = "") -> None:
        self.overlay.set_state(state, msg)
        self.tray.set_state(state, msg)

    def on_pasted(self, text: str) -> None:
        self.overlay.show_pasted(lambda: self.copy_to_clipboard(text))
        self.tray.flash_pasted()

    def on_kept(self, text: str, msg: str) -> None:
        self.overlay.show_pasted(lambda: self.copy_to_clipboard(text), KEPT_S, label=msg)

    def on_loaded(self, transcriber: Transcriber | OpenVinoTranscriber) -> None:
        if self.app.state != State.LOADING: # a background reload: the GPU is back
            self.upgrade = transcriber
            self._apply_upgrade()
            return
        self.app.set_transcriber(transcriber)
        self._note_device(transcriber)
        self.tray.set_state(State.IDLE)
        if self.config_error:
            self.overlay.show_message(self.config_error, seconds=6)
        else:
            self.overlay.show_ready(f"Ready: tap {self.hotkey}")
        log.info("ready (%s on %s, hotkey %s)", type(transcriber).__name__, transcriber.device,
                 self.cfg.hotkey.key)

    def _note_device(self, transcriber: Transcriber | OpenVinoTranscriber) -> None:
        """Tray label, the remembered device, and whether to watch for a missing GPU."""
        kind = recovery.device_kind(transcriber.device)
        self.tray.engine = {"cuda": "NVIDIA GPU", "igpu": "iGPU", "cpu": "CPU"}[kind]
        expected = recovery.last_device(self.data_dir)
        self.watch.loaded(kind, expected)
        if self.watch.missing:
            log.warning("running on the CPU although the last start used the %s; looking for it again",
                        self.watch.missing)
        else:
            recovery.remember_device(self.data_dir, kind)

    def _apply_upgrade(self) -> None:
        """Swaps in the model loaded on the returned GPU once nothing is running."""
        if self.upgrade is None or self.app.state != State.IDLE:
            return
        transcriber, self.upgrade = self.upgrade, None
        if recovery.device_kind(transcriber.device) == "cpu": # the GPU was seen but did not load
            log.warning("the GPU did not load; staying on the CPU model and looking again later")
            self.watch.loaded("cpu", recovery.last_device(self.data_dir))
            return
        self.app.set_transcriber(transcriber)
        self._note_device(transcriber)
        self.tray.refresh()
        log.info("switched to %s on %s", type(transcriber).__name__, transcriber.device)
        self.overlay.show_message("GPU available again", seconds=3)

    def on_reloaded(self, transcriber: object) -> None:
        """The worker loaded the model again after a failed transcription."""
        log.info("model reloaded on %s", transcriber.device)
        self._note_device(transcriber)
        self.tray.refresh()

    def _start_probe(self) -> None:
        """Looks for the missing GPU in a subprocess (worker thread); the answer comes as PROBED."""
        kind = self.watch.missing

        def run() -> None:
            self.events.put((PROBED, gpu_present(kind)))

        threading.Thread(target=run, daemon=True, name="gpu-probe").start()

    def on_probed(self, found: object) -> None:
        kind = self.watch.missing
        if self.watch.probe_done(bool(found)):
            log.warning("the %s is still missing; telling the user", kind)
            self.tray.engine = "CPU (GPU missing)"
            self.tray.refresh()
            self.overlay.show_message("GPU not found: running slower on the CPU. Restarting Windows usually helps.",
                                      seconds=8)
        if found:
            log.info("the %s is back; loading the model on it in the background", kind)
            threading.Thread(target=_load_whisper, args=(self.events, self.cfg.model, True),
                             daemon=True, name="model-reloader").start()

    def on_load_failed(self, message: object) -> None:
        self.overlay.show_message("Could not load the model (see log)", seconds=None)
        self.root.after(5000, lambda: self.events.put((QUIT, None))) # not a user Quit: may be revived

    # ---------- the event loop ----------

    def pump(self) -> None:
        """Handle queued events; on Quit / Restart shut down behind the goodbye animation."""
        if self.stopped or self.app.process_pending():
            return
        self.stopped = True
        restart = self.app.exit_reason == RESTART
        log.info("restarting" if restart else "quitting")
        self.events.wake = lambda: None
        for stop in (self.listener.stop, self.tray.stop): # a failing one must not stop the quit
            try:
                stop()
            except Exception:
                log.exception("could not stop %s", getattr(stop, "__self__", stop))
        self.overlay.show_goodbye("Restarting…" if restart else "WhisperClaude stopped",
                                  lambda: self._finish(restart), restart)

    def _finish(self, restart: bool) -> None:
        lifecycle.mark_clean_exit(self.data_dir)
        if restart:
            release_instance(self.mutex) # let the new instance take over
            spawn_new_instance()
        self.root.destroy()

    def backup_poll(self) -> None:
        """Every BACKUP_POLL_MS: stall and resume detection, the heartbeat, a pump."""
        try:
            now = time.monotonic()
            stall, self.last_poll = now - self.last_poll - BACKUP_POLL_MS / 1000, now
            if 2.0 < stall < self.gap.threshold_s:
                # The UI thread didn't get to run for a while (e.g. cold-disk DLL loading holds the
                # GIL during the model load at logon): the pill can't animate in that time.
                log.warning("UI thread was blocked for %.1f s", stall)
            slept = self.gap.check()
            if slept:
                # Waking from sleep (session kept alive): Windows may have dropped our keyboard
                # hook and PortAudio's device list may be stale. Both are cheap to renew.
                log.info("resume detected (%.0f s gap): re-arming hotkey, refreshing audio", slept)
                self.listener.restart()
                self.recorder.refresh_devices()
            if self.watch.due(resumed=bool(slept)):
                self._start_probe()
            self._apply_upgrade()
            if now - self.last_beat >= HEARTBEAT_S:
                self.last_beat = now
                lifecycle.heartbeat(self.data_dir)
            self.pump()
        except Exception: # this poll is the safety net: it must go on whatever one step did
            log.exception("backup poll failed")
        if not self.stopped:
            self.root.after(BACKUP_POLL_MS, self.backup_poll)


def lean_cpu_math(ram_gb: float | None) -> bool:
    """Below MIN_RAM_GB of RAM, CTranslate2 (NLLB, Whisper on the CPU) runs without Intel's MKL.

    With MKL, CTranslate2 commits memory it never touches: NLLB-1.3B 5.3 GB private against 1.4
    GB in use, Whisper small 2.7 against 0.6 GB. That is not RAM, but it counts against the
    commit limit (RAM + page file), which is tight on small machines. Without MKL (oneDNN) the
    commit equals the use and the output stayed the same (NLLB: 8 of 8 sentences identical), at
    2.2x the translation time and 1.2x Whisper small's (measured 2026-11-11). Big machines keep
    the speed. Must run before ctranslate2 is first imported.
    """
    lean = ram_gb is not None and ram_gb < MIN_RAM_GB
    if lean:
        os.environ.setdefault("CT2_USE_MKL", "0")
    return lean


def main() -> None:
    log_path = setup_logging()
    if lean_cpu_math(total_ram_gb()):
        log.info("less than %d GiB of RAM: CTranslate2 without MKL (less committed memory)", MIN_RAM_GB)
    mutex = claim_instance(log_path.parent)
    if mutex is not None:
        Session(log_path, mutex).run()


if __name__ == "__main__":
    main()
    # OpenVINO's native worker threads can keep the process alive after the window is gone
    # (seen after Restart: a zombie instance holding ~1.2 GB). Everything is shut down at
    # this point, so flush the log and exit hard.
    logging.shutdown()
    os._exit(0)
