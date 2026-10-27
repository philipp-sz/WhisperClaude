# CLAUDE.md

Local, private Wispr Flow clone for Windows: tapping Left Ctrl toggles recording, Whisper transcribes locally (iGPU via OpenVINO, CPU fallback), the text is pasted into the focused app. Runs as a tray app at login.

**Full plan, decisions and implementation order: [docs/PLAN.md](docs/PLAN.md). Read it before starting work.**

## Working rule
After each implementation step, stop and report: files changed, commands run, results, issues. Don't start the next step without an OK.

## Environment
- Native Windows (not WSL — WSL can't see global hotkeys or type into Windows apps). Shell: PowerShell.
- Python 3.14 in `.venv`. Installed editable: `pip install -e .[dev]`.
- Use the venv interpreter: `.venv\Scripts\python.exe`.
- Intel Core Ultra 7 258V (8 cores, 33.8 GB RAM), Intel Arc 140V iGPU, NPU. Default backend: OpenVINO GenAI (large-v3-turbo) on the iGPU; fallback: faster-whisper (small, CTranslate2 int8) on the CPU. See `create_transcriber()`.

## Commands
```powershell
.venv\Scripts\python.exe scripts\check_env.py     # sanity check: versions, compute types, mic
.venv\Scripts\python.exe -m whisperclaude         # run with console + log output
.venv\Scripts\pythonw.exe -m whisperclaude        # run without console (as autostart does)
.venv\Scripts\python.exe scripts\install_autostart.py [--remove]   # logon task (Task Scheduler)
.venv\Scripts\python.exe scripts\benchmark.py     # model speed; record_clip.py + compare_decoding.py for quality
.venv\Scripts\python.exe -m pytest                # fast tests (slow ones are deselected by default)
.venv\Scripts\python.exe -m pytest -m slow        # loads a real Whisper model
```

## Conventions
- Package: `whisperclaude/`, one module per concern (recorder, transcriber, inserter, hotkey, overlay, tray, config, constants).
- Type hints + short docstrings in package code; scripts can be minimal.
- Threading: tkinter on the main thread, hotkey on pynput's thread, transcription + paste on a worker thread, live transcription (finished 30 s chunks while still recording, `LiveTranscription`) on a feeder thread that the worker joins before `finish()`, so the model is never used by two threads at once; communicate via `WakeQueue` (wakes Tk via a virtual event, no polling). Never block the Tk thread for long: the hotkey thread waits on it.
- Hook callbacks (pynput) must stay trivial: Windows silently removes a low-level hook whose callback is slow. `WakeQueue.put()` therefore never waits for Tk. Don't add blocking calls to hotkey callbacks.
- Startup order (see `__main__` docstring): UI, tray, hotkey and event pump first, model load on a thread (`LOADING` state). The app is also restarted at every logon, and on this laptop closing the lid ends the Windows session, so a slow cold start is the normal case. Don't import `whisperclaude.transcriber` (heavy) on the UI path; shared light constants live in `constants.py`.
- The overlay re-asserts its topmost z-order every time the pill is shown (`_raise_topmost`): after long sleeps Windows can bury the window while the flag stays set, so it is "visible" but not on screen. Tk's `-topmost` alone isn't enough.
- Spoken formatting commands (`commands.py`, English only, trigger "command") must stay predictable: false positives on normal speech are worse than missed commands. Whole words only, no fuzzy matching, nothing but spaces between trigger and command, a determiner before the trigger blocks it. Don't put command examples into Whisper's prompt again: on short, unclear speech it copied them ("Command. Command. Command." instead of what was said); the commands work without them. After changing the parser or the prompt, rerun the unit tests and the TTS corpus (see PLAN.md); score punctuation as well as structure.
- Windows can kill all user programs at a *failed* shutdown attempt without ending the session (exit code 0x40010004, no log line, no crash report), so the logon task alone is not enough: `install_autostart.py` also registers `WhisperClaude Revive` (unlock + wake events, `--revive`). Launch flags: `--autostart` (logon task) and `--revive` (revive task) exit silently if already running; `--revive` also respects the `stopped-by-user` marker (tray Quit). Keep any new start path silent on duplicates, or every login/wake shows a message box. `lifecycle.py` holds the markers; the `running` marker's mtime is a 60 s heartbeat.
- Chunking (`transcriber._speech_groups`): Whisper picks one language per chunk, so never merge VAD segments of different languages (it translates the second one); every chunk starts with `LEAD_IN_S` of silence (else it skips first words); live transcription only decides on segments that have ended. Measure quality changes with `scripts/eval_quality.py` on the recorded corpus (`scripts/record_corpus.py`), including the joined long dictations, and check speed/GPU time too.
- Tray (`tray.py`): menu rebuilds only through `Tray.refresh()` (deferred while the menu is open, serialized; pystray would otherwise rebuild an open menu from any thread). Animated icons go through `_Animator` (cached in-memory HICONs, no ticking when idle), never through pystray's `icon.icon` setter, which writes a temp .ico file per change.
- Settings live in `config.toml` (validated by `config.py`); logs in `%LOCALAPPDATA%\WhisperClaude\log.txt`. Never log transcript text (privacy). The only place transcripts are stored is the dictation history (`history.py`, `%LOCALAPPDATA%\WhisperClaude\history.json`, last `[history] keep`, saved *before* pasting so a paste into nothing can't lose a dictation); never anywhere else, never in the repo, never uploaded (clipboard writes carry `CanUploadToCloudClipboard = 0`).
- Tests: pytest, lean. Mark model-loading tests `@pytest.mark.slow`.
- Don't commit audio: all `*.wav` are git-ignored, including `tests/data/` (slow tests use clips the user records with `scripts/record_clip.py --dir tests/data`, else they skip). Never commit recordings of the user's voice or personal details (names, user paths, places).
- OpenVINO's native threads block interpreter shutdown: the app (`__main__`) and pytest (`tests/conftest.py`) exit via `os._exit` once everything is done. Keep that when adding entry points.
- README images live in `docs/images/` (demo.gif, tray-icons.png). Capture on a plain background window, never the real desktop.
