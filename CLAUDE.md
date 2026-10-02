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
- Threading: tkinter on the main thread, hotkey on pynput's thread, transcription + paste on a worker thread; communicate via `WakeQueue` (wakes Tk via a virtual event, no polling). Never block the Tk thread for long: the hotkey thread waits on it.
- Hook callbacks (pynput) must stay trivial: Windows silently removes a low-level hook whose callback is slow. `WakeQueue.put()` therefore never waits for Tk. Don't add blocking calls to hotkey callbacks.
- Startup order (see `__main__` docstring): UI, tray, hotkey and event pump first, model load on a thread (`LOADING` state). The app is also restarted at every logon, and on this laptop closing the lid ends the Windows session, so a slow cold start is the normal case. Don't import `whisperclaude.transcriber` (heavy) on the UI path; shared light constants live in `constants.py`.
- Settings live in `config.toml` (validated by `config.py`); logs in `%LOCALAPPDATA%\WhisperClaude\log.txt`. Never log transcript text (privacy).
- Tests: pytest, lean. Mark model-loading tests `@pytest.mark.slow`.
- Don't commit audio: all `*.wav` are git-ignored, including `tests/data/` (slow tests use clips the user records with `scripts/record_clip.py --dir tests/data`, else they skip). Never commit recordings of the user's voice or personal details (names, user paths, places).
- OpenVINO's native threads block interpreter shutdown: the app (`__main__`) and pytest (`tests/conftest.py`) exit via `os._exit` once everything is done. Keep that when adding entry points.
- README images live in `docs/images/` (demo.gif, tray-icons.png). Capture on a plain background window, never the real desktop.
