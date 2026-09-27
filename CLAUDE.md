# CLAUDE.md

Local, private Wispr Flow clone for Windows: tapping Left Ctrl toggles recording, faster-whisper transcribes on CPU, the text is pasted into the focused app. Runs as a tray app at login.

**Full plan, decisions and implementation order: [docs/PLAN.md](docs/PLAN.md). Read it before starting work.**

## Working rule
After each implementation step, stop and report: files changed, commands run, results, issues. Don't start the next step without an OK.

## Environment
- Native Windows (not WSL — WSL can't see global hotkeys or type into Windows apps). Shell: PowerShell.
- Python 3.14 in `.venv`. Installed editable: `pip install -e .[dev]`.
- Use the venv interpreter: `.venv\Scripts\python.exe`.
- CPU only (Intel Core Ultra 7 258V, 8 cores, 33.8 GB RAM). CTranslate2 int8 is supported.

## Commands
```powershell
.venv\Scripts\python.exe scripts\check_env.py     # sanity check: versions, compute types, mic
.venv\Scripts\python.exe -m whisperclaude         # run with console + log output
.venv\Scripts\pythonw.exe -m whisperclaude        # run without console (as autostart does)
.venv\Scripts\python.exe scripts\install_autostart.py [--remove]   # login autostart
.venv\Scripts\python.exe scripts\benchmark.py     # model speed; record_clip.py + compare_decoding.py for quality
.venv\Scripts\python.exe -m pytest                # fast tests (slow ones are deselected by default)
.venv\Scripts\python.exe -m pytest -m slow        # loads a real Whisper model
```

## Conventions
- Package: `whisperclaude/`, one module per concern (recorder, transcriber, inserter, hotkey, overlay, tray, config).
- Type hints + short docstrings in package code; scripts can be minimal.
- Threading: tkinter on the main thread, hotkey on pynput's thread, transcription + paste on a worker thread; communicate via `WakeQueue` (wakes Tk via a virtual event, no polling). Never block the Tk thread for long: the hotkey thread waits on it.
- Settings live in `config.toml` (validated by `config.py`); logs in `%LOCALAPPDATA%\WhisperClaude\log.txt`. Never log transcript text (privacy).
- Tests: pytest, lean. Mark model-loading tests `@pytest.mark.slow`.
- Don't commit audio: `*.wav` is ignored except `tests/data/*.wav`, which are Windows text-to-speech clips. Never commit recordings of the user's voice or personal details (names, user paths, places).
