# WhisperClaude — Plan

## Context
A local, private clone of Wispr Flow for a Windows laptop (Intel Core Ultra 7 258V, 8 cores, iGPU + NPU, no dGPU).
Press Ctrl+Alt+Space → speak → press again → the transcript is pasted into whatever text box has focus (Word, browser, Claude app).
The app starts silently at Windows login (tray icon). There's no terminal in daily use.
Why native Windows, not WSL: WSL can't see global hotkeys and can't type into Windows apps.

## Decisions (from Q&A)
| Topic | Choice |
|---|---|
| Hotkey | Ctrl+Alt+Space, toggle (press to start, press to stop) |
| Language | German + English, auto-detect |
| Cleanup / formatting | Not in v1. Planned for later (see Roadmap) |
| Startup | Autostart at Windows login, tray app, no console |
| Feedback | Small floating overlay (Wispr-style pill) + tray icon |
| Location | Native Windows project folder (not WSL) |
| Python | Installed 3.14.7 (not 3.12 as first planned); all deps have Windows cp314 wheels |

## Tech stack
| Part | Library | Why |
|---|---|---|
| Speech-to-text | faster-whisper (CTranslate2), CPU, int8 | ~4× faster than openai-whisper on CPU, low RAM |
| Model | large-v3-turbo (int8, ~0.8 GB) default; small as fallback | Benchmark decides |
| Audio | sounddevice + numpy (16 kHz mono) | Simple, Windows wheels |
| Hotkey | pynput GlobalHotKeys | System-wide on Windows |
| Text insertion | Clipboard + simulated Ctrl+V (pyperclip + pynput), then restore old clipboard | Reliable for umlauts/Unicode, works in every app |
| Overlay | tkinter borderless, topmost, WS_EX_NOACTIVATE (via ctypes) | No extra dependency, doesn't steal focus |
| Tray | pystray + Pillow | Menu: status, open log, quit |
| Silence handling | faster-whisper vad_filter=True | Trims silence, reduces hallucinations |

The model is loaded once at startup and kept warm (~1–1.5 GB RAM).

## Structure
```
WhisperClaude/
  pyproject.toml            # deps, entry point
  config.toml               # hotkey, model, language, compute_type, beam_size
  whisperclaude/
    __main__.py             # wires everything; tkinter mainloop on main thread
    config.py               # load/validate config (dataclass)
    recorder.py             # start()/stop() -> np.ndarray, via sounddevice callback
    transcriber.py          # FasterWhisper wrapper: transcribe(audio) -> str
    inserter.py             # paste_text(text): clipboard save -> set -> Ctrl+V -> restore
    hotkey.py               # toggle hotkey -> events into a queue
    overlay.py              # pill: "● Recording 0:04" / "Transcribing…" / hidden
    tray.py                 # pystray icon (runs detached)
  scripts/
    check_env.py            # sanity check (added in step 0)
    benchmark.py            # time small/turbo on sample audio -> pick model
    install_autostart.py    # .lnk in shell:startup -> pythonw.exe -m whisperclaude
  tests/                    # pytest, lean
```

## Flow & threading
- Main thread: tkinter overlay mainloop, polls an event queue.
- pynput thread: hotkey puts TOGGLE into the queue.
- Worker thread: transcription, so the UI never freezes.
- States: IDLE → RECORDING → TRANSCRIBING → IDLE. Toggle during TRANSCRIBING is ignored.
- After stop: wait ~50 ms and release modifiers (Ctrl/Alt from the hotkey) before pasting.
- Errors (no mic, empty audio) show as a short overlay message and are logged to %LOCALAPPDATA%\WhisperClaude\log.txt.

## Implementation order
Working rule: after each step, stop and report what was done: files changed, commands run, results, issues.

0. Setup — DONE. venv, `pip install -e .[dev]`, env check passed (imports OK, CT2 int8 on CPU, mic detected; 8 cores, 33.8 GB RAM).
1. scripts/benchmark.py + transcriber.py: benchmark small vs large-v3-turbo on a 10 s German and a 10 s English clip. Per model measure:
   - latency (s) and real-time factor
   - peak and average CPU load (% of all cores, psutil sampling every 100 ms)
   - peak RAM of the process
   - model load time
   - Decision rule: use large-v3-turbo only if latency < 2 s for 10 s of audio AND the CPU isn't pinned near 100 % for long. Otherwise use small (battery, responsiveness). Report numbers to the user before deciding.
2. recorder.py → inserter.py → hotkey.py: minimal console version working end-to-end.
3. overlay.py + tray.py: switch to pythonw (no console).
4. install_autostart.py, logging, config file.
5. Tests.

## Tests (pytest, lean)
- config.py: defaults and invalid values.
- inserter.py: clipboard restored after paste (mock clipboard + keyboard).
- State machine: toggles during TRANSCRIBING are ignored.
- transcriber.py: one slow opt-in test (-m slow) on a bundled short WAV.

## Verification (end-to-end)
1. `python scripts/benchmark.py`: latency + peak RAM, pick model.
2. `python -m whisperclaude`: Ctrl+Alt+Space in Notepad, Word, Chrome (claude.ai), Claude desktop app. German + English sentence, check text and umlauts.
3. Clipboard content from before dictation is still there afterwards.
4. install_autostart.py, reboot, hotkey works with no terminal opened.
5. pytest.

## Roadmap (later, not v1)
- Cleanup: remove filler words ("ähm") with a local LLM (llama-cpp-python + Qwen2.5-1.5B/3B Q4), once the benchmark shows remaining CPU/RAM.
- Format mode: second hotkey that formats into bullets/paragraphs without changing wording (constrained prompt + diff check).
- Speed-up: OpenVINO backend on the iGPU/NPU.
- Hold-to-talk option, custom vocabulary (initial_prompt), language lock.
