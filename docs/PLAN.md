# WhisperClaude — Plan

## Context
A local, private clone of Wispr Flow for a Windows laptop (Intel Core Ultra 7 258V, 8 cores, iGPU + NPU, no dGPU).
Tap Left Ctrl → speak → tap again → the transcript is pasted into whatever text box has focus (Word, browser, Claude app).
The app starts silently at Windows login (tray icon). There's no terminal in daily use.
Why native Windows, not WSL: WSL can't see global hotkeys and can't type into Windows apps.

## Decisions (from Q&A)
| Topic | Choice |
|---|---|
| Hotkey | Tap Left Ctrl alone (< 0.5 s, no other key / mouse click / scroll in between), toggle. The laptop has no Right Ctrl. Was Ctrl+Alt+Space, but that opens Claude desktop's quick entry |
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
| Model | small, int8, beam_size=1, cpu_threads=4 | Decided by benchmark (step 1), see below |
| Decoding | BatchedInferencePipeline (VAD cuts at pauses into ≤ 30 s chunks), without_timestamps, bilingual initial_prompt (batch_size=4) | See "Decoding comparison" below |
| Audio | sounddevice + numpy (16 kHz mono) | Simple, Windows wheels |
| Hotkey | pynput Listener + own tap detection | System-wide on Windows |
| Text insertion | Clipboard + simulated Ctrl+V (pyperclip + pynput), then restore old clipboard | Reliable for umlauts/Unicode, works in every app |
| Overlay | tkinter borderless, topmost, WS_EX_NOACTIVATE (via ctypes) | No extra dependency, doesn't steal focus |
| Tray | pystray + Pillow | Menu: status, open log, quit |
| Silence handling | faster-whisper vad_filter=True | Trims silence, reduces hallucinations |

The model is loaded once at startup and kept warm (~0.7 GB RAM for small).

## Benchmark result (step 1, 2026-09-26)
10 s clips, median of 3 warm runs. Timings vary ±20 % with power state (battery = throttled).

| Config | Latency (10 s clip) | CPU | RAM | Note |
|---|---|---|---|---|
| small, 4 threads, beam 5 | 2.9 s (AC?) / 3.6–4.5 s (battery) | ~50 % | 0.7 GB | "Diktier Software", some commas missing |
| small, 8 threads, beam 5 | 3.5 s | ~93 %, pinned | 0.7 GB | slower than 4 threads (P/E-core sync overhead) |
| **small, 4 threads, beam 1** | **3.3–3.4 s (battery)** | ~49 % | 0.7 GB | **chosen**: ~10 % faster than beam 5, same quality |
| medium, 4 / 8 threads | 15.5 s / 11 s (battery) | 48 % / 90 % | 1.8 GB | no better than small on German |
| large-v3-turbo, 4 / 8 threads | 19–23 s / 13 s | 49 % / 91 % | 1.9 GB | best quality, far too slow on CPU |

## Decoding comparison (2026-09-27, scripts/compare_decoding.py)
Three real dictations with pauses: DE 63 s, EN 60 s, DE/EN mixed 33 s.

| Variant | Result |
|---|---|
| A timestamps (old default) | EN: capital letter after every pause, no punctuation. Mixed: English part translated to German |
| B without_timestamps | Loses/garbles text at the 30 s window boundary, hallucinated tail |
| C timestamps + prompt | Better, still breaks at pauses |
| D without_timestamps + prompt | Great punctuation, but still boundary errors > 30 s |
| **E batched + prompt (chosen)** | Complete, well punctuated, languages kept. Only flaw: capital letter at a chunk start if the cut is mid-sentence. Same speed as A |

Why: Whisper was trained on subtitles; timestamps split text at pauses and each "subtitle line" starts capitalized. The prompt sets the punctuation style and stops translation of mixed audio.

The < 2 s target is not reached on CPU. Best quality (turbo) needs the iGPU/NPU (OpenVINO, see Roadmap).

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
- Before pasting: wait until the user has physically released all modifiers (no fake key-ups: a synthetic Alt-up can open menus), then ~50 ms.
- Errors (no mic, empty audio) show as a short overlay message and are logged to %LOCALAPPDATA%\WhisperClaude\log.txt.

## Implementation order
Working rule: after each step, stop and report what was done: files changed, commands run, results, issues.

0. Setup — DONE. venv, `pip install -e .[dev]`, env check passed (imports OK, CT2 int8 on CPU, mic detected; 8 cores, 33.8 GB RAM).
1. DONE (→ small, beam 1, 4 threads). scripts/benchmark.py + transcriber.py: benchmark small vs large-v3-turbo on a 10 s German and a 10 s English clip. Per model measure:
   - latency (s) and real-time factor
   - peak and average CPU load (% of all cores, psutil sampling every 100 ms)
   - peak RAM of the process
   - model load time
   - Decision rule: use large-v3-turbo only if latency < 2 s for 10 s of audio AND the CPU isn't pinned near 100 % for long. Otherwise use small (battery, responsiveness). Report numbers to the user before deciding.
2. recorder.py → inserter.py → hotkey.py: minimal console version working end-to-end.
3. DONE. overlay.py + tray.py: switch to pythonw (no console). Also added: log file, single-instance mutex, offline model loading.
4. DONE. install_autostart.py, logging, config file (config.toml, tray "Open config"). Also: event-driven Tk loop instead of 50 ms polling (idle CPU 1.04 % → 0.05 % of a core); paste moved to the worker thread.
5. DONE. Tests: 31 fast (app, config, hotkey, inserter) + 3 slow (real model on TTS clips in tests/data).

## Tests (pytest, lean)
- config.py: defaults and invalid values.
- inserter.py: clipboard restored after paste (mock clipboard + keyboard).
- State machine: toggles during TRANSCRIBING are ignored.
- transcriber.py: one slow opt-in test (-m slow) on a bundled short WAV.

## Verification (end-to-end)
1. `python scripts/benchmark.py`: latency + peak RAM, pick model.
2. `python -m whisperclaude`: Left Ctrl tap in Notepad, Word, Chrome (claude.ai), Claude desktop app. German + English sentence, check text and umlauts.
3. Clipboard content from before dictation is still there afterwards.
4. install_autostart.py, reboot, hotkey works with no terminal opened.
5. pytest.

## Roadmap (later, not v1)
- Cleanup: also fix spacing when appending to existing text (dictation currently glues onto the previous word; the app can't see the text field's content).
- Cleanup: remove filler words ("ähm") with a local LLM (llama-cpp-python + Qwen2.5-1.5B/3B Q4), once the benchmark shows remaining CPU/RAM.
- Format mode: second hotkey that formats into bullets/paragraphs without changing wording (constrained prompt + diff check).
- Speed-up: OpenVINO backend on the iGPU/NPU.
- Hold-to-talk option, custom vocabulary (initial_prompt), language lock.
