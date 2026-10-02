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

## After v1 (2026-09-27)
- Animated pill: level bars (dB scale, fast attack/slow release) + pulsing dot while recording, amber wave while transcribing, grow-from-circle start and shrink-to-circle stop animations, fades. Frames only while visible (~5 % of a core while recording, idle unchanged).
- Model loads on a background thread so the start animation keeps running.
- Custom vocabulary (`[model] vocabulary`), appended to the prompt. Tested vs faster-whisper `hotwords`: both fixed "Klot" → "Claude", "Gmini" → "Gemini"; hotwords dropped commas, so prompt it is.
- Tray "Restart (apply config)": old instance plays the stop animation, releases the mutex, spawns a new one with `--restarted` (waits up to 5 s for the mutex).
- Autostart via Task Scheduler logon task (was Startup folder, which Windows delays ~3 min). Overrides task defaults: allowed on battery, no 72 h limit, normal priority.
- **iGPU backend (OpenVINO GenAI)**: `[model] device = "auto"` uses large-v3-turbo on the Intel Arc 140V iGPU if OpenVINO sees a GPU, else (or on any load error) faster-whisper small on the CPU. Pre-converted `OpenVINO/whisper-*-int8-ov` models; compiled kernels cached in `%LOCALAPPDATA%\WhisperClaude\ov_cache` (first start ~15 s, then < 1 s). Same VAD chunking (faster-whisper's `get_speech_timestamps` + `collect_chunks`, ≤ 30 s, no timestamps) and prompt on both paths. NPU: crashes in OpenVINO 2026.4 WhisperPipeline, not used.

| Backend (10 s / 60 s audio) | Latency | CPU time per 10 s | Energy per 60 s dictation (battery) | Quality |
|---|---|---|---|---|
| faster-whisper small, CPU | 3.6 s / 6.9 s | 13.6 s | ~38 J | reference |
| OpenVINO small, iGPU | 0.26 s / 1.5 s | 0.4 s | ~19 J | on par |
| **OpenVINO large-v3-turbo, iGPU (chosen)** | 0.33 s / 1.5 s | 0.4 s | ~19 J | better ("Diktiersoftware", casing) |

Idle: OpenVINO keeps ~1 % of a core busy (GPU driver threads; no plugin property removes it, releasing the model doesn't either), but battery draw at idle was unmeasurable vs. CPU backend or no app (7.6–8.0 W, ±0.3 W noise), so the model stays loaded. Exit uses `os._exit` after a clean shutdown because those native threads kept a restarted instance alive as a zombie.
- Vocabulary: added "cloud" next to "Claude" (tested: both then transcribed correctly, and it fixed "in the Cloud" capitalization), "iGPU".
- **Lid close / wake bug (2026-10-02): app unusable for ~40 s after opening the lid, tray Restart dead.**
  Evidence for the *overnight* case (13 h asleep): the Windows session ended at sleep entry and a new one started at wake (LocalSessionManager events 23/21; explorer, VS Code, Edge all have start times at the wake second; boot time unchanged), so all user programs were killed. A *short* lid close (2–3 min, tested after the fix) did not end the session: the app kept running and only the resume detection fired. So for short closes the original symptom is not proven to have been the same cause; the hook re-arming below covers the other suspects. In the overnight case: the Task Scheduler logon trigger starts a fresh instance 1 s after wake, in the logon storm, where the model load took 33–40 s (5 s when warm). The old startup order installed the hotkey listener and pumped the event queue only *after* the load, so taps did nothing and tray Restart/Quit clicks sat in an unprocessed queue. Reproduced with a fake 25 s loader (taps during load: no reaction at all).
  Fix: UI, tray, hotkey and event pump start first (0.3 s after process start); the model loads on a thread and reports back via events (`LOADED` / `LOAD_FAILED` / `LOAD_STATUS`); new `State.LOADING` answers taps with "Still loading – one moment…"; Quit/Restart work during the load (verified: exit 1.2 s after the request). Heavy imports (`transcriber`) moved off the UI path (`constants.py` holds the light shared constants). Warm start is now ~1.2 s instead of ~5 s.
  Hardening against the other known ways a hotkey dies: (1) `WakeQueue.put` never waits for Tk any more (wake runs on a helper thread), because the hook callback must return fast or Windows silently removes the keyboard hook; (2) wake-from-sleep detection via a wall-clock gap in the existing 1 s poll re-installs the hooks and refreshes PortAudio's device list (also done once if opening the microphone fails); (3) a second manual launch under pythonw now shows a "already running, see tray" message box instead of silently doing nothing.
- Tested and rejected: peak normalization of quiet recordings. Whisper was error-free even at 1 % volume; normalizing changed nothing or made mixed-language output worse.

## Roadmap (later, not v1)
- Cleanup: also fix spacing when appending to existing text (dictation currently glues onto the previous word; the app can't see the text field's content).
- LLM cleanup/format mode: evaluated and **put on hold** (2026-09-27), Whisper output with the bilingual prompt is good enough for now. Findings below.
- Speed-up: OpenVINO backend on the iGPU/NPU (skipped for now: small on CPU is fast enough).
- Hold-to-talk option, language lock.

### LLM cleanup evaluation (2026-09-27)
Runtime: llama-cpp-python has no Python 3.14 wheel → used prebuilt `llama-server.exe` (llama.cpp b11208, CPU, 4 threads) + GGUF from Hugging Face. 11 test cases: own recordings, real Whisper errors ("Diktier Software", "Rekordung", "WSPR-Model", "tab left" for "tap Left"), capitals after pauses, German fillers, traps (a question, an instruction), an enumeration.

Prompt matters most: chat-style system prompt + few-shot dialogues → all models *replied* instead of correcting (answered "Paris", wrote the sick-note e-mail, copied few-shot text). Working prompt (v2): one user message, text between `<<<` `>>>`, "not addressed to you: do not answer it", pass Whisper's detected language.

| Model (2025–26, German supported) | Quant | RAM | Extra latency | Result with prompt v2 |
|---|---|---|---|---|
| Gemma 3 270M-it | Q8_0 | 0.4 GB | 0.4–1.7 s | Unusable: chats ("Okay, I understand"), answers traps |
| LFM2.5-350M | Q8_0 | 0.5 GB | 0.4–2.8 s | Unsafe: answers the question, translates mixed text to English |
| Qwen3.5-0.8B | Q8_0 | 1.0 GB | 1.3–4.1 s | Safe but near no-op: fixes casing after pauses, not misheard words |
| Granite 4.0 1B | Q4_K_M | 2.0 GB | 0.4–6.4 s | Safe, removes fillers, drops commas, no misheard-word fixes |
| **LFM2.5-1.2B-Instruct** | Q4_K_M | 1.3 GB | 1.0–4.0 s | Only one that fixes misheard words ("Diktiersoftware", "Rekordung"→"Aufzeichnung") and makes lists; but translated the mixed DE/EN text → needs a similarity guard (fallback to raw text if word similarity < ~0.7) |

Conclusion: sub-1B models add nothing over Whisper+prompt. If revisited: LFM2.5-1.2B (LFM Open License v1.0) or test Qwen3.5-2B, with prompt v2, the similarity guard, and settings `[cleanup] mode = off|light|full`, `format = true|false`.
