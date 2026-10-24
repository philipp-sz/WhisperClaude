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
| Text insertion | Clipboard + simulated Ctrl+V (Win32 via ctypes + pynput), then restore the old clipboard with all its formats | Reliable for umlauts/Unicode, works in every app |
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
- **Invisible pill after 14 h asleep (2026-10-03).** Session survived (resume detected, hotkey and recording worked, log fine), but no pill. Live inspection of the process: the pill window existed, `IsWindowVisible`, topmost flag set, alpha 235 while recording, correct size/position, yet a screenshot showed nothing. Repair experiments on the live window: redraw did nothing, `SetWindowPos(HWND_TOPMOST)` brought it back immediately. So Windows had moved the window below others while it kept the topmost flag; Tk's `-topmost` was only applied once at startup. Fix: `Overlay._raise_topmost()` (SetWindowPos HWND_TOPMOST, NOACTIVATE) on every show (`_fade` with target > 0). Reproduced by dropping the window to the normal z-order under a covering window: invisible before, visible after; foreground window unchanged.
- **Spoken formatting commands (2026-10-03).** `command new line | new paragraph | bullet | separator | open quote | close quote` -> `\n`, `\n\n`, `\n- `, `\n\n---\n\n`, `"`, `"` (Markdown style). Pure text rewriting of the finished transcript (`commands.apply_commands`, called by `App` before pasting), no LLM, microseconds. The LLM route was rejected (too big, and unwanted reformatting).
  - *Why trigger + command, English only:* a first version without trigger (command only when it is its own sentence) found most commands but depended on Whisper's punctuation and gave false positives on ordinary sentences, so a trigger word was added. German command words work for "neue Zeile"/"neuer Absatz" but long compounds ("Aufzählungspunkt", "Anführungszeichen") are mangled, so English commands everywhere, also inside German speech.
  - *Recognition needs priming:* the iGPU model drops or glues a short English phrase inside German speech (not primed: 54/66 correct, every German "new paragraph" and quote lost). The examples `Command new line. Command new paragraph. ...` are appended to the prompt (`prompt_examples`): 66/66. They never leaked into normal speech, silence, noise or a tone.
  - *Predictability rules (each one found by a failing test):* whole words only, no fuzzy matching; only spaces between trigger and command (not "." and not ",": "use this command. New line characters..." and "every command, bullet and quote..." are normal sentences, and a missed command stays visible as text, which beats a silent false hit); a determiner/possessive directly before the trigger ("the/each/your command bullet...") means it's a noun, not a trigger (this fixed "Each command bullet point and quote is explained", which Whisper wrote without a comma); a bullet item ends at the end of its sentence (or at the next command) and text after it starts after a blank line, which is also what ends a list in Markdown (first version: items ran until the next command, so you had to say "new paragraph" to leave a list; tested bullets-then-text on 15 new utterances, 15/15); no blank line is forced before the first bullet; punctuation Whisper adds around a command is swallowed ("Command new line." etc.); pasting converts LF to CRLF.
  - *Test corpus (synthetic Windows voices Zira/Hedda; the real check is the user's voice), all rerun after the last prompt/parser change:* corpus 1 = 66 command utterances (EN, DE with English commands; 3 speeds; pauses vs flow) + 84 normal sentences; held-out corpus 2, written afterwards = 72 + 66 with new wording and many traps (bullet journal, command line, command prompt, "open quote character", ...); corpus 3 = 15 bullets followed by normal text; 24 separator utterances; long dictations (79 s EN, 85 s DE, 3 chunks each) with all commands. Production (turbo/iGPU, primed): **177/177 command utterances with correct structure and clean punctuation** (6 flagged only because Whisper wrote "Teil 1" for "Teil eins"), **normal speech untouched 149/150**; the one change: "Each command, bullet point and quote is explained in the manual." where Whisper dropped the leading "Each" once, leaving "Command bullet point and quote..." at the start of the text, indistinguishable from a real command. Long dictations correct across chunk borders. CPU fallback (small): 170/177; the 7 misses are German-context phrases it dropped entirely (unformatted text, never wrong formatting). Contrived sentences that literally contain trigger + command without a preceding determiner ("say command new line") can't be told apart; that is inherent.
  - *Corpus blind spot found late:* scoring ignored punctuation, so `Anna said. ". We should...` (Whisper put a period after "open quote") passed; found in the long dictation. Fixed (`_AFTER_OPEN`, doubled period after close quote) and the scoring now checks cleanliness.
  - Separator word: "separator", "divider" and "horizontal rule" were all recognized 24/24 by the iGPU model; on the CPU fallback 20/24, 16/24 and 24/24. Kept "separator" (the user's own word; the CPU path is the rare one).
  - Ideas not done: `command heading` (user: not now; would be a line starter like bullet with `## `; levels are fragile since Whisper hears "to" for "two"), capitalize the first word after a line break, automatic paragraph breaks at long pauses (pauses are already known from the VAD).
- **App dead after waking from standby (2026-10-07, new failure mode).** No tray icon, hotkey dead, nothing in the log after 09:55. Evidence: no `pythonw` process; scheduled task `LastTaskResult = 0x40010004` (DBG_TERMINATE_PROCESS = terminated by a shutdown/logoff event); at 10:00:31 `User32 1073` (a shutdown/restart attempt *failed*) in the same second the laptop entered modern standby; VS Code, the Claude app and Edge all have start times after the 16:03 wake, while explorer/OneDrive from the 09:43 logon survived; no crash report, no "resume detected" line at the 10:49 and 16:03 wakes. So Windows closed the user's programs for a shutdown that then did not happen, the session survived (no logoff event, no new logon), and the "at logon" task, our only starter, never fired again. (Earlier nights the session really ended and the logon task restarted the app, which hid this.) What triggered the shutdown attempt (lid or power button) was not determined.
  Fix: second task `WhisperClaude Revive` (`install_autostart.py` now registers both from XML, no admin needed): triggers = session unlock, and the wake event (System log: Kernel-Power 507 modern standby, 107 sleep, Power-Troubleshooter 1; the XPath was checked against today's real wake events; 3 s delay). It runs `--revive`; the logon task runs `--autostart`. Both exit silently if an instance is already running (no message box; only a manual second launch shows one). `lifecycle.py`: marker `stopped-by-user` (written on tray Quit/Ctrl+C; `--revive` then does nothing, so Quit stays quit until the next login or any normal start) and marker `running` (touched every 60 s, removed on a clean exit; a start that finds it with a dead pid logs `previous instance (pid N) did not shut down cleanly; last seen HH:MM:SS`, which makes the next occurrence self-explaining). Not verifiable by me: a real failed shutdown. Verified: the scheduler path with a hard kill (back in 3.8 s), duplicates, Quit/revive/login sequences in isolated fake-model runs (13 checks), 161 unit tests.
- **Live transcription + warm-up (2026-10-21).** The log showed 2–5 min dictations are common, and they took 4–9 s after the stop tap (~1 s per 30 s chunk, plus ~1 s VAD over the whole recording). Now a feeder thread hands the audio so far to `LiveTranscription.feed()` every 5 s while recording; a chunk is transcribed as soon as the VAD has started a later chunk after it (more audio can't change it then) and it ends > 5 s before the newest audio. The stop tap only leaves the last chunk or two. Same chunking as before (own greedy grouping of the VAD segments, because `collect_chunks`' metadata drops the first segment of every chunk after the first). Measured on a 313 s clip (bench_audio long clips ×2): iGPU 8–13 s → 0.9 s after the stop, text identical except 1 comma / 1 compound / 1 capital at chunk borders; CPU fallback 28 s → 6.6 s, and the old one-pass batched path had dropped two whole sentences that the live path keeps. Cost while recording: the same decode work, done earlier (VAD only on the not-yet-transcribed part). A failed feed is retried at the stop. Also: the first dictation after a start was ~0.8 s slower (1.53 vs ~0.6 s for a 10 s clip); the loader now runs the VAD and one `generate()` on 1 s of silence (first call then ~0.75 s).
- Bugs fixed in the same pass: a microphone that vanished mid-recording (headset unplugged) made `Recorder.stop()` raise, leaving the app stuck in RECORDING (every tap raised again); now the stream is closed anyway and the recorded audio is still transcribed. A failed clipboard restore after a successful paste was shown as "paste failed".
- **Clipboard, text fixes, auto-stop, Esc (2026-10-21, after the user's first real use).**
  - *Clipboard kept the dictation* (user report): pyperclip only reads text, so an empty clipboard, an image, copied files or other non-text content read as "" and the restore was skipped. Plain text was always restored (verified on the real clipboard). Now `inserter.py` saves and restores every memory-based format through Win32 (ctypes; GDI-handle formats are skipped, Windows re-synthesizes CF_BITMAP from the DIB), and restores an empty clipboard as empty. Verified on the real clipboard: plain text, empty, DIB image, copied file (CF_HDROP), HTML + text, a custom format. The dictation is marked `CanIncludeInClipboardHistory = 0` / `CanUploadToCloudClipboard = 0` (stays out of Win+V history and cloud sync); if the clipboard no longer holds the dictation at restore time (you copied something), it isn't overwritten. pyperclip removed.
  - *Linking words at chunk starts* (`textfix.py`): if the previous chunk doesn't end a sentence and the next starts with a linking word (und, um, oder, and, but, ...), it is lowercased. Tested idea rejected first: giving each chunk the previous chunk's text as prompt fixed "Und" -> "und" too, but at both German->English borders of the test clip it "translated" the English chunk into German nonsense, and it lowercased "Um" after a period. Also from the user's first real dictation: Whisper wrote the pronoun as "i" twice -> standalone i / i'm / i've / i'll / i'd become "I" (not i.e., i-Tüpfelchen, iPhone).
  - *Forgotten recording*: the log showed a 249 s recording with almost no speech. The live feeder also runs the VAD on the newest ~6 s every 5 s (cheap); no speech for `[recording] stop_after_silence_s` (60 s) stops it. The text is *copied* (and kept in Win+V history), not pasted, because a minute later the user may be typing elsewhere; the pill says so. VAD check on real audio: speech found in 12/12 windows (also at 1/10 volume); silence, white noise (-40/-26 dB), fan-like rumble and keyboard clicks: no speech. End to end (5x playback): stopped, nothing pasted, text in the clipboard.
  - *Esc cancels a recording* (`[hotkey] cancel_key`, "" = off); Esc outside a recording is ignored. The key isn't swallowed, so the focused app also sees it.
  - Measured, not changed: opening the microphone takes 130–430 ms (PortAudio MME; `start()` on an opened stream 1–2 ms), so the first ~0.2–0.4 s are lost if you speak right at the tap. Keeping the stream opened-but-stopped between dictations would remove that, but whether Windows' mic indicator then stays on couldn't be checked from here (the consent-store registry times never changed for this process).
- **Transcription quality on the user's own voice (2026-10-24).** User reports: "reinigen" for "einigen", an invented "command new page" in German speech, German that suddenly continued as translated English gibberish. Corpus: `scripts/record_corpus.py` (13 read texts, ~25 s each: German with the einigen/gereinigt trap, German full of anglicisms, English messages/e-mails/prompts, mixed, spoken commands; recordings git-ignored), `scripts/eval_quality.py` (WER on words, case errors, invented "command", time per 10 s, CPU and GPU seconds per audio minute via the per-process GPU counter, RAM, idle load). Besides the clips: five joined long dictations (all texts in random orders, 2–3.5 s pauses), each text with an extra 2.5 s pause at a random point mid-sentence (where real dictation gets cut).
  | Tested | Result |
  |---|---|
  | turbo int8 (default) | short clips 3.6 % WER; errors are real mishearings ("Anbei sende" -> "anweisende", "waiting" -> "watching", "Pfandflaschen" -> "Pfannenflaschen", "He" -> "She") |
  | turbo fp16 | same WER, +0.8 GB RAM, 28 s first compile -> no |
  | large-v3 int8 (full decoder) | same WER (fixes some errors, makes others), 4x slower, 4.6x GPU time, 3 GB RAM -> no |
  | beam search | not implemented on the iGPU in OpenVINO GenAI 2026.4 ("Not Implemented") |
  | high-pass 80 Hz, loudness normalization | no change |
  | prompt without the command examples / no prompt | 3.8 % / 4.0 % (slightly worse) |
  - *Root cause of "German turns into English"*: Whisper picks one language per chunk, from its start. With VAD segments merged blindly into 30 s chunks, a German passage that shared a chunk with a preceding English one was translated (runs of 12–48 wrong words in 3 of 5 long test dictations). One chunk per segment fixed that but made Whisper invent words in short half-sentence chunks ("dass ich Command reparieren", "Command new paragraph" copied from the prompt: the user's "command new page"). Fix: consecutive segments are merged only if their language matches (one detection pass per segment, `max_new_tokens = 1`, 0.26 s GPU; skipped for single-segment dictations and when `language` is set).
  - *Words lost at chunk starts*: a chunk that starts right on a word made Whisper skip its first words, chaotically (a 0.1 s shift changes the result). 130 random mid-sentence cuts: 13 % lost 3+ words; with 0.3–0.5 s of silence before the chunk 5 % (`LEAD_IN_S = 0.5`, free since Whisper pads to 30 s anyway).
  - Long dictations with mid-sentence pauses, WER: old 10.0 % -> **4.0 %**, no translated passages, no invented commands; short clips 3.6 -> 3.4 %. GPU time 1.5 -> 2.3 s per audio minute; short dictations unchanged in speed (0.38 s per 10 s).
  - Live transcription only groups segments that have ended (>= 5 s before the newest audio): detecting the language of a passage from its first second while it was still spoken gave wrong answers, split German passages and lost words again. Cost: a 5-minute dictation is pasted ~1.3–2.4 s after the stop tap (was ~1 s), short ones still ~0.5 s.
  - Energy (battery, continuous transcription vs. idle with the model loaded): ~13 J extra per minute of dictation before this change, ~20 J estimated after (+54 % GPU time): an hour of dictation ~0.5 % of the 71 Wh battery. Idle: 0.3–0.6 % of one CPU core, 0 % GPU.
  - Not fixed: the CPU fallback keeps blind merging (detection on the CPU is expensive; the path is only used without a GPU).
- **Command examples removed from the prompt (2026-10-24).** User report, with the new version: "...abarbeiten. Command. Command. Command. Command. Command. Command. Command. Also lass dir..." where "und schau dir..." (trailing off) was said. "Command" can only come from the prompt priming (`Command new line. Command new paragraph. ...`). Reproduced with the corpus: a short (0.4–1.2 s) quiet piece of speech or a breath-like noise alone between two 2.5 s pauses, 66 dictations: with the examples 3 invented ("Command tickets", "Command number", "Command on"), without 0; reversed-speech mumbling glued to speech: 0 either way; no other prompt word (vocabulary, example sentence) ever leaked. Without the examples: normal speech equal or better (joined long dictations 84 vs 92 word errors), both recorded command texts still give the right structure (bullets, paragraphs, quotes); only casing after a command got worse ("und bitte" after a new paragraph). The earlier priming result (54/66 -> 66/66) was on synthetic TTS voices. Also tried: attaching segments < 1.5 s to the previous chunk without a language check (stress WER 10.3 -> 9.7 %), not done, since it would mix languages again. Commands are now **off by default** (`[commands] enabled = false`): the user doesn't use them, and opt-in keeps "command ..." in normal speech from ever being reformatted.
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
