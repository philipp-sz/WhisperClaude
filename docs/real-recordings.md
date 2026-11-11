# Evaluation on the author's own recordings (2026-11-11)

The first evaluation on free speech instead of read sentences (FLEURS) or the 13-clip corpus. Recordings, references and engine outputs live in `bench_audio/real/` (git-ignored: it is the author's voice); the tools are [compare_gui.py](../scripts/compare_gui.py) and [real_eval.py](../scripts/real_eval.py). See [model-search.md](model-search.md) for the engines and the vote.

## The data

- 24 recordings from 2026-11-09 to 2026-11-11, 32 minutes, 3,152 reference words (from 2 to 324 per recording). German, English, and two recordings that switch between them; one German recording with English words in it; two recordings of a podcast played through the laptop speakers; one with long thinking pauses and self-corrections; one with e-mail addresses, file paths, times and units; one with brand names; one with a cough and typing; one 6-second request.
- Most references were made by copying Whisper's output and correcting it, so errors that the author overlooked count against the other engines. On 2026-11-11 Claude went through every reference: 74 fixes in 19 recordings (spelling and capitalization, separable verbs such as "draufgekommen", "reingeben", grammar and word choice where the reference had copied a model's mistake, such as a wrong declension, "Lineal-" for "Linear-", a misspelled author's name), names marked `[[like this]]` (68 name words), and one missing reference filled in from the agreeing transcripts of a short request. A backup of the old references was kept until the author checked the changes. Three unclear words were confirmed by the author.
- Where a reference differs from Whisper's output, the author changed it on purpose; those places were kept (for example "noch" where both models heard "auch").

## Results

Errors are word errors (substitutions, deletions, insertions) after lower-casing and removing punctuation. "Casing" counts reference words that an engine wrote correctly but with the wrong first-letter case; "false breaks" counts sentence ends (`.`, `?`, `!`) the engine put inside a reference sentence. Whisper turbo ran with the app's settings (prompt, vocabulary) through OpenVINO on the CPU (the GPU was missing, see below; same int8 model, and the 4 recordings that the iGPU had transcribed on 2026-11-09 came out identical), Parakeet TDT 0.6B v3 on the CPU through the PyTorch bridge, Granite from the comparison window (its output does not depend on the device).

| Engine | Errors (3,152 words) | Names wrong (of 68) | Casing | False breaks |
|---|---|---|---|---|
| **Whisper large-v3-turbo**, repetition loop cut | **147 (4.66%)** | **8** | 88 | 22 |
| Whisper large-v3-turbo as it is (one loop: "dx dx dx …", 368 words) | 506 (16.0%) | 8 | | |
| Parakeet TDT 0.6B v3 | 168 (5.33%) | 14 | 86 | 29 |
| Granite Speech 5.0 470M TurboCTC | 246 (7.80%) | 16 | 111 | 34 |
| Whisper `small` (CPU fallback; the comparison window's old cache) | 298 (9.45%) | 16 | 50 | 6 |
| Vote, turbo as pivot (with the loop: the vote removes it) | **143 (4.54%)** | 11 | 87 | 20 |
| Vote, Parakeet as pivot | 157 (4.98%) | | 86 | 30 |
| Vote, Granite as pivot | 173 (5.49%) | 14 | 112 | 35 |

Forgiving split and merged compounds ("noch mal" for "nochmal") changes the totals by 5–8%, not the order. The old cache is kept as engine `whisper-old` in `bench_audio/real/hyp/`; `whisper` now holds the turbo outputs (time not recorded, shown as 0.00x).

### What the numbers say

- **Whisper turbo is the most accurate engine on the author's free speech** (4.7%) and the best on names (8 of 68 wrong, the others 14–16; the vocabulary in the prompt helps). Parakeet follows (5.3%), Granite is clearly behind (7.8%), the opposite of FLEURS, where Granite was ahead.
- **The vote does not pay off**: 143 against 147 errors, for three engines' time and a Parakeet runtime that does not exist yet. Its one real benefit, removing Whisper's repetition loop, a simple loop guard gives too. Decision proposed: do not build the Parakeet runtime for the vote.
- **Whisper's real weakness is the repetition loop**: one of 24 recordings (spoken math, "x Quadrat dx") produced "dx" 368 times. Nothing in the app guards against it.
- **The author's earlier impressions came from `small`**: since 2026-11-10 the app and the comparison window ran Whisper `small` on the CPU without saying so. `small` translated the English half of 20261110_185724 into German (116 errors; turbo: 5), wrote "als siebter Teil" for "als zip-Datei" and made twice turbo's errors overall.
- **Granite's errors are the most visible ones**: twice turbo's casing errors at the same accuracy level, sentence breaks at thinking pauses (a period inside a sentence wherever the speaker paused), German compounds in pieces ("Teil auf Gel B" for "Teilaufgabe b"), and doubled fragments no other engine makes ("textxt", "modelss", "glaubeub", "Metallddeckel", "Claudeed": 8 cases). The author's impression that Whisper is better is right.

## Error types

From the error lists of Granite, Parakeet and Whisper `small` (turbo makes fewer of each kind except the loop):

| Type | Examples | Engines | Source |
|---|---|---|---|
| Lost speech before any model ran | a short "thank you" after a cough (whole recording empty), three passages of the quiet podcast clip (25 words), "Aber am Ende" after a thinking pause | all three, identically | the VAD (below) |
| Translation of a whole chunk | the English half of 20261110_185724 written in German | Whisper `small` (turbo got it right) | language detection per chunk (below) |
| Names and products | Claude → "cloud", "Claw", "clawed"; ChatGPT → "GPD", "ChagGPT"; Grok → "GROG", "grog"; Whisper → "Whisperer"; Hugging Face → "hugging phase"; Hoare → "HUA", "Huawei", "Hoir" | all (14–16 of 68 name words) | model; Whisper's vocabulary has "Claude" and also "cloud" |
| Domain words at speed | Bijektion → "B-Ektion", "Projektion", "Bektion"; abzählbar → "absehbar" | all | fast speech; the author says the word works when spoken slowly |
| Sound-alike words | was/dass, "needed" for "need it", "said"/"set", "daily"/"tailor", "compute"/"computer" | all | ambiguous audio, context needed |
| Spoken formats | a spoken e-mail address ("name dot name at domain dot com"), a spoken Windows path, times ("1735", "17.35", Parakeet: "seventeen thirty five"), math ("x Quadrat" → "x2") | all | no inverse text normalization; Parakeet writes English numbers as words |
| Invented words | "Vielen Dank." appended at the end of a German recording, "It", "It's" at a sentence start | Whisper | Whisper's language model on trailing silence |
| Short utterance | a 6 s German request with "zip-Datei": every engine wrong ("siebter Teil", "Sippdatei"), Whisper on the NPU without prompt translated it | all | little context; short chunks are known to be weak |
| Quiet or mumbled speech on purpose | "Teil of KBB", "Telefon-A" for "Teilaufgabe a" | all | the speech itself; rare in normal use (author) |
| Disfluencies | "das das", "du du", "ich, das" dropped | all | wanted for dictation; the references keep them, so they count as errors (about 6 words) |

Microphone level: the author's own recordings are at −27 to −33 dBFS RMS. Only the podcast clip played from the speakers (−40.6 dBFS) is much quieter, and it lost the most words (33–42 errors of 147), mainly through the VAD.

## Findings outside the models

### The GPU disappeared (2026-11-10)

Since 2026-11-10 11:29 the running app uses **Whisper `small` on the CPU**: the instance before it ended uncleanly at 10:21 (after a wake), and the new one logged `no OpenVINO GPU found, using CPU`. Windows still lists the Arc GPU as working, but OpenVINO only offers CPU and NPU (`ov.Core().available_devices`, checked 2026-11-11 in and outside the tool sandbox). The app keeps that fallback until it is restarted and does not tell the user. Visible effects: about 4.5 s from the stop tap to the text for a 6 s dictation (0.01–0.05 s before), and `small`'s lower accuracy (FLEURS German 12.3% against 4.6%). The comparison window loads the app's configuration and fell back the same way: its cached Whisper outputs after 2026-11-09 are `small` (confirmed: `small` reproduces them exactly).

### The VAD drops quiet speech

Silero VAD (faster-whisper's, threshold 0.5, 2 s minimum silence, 400 ms padding) decides what reaches any engine. Speech kept by the VAD in seconds, for the affected recordings:

| Recording | Audio | Threshold 0.5 (now) | 0.35 | 0.25 | Gain to −26 dBFS before the VAD |
|---|---|---|---|---|---|
| podcast, −40.6 dBFS | 49.0 s | 37.7 | 40.2 | 41.2 | 42.3 |
| a short "thank you" after a cough | 38.0 s | 0.0 | 0.0 | 0.0 | 0.0 |
| thinking pauses | 126.8 s | 49.4 | 51.3 | 55.1 | 49.4 |
| quiet on purpose | 33.8 s | 22.8 | 25.4 | 25.8 | 21.4 |

The "thank you" never exceeds a speech probability of 0.15 and is lost with any setting.

Errors on these four recordings (325 words), turbo and Granite on the CPU, the models still get the original audio:

| VAD input | Whisper turbo | Granite |
|---|---|---|
| now (threshold 0.5) | 41 | 60 |
| threshold 0.35 | 97 (more noise reaches Whisper, which invents words) | 55 |
| **audio scaled to −26 dBFS RMS for the VAD only** | **30** | **41** |
| scaled and threshold 0.35 | 29 | 41 |

Scaling the VAD's input (not the model's) recovers about a quarter of these errors. On the other recordings it changes the kept speech by less than 1.5 s; whether it costs anything there has to be measured on the whole set before it goes into the app.

### Whisper's language per chunk

Every chunk is transcribed with automatic language detection; when it picks the wrong language, Whisper translates. `small` did that with the English half of 20261110_185724; turbo transcribed the same recording correctly (5 errors), and Whisper on the NPU without the prompt translated the 6 s German request into English. The prompt and the larger model keep it rare; Granite and Parakeet cannot translate at all.

### The NPU (2026-11-11)

The laptop's NPU (Intel AI Boost) can run both models:

- **Granite TurboCTC**: the IR has dynamic input shapes, which the NPU compiler rejects; reshaped to the app's fixed chunk shape (`[1, 1525, 320]` features, `[1, 1525]` mask) and compiled with `INFERENCE_PRECISION_HINT = f16` (the NPU rejects `f32`), it compiles in 69 s (once, cacheable), runs **250 ms per chunk** of any length up to 30.5 s (iGPU 65–120 ms), and gives **the same tokens as the CPU in 9 of 9 chunks**.
- **Whisper turbo (int8) through `openvino_genai.WhisperPipeline(path, "NPU")`**: compiles in 6.5 minutes the first time (with the CPU busy; 3 s from the cache afterwards) and transcribes at about the iGPU's speed (0.41 s for 6 s, 1.58 s for 31 s, 2.30 s for 58 s of audio). But **`initial_prompt` fails on the NPU** (`Check '*roi_end <= *max_dim' failed`, even with the prompt sentence alone, OpenVINO GenAI 2026.4.0), so the bilingual prompt and the vocabulary cannot be used, and without the prompt the 6 s German request came out as an English translation.

## Ideas and questions from the recordings

The later recordings are also the author's idea inbox. Each item: what was said, then the assessment *(Claude, 2026-11-11, not decided)*.

| # | Recording | Idea or question | Assessment |
|---|---|---|---|
| 1 | 084540 | The tool must be reliable: no invented, missing or wrong words. | The goal behind everything below. Today's biggest losses were not turbo's errors: a GPU that vanished (so `small` ran instead), a VAD that drops quiet speech, and one repetition loop. |
| 2 | 084540 | Why are newer, bigger models worse than Whisper on our tests when the leaderboards rank Whisper lower? (Also: when was Whisper published?) | The leaderboards are English read speech; our tests are German, mixed, and accented free speech. Whisper was trained on 680,000 hours (large-v3: 5 million, with pseudo-labels) of multilingual web audio, much of it German; the speech LLMs (Qwen3-ASR, Granite 2B) are English-centred and translate mixed sentences. On the author's free speech, turbo is indeed still the best of all tested engines (4.7%; Parakeet 5.3%, Granite 7.8%). Dates: Whisper was published in September 2022, large-v3 in November 2023, turbo in October 2024. |
| 3 | 085403 | Claude's dictation shows each word as you speak; ours transcribes every 30 s. A different kind of model, or only the surface? | Mostly the surface. Streaming recognizers emit partial text every few hundred milliseconds and revise it; ours already transcribes finished chunks while recording (live transcription) but shows nothing until the end, because text pasted into another app cannot be revised. Showing the finished chunks in the pill would be possible. Which model Anthropic's dictation uses is not public. |
| 4 | 184736 | Cloud dictation feels better because it is more than a transcription model: perhaps a small LLM that hears audio and knows context; that would also explain word-by-word output. | Speech LLMs of this kind exist (Qwen3-ASR, Granite 2B, Voxtral; tested here: slow on the iGPU and they translate mixed speech). Whisper's decoder is itself an autoregressive language model, so autoregression alone does not explain the word-by-word display; streaming does. Context does help: Whisper's prompt and vocabulary, Granite 2B's keywords. |
| 5 | 185231 | Granite was first faster with more errors, now slower with fewer. Why? | The speed: Granite ran at 0.006× real time in the first recordings (iGPU) and about 0.05× later, because the comparison window runs engines side by side and, since 2026-11-10, the GPU is gone. The errors: the first recordings were short re-enactments of hard cases (quiet, fast, exercise vocabulary), where Granite was weakest. Over all recordings Granite is behind Whisper turbo. |
| 6 | 185724 | Add a local LLM (for offline use, tool use via OpenClaw)? Small LLMs could not format or summarize. | A separate project, as the author suggests: it shares nothing with dictation except the microphone. In this tool an LLM only makes sense for narrow, checkable steps; two rounds of tests rejected cleanup ([transcription.md](transcription.md)). |
| 7 | 190215 | Free hand for creative or impressive ideas, and ideas for other projects. | See the backlog: the NPU path is new ground (the research found no Windows dictation app that uses an NPU). |
| 8 | 191307 | Signal processing, filtering, physical models? | Tested before: noise reduction and normalization of the model input gained at most 1 point on noisy FLEURS and nothing on the author's recordings ([transcription.md](transcription.md)). New today: the level of the *VAD input* matters (see above). |
| 9 | 191307 | No fine-tuning on the author's voice (it must work for everyone); fine-tuning on public data is pointless; too expensive. | Agreed. |
| 10 | 191307 | Must work on weak and strong hardware. | The automatic choice exists (CUDA `large-v3`, iGPU turbo, CPU `small`); today showed it needs to recover and to tell the user when it falls back. The NPU would be a fourth path. |
| 11 | 191307 | Build the tool as a harness: any speech-to-text model can be plugged in, setup scripts for a few models, easy to add new ones. | Half there: `create_transcriber(engine=...)`, the shared chunking, `install_granite.py`, and the bridge in `real_engines.py`. A second installable engine (Parakeet) would show what the shared parts should be; a registry before that would be guessing. |
| 12 | 205136 | The pill started a recording by itself, several times. | Not explained. The tap detector ignores taps with a mouse click or scroll in between but accepts Ctrl events injected by software (touchpad or touchscreen pinch gestures, pen, remote or automation tools); the log does not record recording starts. Next step: log each start with the key event's injected flag and hold time (no content), and ask what the author was doing. |
| 13 | 205136 | Correct the spelling of the references. | Done (above). |
| 14 | 210615 | Math does not work well. | The models write words ("x hoch 3"); turning spoken math into notation is formatting, not recognition. A rule-based math mode is possible but niche. |
| 15 | 170541 | Do I speak too quietly or unclearly? | No: −27 to −33 dBFS is fine, and the quiet podcast lost more. Only deliberately quiet speech (two recordings) suffered. |
| 16 | 185231 note | The usual case is English words inside German, not half and half. | Turbo and Parakeet: 0 errors on that recording, Granite 4. |
| 17 | notes | "was das ist"/"dass das ist", P1/P2/B capitals, "Bijektion" at speed. | Ambiguous audio: only context helps (vocabulary for terms; nothing for capitalized variables). |
| 18 | 174059 | Self-corrections ("A, no, I mean B"). | Kept verbatim by all engines, which is the predictable behaviour; removing the wrong half needs an LLM (research file 2.1), rejected for now. |
| 19 | 173618 | E-mail addresses, paths, times, units. | Units and version numbers worked; e-mails, paths and times did not in any engine. Deterministic rules ("at" → "@", "dot" → "." between word pieces) would fit `commands.py`'s style but need the same care against false positives. |

## Ideas from the research files (2026-11-10)

Two research notes made with Claude (an overview of open-source dictation tools, and a brainstorming list) were read as candidate ideas, not as tasks. Only the ones that touch this tool are listed; the rest (meetings, diarization, speaker separation, silent speech, lip reading, multi-output rewriting, signed releases, server APIs) does not fit a single-user dictation tool.

| Idea (source) | Relation to this project | Verdict |
|---|---|---|
| Personal benchmark set with name and number metrics (brainstorm #19) | Exists now: `compare_gui.py`, `real_eval.py`, `[[names]]` | done; add casing and false-break counts to the report |
| Silero VAD in front of everything, explicit anti-hallucination thresholds (tools §5.1, OpenWhispr) | VAD exists and is now the source of lost words; Whisper on OpenVINO has no no-speech or log-probability thresholds ("Vielen Dank." appended) | fix the VAD level problem; thresholds only if Whisper stays the engine |
| ROVER-style vote of several models (brainstorm #5, paper: Whisper + Canary + Parakeet 6.26 → 3.92%) | `vote.py`; on real speech 143 against 147 errors for turbo | not worth it here |
| Cascade: second model only when unsure (brainstorm #5) | Tested on FLEURS: confidence predicts errors too weakly | rejected (measured) |
| Context biasing inside the decoder instead of the prompt (brainstorm #1; CTC boosting in FluidAudio apps) | Names are 14–16 of 68 wrong in every engine; Whisper's prompt vocabulary already shows the prompt's side effects | later, after the engine decision (CTC and TDT both allow it) |
| Screen or repository content as the bias source (brainstorm #2, "not found" elsewhere) | Would feed the bias list automatically (window title, VS Code workspace symbols) | interesting, after decoder biasing exists |
| Microphone health check (brainstorm #14) | The quiet podcast clip; and a "running on CPU" warning is the same kind of health signal | small, with the GPU warning |
| Pin the paste target to the window where recording started (OpenWhispr) | Not done: the text goes to whatever has the focus after transcription, now 4.5 s on the CPU fallback | small, worth doing |
| Reliable paste via clipboard read notifications (Handy) | The app uses a fixed restore delay and never overwrites a clipboard it cannot restore | not needed so far |
| Auto-send (Return after paste, VoiceInk) | Useful for chat boxes and Claude Code | optional, off by default; low |
| Deterministic inverse text normalization (FluidAudio) | Parakeet writes English numbers as words; e-mails and paths fail everywhere | needed if Parakeet becomes the engine |
| Live text while speaking, stable-prefix commit (WhisperLiveKit) | Answers question 3; chunks are already transcribed while recording | pill preview of finished chunks: medium |
| Faithfulness guard after LLM cleanup (brainstorm #4) | No LLM step in the app | not applicable |
| N-best + LLM correction, speech LLM in one step, self-correction model (brainstorm #3, #9, #11) | LLM cleanup rejected twice; speech LLMs too slow here | rejected for now |
| LoRA from the user's corrections (brainstorm #7) | The author wants no personal fine-tuning | rejected |
| NPU use on Windows (tools §4.3, "not found") | Works for Granite, and for Whisper turbo without the prompt | new ground, see the backlog |
| Remote control through command-line flags (Handy) | The single-instance code already handles `--autostart`/`--revive` | low |
| Per-app profiles (VoiceInk, FluidVoice) | Nothing app-specific in the pipeline yet | low |

## Backlog (proposed, ranked; nothing decided)

*Superseded the same day: the first round's Whisper numbers were Whisper `small` (see "The GPU disappeared"). What was built is in "Third round".*

| # | Item | Why | Cost | Measure |
|---|---|---|---|---|
| 1 | **Never fall back silently, and recover the GPU**: a tray warning and log line when Whisper runs on the CPU although a GPU is expected; retry the GPU at each resume and a minute after start | The app ran `small` for a day unnoticed; twice the errors, 4.5 s latency | small | the fallback path in tests (mocked devices) |
| 2 | **Repetition guard** for Whisper: cut a chunk's output at a loop (the same word or n-gram repeated, e.g. 4+ times) | 368 invented words in 1 of 24 recordings; the only big turbo failure | small | unit tests; the math recording |
| 3 | **Scale the VAD input to a fixed level** (VAD only, the model keeps the original audio) | −27% errors on the quiet and paused recordings | small | whole set + FLEURS before/after |
| 4 | Diagnose the phantom recording start: log each start with injected flag and hold time (no content); then decide whether to ignore injected Ctrl events | Reported several times, cause unknown | small | the log after the next occurrence |
| 5 | NPU as the fallback before the CPU: Granite works today; Whisper turbo needs the language set per chunk (the NPU pipeline has no prompt) | Keeps turbo-class quality when the GPU is gone, at low power; no Windows dictation app does it yet | medium | the set on the NPU against the CPU run |
| 6 | Pin the paste target to the window where recording started (or don't paste and say so if the focus moved) | Text could land in the wrong app, worse with slow fallbacks | small | manual test |
| 7 | Vocabulary review: "cloud" is in the list next to "Claude"; add terms the author uses (Bijektion, Hoare, …) | Names are turbo's main error type besides misheard words | trivial | real_eval before/after |
| 8 | Pill preview of finished chunks while recording | Answers the "word by word" wish partly; the texts already exist | medium | — |
| 9 | Spoken e-mails and paths (`commands.py`-style rules) | Failed in every engine | medium | dedicated recordings |
| 10 | Decoder biasing for names (CTC/TDT) and screen context | Names; but turbo's prompt vocabulary already does well (8 of 68) | large | `[[names]]` |

Not proposed: the Parakeet runtime and the vote (+4 errors' worth of gain), Granite as the default engine, LLM cleanup, fine-tuning, a local LLM chat (separate project).

## Second round (2026-11-11 evening, GPU back after a reboot)

A restart of the app did not bring the GPU back; a reboot of the laptop did. All numbers below on the iGPU, 24 recordings, 3,152 words.

- **Turbo on the iGPU: 149 errors (4.7%), 9 of 68 names wrong, 0.04× real time**, no loop (the math recording came out nearly perfect, even "x² dx"). The CPU run's loop came from small numeric differences that tipped greedy decoding. The cache `whisper` now holds these iGPU outputs.
- **Loops are still possible on the iGPU**: with the VAD input scaled (different chunk borders) the math recording looped there too ("dx" 387 times).
- **Recovering a loop** (the CPU case, same chunk): beam search with 4 beams gave the correct text ("x² dx."), so did `repetition_penalty = 1.1` and sampling at temperature 0.4 (random: "bis m"); `no_repeat_ngram_size = 4` did not stop it; no prompt nearly. Proposed guard: detect a loop (a word or phrase repeated, or far more words than the chunk's seconds allow), decode only that chunk again with 4 beams; nothing is cut, and normal chunks cost nothing. The author's condition (2026-11-11): a guard must recover the text, not drop it.

| Variant (loops collapsed for the count) | Errors | Names wrong |
|---|---|---|
| Base (app settings) | 149 | 9 |
| **VAD input scaled to −26 dBFS** | **133 (−11%)** | 9 |
| Model input normalized to −26 dBFS | 137 | 9 |
| High-pass 80 Hz on the model input | 149 | 9 |
| Wiener noise reduction on the model input | 151 | 10 |
| Prompt sentence without the vocabulary | 219 | 17 |
| No prompt | 157 | 18 |
| Previous chunk's text added to the prompt (context carry-over) | 424 | 22 |
| Carry-over without the vocabulary | 298 | 18 |
| **Automatic vocabulary** (26 words from the author's own texts, see below) | **146** | **6** |
| Automatic + manual vocabulary | 143 | 7 |

- **Filters** (the author's question): high-pass and noise reduction change single recordings by a few words both ways and nothing in total, as on FLEURS before. The gain of normalizing comes from the VAD seeing a louder signal; scaling only the VAD's input gives it without touching the model input.
- **Context carry-over** is clearly harmful: Whisper copies from the prompt and drifts (the podcast clip went from 28 to 146 errors), the known reason `condition_on_previous_text` is off.
- **The vocabulary works but is tedious** (the author: there are hundreds of such words). **Automatic vocabulary**: words from the author's own texts (this repository's docs and README, the general knowledge base; today's recording doc excluded to avoid leaking the references) that appear at least 3 times, are capitalized, and that Whisper's tokenizer splits into several pieces even in lower case (a sign the word was rare in its training data); ranked by frequency × extra pieces, 26 words, 128 prompt tokens with the prompt sentence. It beat the manual list (names 6 against 9 wrong) without curation. Caveat: the texts are about the same project as many recordings. The idea generalizes to other sources (window title, the open project's files, notes, the dictation history), all local.
- **Phantom recording starts**: the author had two fingers on the touchpad. Hypothesis: a precision-touchpad pinch is delivered to many apps as Ctrl + wheel, with an injected Ctrl press and release; if the wheel part does not pass the low-level mouse hook, the tap detector sees a lone Ctrl tap. A 30 s probe script that prints Ctrl events with Windows' injected flag and wheel events was prepared (in the session's scratchpad); to be confirmed before changing the detector.
- **Flicker**: random, while recording and while transcribing (author); both are the animated states of the pill and the tray icon. Not investigated yet.
- **Models** (web search, 2026-11-11): NVIDIA Nemotron 3.5 ASR Streaming 0.6B (June 2026, 40 language-locales, OpenMDW license, a real streaming transducer like Parakeet; German not confirmed), Cohere Transcribe 2B (Apache-2.0, 14 languages incl. German; gated repository, autoregressive), German fine-tunes of turbo (e.g. primeline, Tuda-De 6.4% against 8.3%; community repositories, need the author's OK). None tested.

### Robustness requirements (author, 2026-11-11)

- The app should first try to solve problems itself (reconnect to the GPU, retry) and warn only when that fails; such problems should be rare.
- Fallback chains for both target platforms: Intel (iGPU → NPU → CPU) and NVIDIA (CUDA → CPU); the CPU fallback stays because not every laptop has an NPU. Robust and efficient. The desktop with the NVIDIA GPU will be tested extensively once the changes work on the laptop.

### Backlog after the second round (proposed)

*Status 2026-11-11 night: 1 built without the NPU tier (not usable yet), 2–5 built, 6 diagnosed (Panel Self Refresh); see "Third round".*

1. Self-healing device choice: on a missing GPU, retry at each resume and periodically (e.g. every few minutes while idle), switch back when it appears; NPU between iGPU and CPU (Granite works; Whisper on the NPU needs the language set per chunk, no prompt); a tray notice only if the fallback persists.
2. Loop guard with beam-4 retry of the affected chunk.
3. VAD input scaled to a fixed level (−11% here; check FLEURS before).
4. Automatic vocabulary from local sources, replacing the manual list as the default (the manual list stays for additions).
5. Confirm the touchpad hypothesis with the probe; then ignore injected Ctrl events (or require no wheel/gesture activity) in the tap detector.
6. Flicker investigation (overlay repaint, tray animator).

## Third round: built and measured (2026-11-11 night)

All on the iGPU, 24 recordings, 3,152 words, through the app's own code.

| Pipeline | Errors | Names wrong |
|---|---|---|
| Before (manual vocabulary, VAD as it was, no loop guard) | 149 (4.72%) | 9 / 68 |
| + VAD input raised (amplify only, at most +20 dB) + loop guard | 131 (4.15%) | 9 / 68 |
| + automatic vocabulary (25 words from the docs and notes, today's recording doc excluded; 158 prompt tokens) | **122 (3.87%)** | **6 / 68** |

The VAD gain left the 13-clip corpus unchanged (19 errors on 506 words) and white noise from −70 to −40 dBFS still produced no text.

What was built (2026-11-11, tests in brackets):

- **Loop guard** in `OpenVinoTranscriber` (`test_transcriber_guards.py`): compression ratio above 2.4 or more than 6 words/s (above 20 words) marks a loop; normal chunks of the recordings reached at most 1.88 and 3.5 words/s. The retry uses `repetition_penalty = 1.1`: beam search, which recovered the text on the CPU, fails on the iGPU ("Not Implemented"). On the iGPU the guard turned the looped math recording from 390 into 7 errors. Retry that loops too: repeats collapsed.
- **VAD input level** (`vad_input()` in `transcriber.py`).
- **Self-healing device choice** (`recovery.py`, `gpu_present()`, `Session` in `__main__`; `test_recovery.py`, `test_session_recovery.py`): see [how-it-works.md](how-it-works.md). The probe subprocess takes 0.4 s. A failed transcription loads the model again and retries the audio once (`App.reload`; `test_app.py`).
- **Automatic vocabulary** (`vocabulary.py`, `[model] vocabulary_sources`, default `["~/code", "~/projects"]`; `test_vocabulary.py`). Caps: at most 40 words, the whole prompt within 200 tokens, at most 2,000 files and 20 MB per start, only `.md`/`.txt`, tool and build folders skipped; a file's counts halve every 30 days (the last day counts fully), so the list follows current work and never grows. On this laptop: 31 files, 29 words, 0.11 s. Documents is not a default source: reading cloud-synced placeholder files would download them.
- **Injected Ctrl ignored** (`hotkey.py`; `test_resume.py`): pynput passes Windows' `LLKHF_INJECTED` flag; injected record-key events never form a tap, injected other keys still interrupt one. Each recording start is now logged (no content), so a phantom start would show in the log.
- **Comparison tool** (`scripts/real_engines.py`): refuses to run Whisper when it fell back to the CPU on a machine that last ran on a GPU; only Whisper runs in-process now, Parakeet and Granite through the PyTorch bridge; the vote is gone.

Measured, not built:

- **Whisper on the NPU**: without the prompt 595 errors (18.9%, whole recordings translated into English, once "KVcache is full"); decoding every chunk in German and English and keeping the higher score fails because the NPU pipeline returns a constant score of 1.0 (1,530 errors). Not usable until the NPU pipeline accepts a prompt.
- **Memory**: the running app showed 7.1 GB private memory. Whisper turbo on the iGPU: 1.6 GB private, 1.4 GB working set; NLLB-1.3B: 5.3 GB private but 1.5 GB working set, the same with `int8_float32` and one thread. The cause is Intel's MKL inside CTranslate2 (commit charge, not RAM); below 12 GiB of RAM the app now runs CTranslate2 without it, see [translation.md](translation.md#committed-memory-and-mkl-2026-11-11).
- **Nemotron 3.5 ASR Streaming 0.6B** (`nvidia/nemotron-3.5-asr-streaming-0.6b`, 2026-06-04, OpenMDW-1.1, safetensors and transformers support, cache-aware FastConformer-RNNT, 40 locales incl. de-DE, punctuation and casing): NVIDIA's model card gives 8.31% WER on German FLEURS (test, 1.12 s chunks) against 4.58% for turbo here (dev split). Not tested: built for low-latency streaming at scale, not accuracy; Parakeet of the same family already lost to turbo. `nemotron-labs-unified-asr-0.6b` (2026-10-27) is only a `.nemo` file (pickle, NeMo); the 30B "Transcribe" models are far too large.
- **Flicker** (pill and tray icon, random, only while animating, gone when the mouse moves over it): the tray icon is drawn by Explorer, so a shared cause in the display path is likely, matching Intel's Panel Self Refresh (the panel shows a stored frame while the screen is "static"; small animated areas can show stale pixels until a refresh, e.g. a mouse move). Test: turn off Panel Self Refresh in the Intel Graphics Software (Display / Power) and watch whether it stays away. Not caused by an app change found in the code.
