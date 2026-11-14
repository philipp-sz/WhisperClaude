# Transcription

Measurements without other hardware named were taken on the laptop (Intel Core Ultra 7 258V, Arc 140V iGPU), on battery. Timings vary by about ±20% with the power state.

## Backend and model

### CPU: faster-whisper

10 s clips, median of 3 warm runs, CTranslate2 int8:

| Model and settings | Time per 10 s clip | CPU load | RAM |
|---|---|---|---|
| small, 4 threads, beam 5 | 3.6–4.5 s | ~50% | 0.7 GB |
| small, 8 threads, beam 5 | 3.5 s | ~93% | 0.7 GB |
| small, 4 threads, beam 1 (chosen) | 3.3–3.4 s | ~49% | 0.7 GB |
| medium, 4 / 8 threads | 15.5 / 11 s | 48% / 90% | 1.8 GB |
| large-v3-turbo, 4 / 8 threads | 19–23 / 13 s | 49% / 91% | 1.9 GB |

Eight threads were slower than four for `small` because of synchronization between P-cores and E-cores. Beam size 1 was about 10% faster than 5 with the same output quality. `medium` was no more accurate than `small` on German.

### iGPU: OpenVINO GenAI

| Backend | 10 s clip | 60 s clip | CPU time per 10 s | Energy per 60 s dictation |
|---|---|---|---|---|
| faster-whisper small, CPU | 3.6 s | 6.9 s | 13.6 s | ~38 J |
| OpenVINO small, iGPU | 0.26 s | 1.5 s | 0.4 s | ~19 J |
| OpenVINO large-v3-turbo, iGPU (chosen) | 0.33 s | 1.5 s | 0.4 s | ~19 J |

`large-v3-turbo` on the iGPU is about 10 times faster than `small` on the CPU, uses about half the energy and transcribes compound words and casing more accurately. The pre-converted `OpenVINO/whisper-*-int8-ov` models are used. Compiled kernels are cached, so only the first start takes about 15 s. The NPU is not used because OpenVINO 2026.4's `WhisperPipeline` crashes on it.

Further variants of the iGPU model, tested on the quality corpus below:

| Variant | Result |
|---|---|
| turbo int8 (chosen) | 3.6% WER on short clips |
| turbo fp16 | same WER, +0.8 GB RAM, 28 s first compile |
| large-v3 int8 | same WER, 4 times slower, 3 GB RAM |
| beam search | not implemented on the iGPU in OpenVINO GenAI 2026.4 |
| 80 Hz high-pass, loudness normalization | no change |

### NVIDIA GPU: faster-whisper on CUDA

With an NVIDIA GPU, Whisper runs through faster-whisper (CTranslate2) on CUDA: `large-v3` or `large-v3-turbo` by free memory, see below. It uses the same chunking as the iGPU path: chunks split by detected language, one model call per chunk. The batched pipeline of the CPU path detects one language for all chunks. On an RTX 4070, `large-v3-turbo` (float16, beam 1) transcribed an 8–11 s dictation in 0.30–0.32 s and a 28 s dictation in three language-split chunks 1.23 s (text-to-speech clips). The GPU is not the limit there: faster-whisper computes the mel spectrogram on the CPU, and each chunk is encoded twice (language detection, then transcription). CTranslate2 loads cuBLAS and cuDNN by name on first use, and Windows does not search the folders of the `nvidia-*` pip packages. The app therefore loads these DLLs itself before the model (`load_cuda_libraries`). A cuBLAS on `PATH` (for example from conda) is not used.

Precision and beam size on CUDA, tested on FLEURS dev (150 read sentences each in German and English, about 3,000 words per set; sentences with digits left out, since only number formatting differs there). "Noisy" adds white noise 10 dB below the speech. Settings as in the app (prompt, 0.5 s lead-in, no timestamps). RTX 4070:

| `large-v3-turbo` | German clean / noisy | English clean / noisy | Time per 10 s | VRAM |
|---|---|---|---|---|
| float16, beam 1 | 4.67% / 14.39% | 4.89% / 9.17% | 0.28–0.35 s | 2.2 GB |
| int8_float16, beam 1 | 4.49% / 14.95% | 4.93% / 9.36% | 0.26–0.31 s | 1.0 GB |
| float32, beam 1 | 4.70% / 14.32% | 4.99% / 9.10% | 0.48–0.58 s | 4.0 GB |
| float16, beam 5 (chosen) | 4.49% / 14.11% | 4.80% / 8.65% | 0.30–0.35 s | 2.0 GB |

The precision made no measurable difference: the gaps are 6–17 words of about 3,000. Noise doubled to tripled the errors. Beam 5 was better in all four sets at about 5% more time, so `beam_size = 0` (auto) means 5 on CUDA. The CUDA path uses float16, or int8 if the GPU has less than 4 GB free (read from NVML when the model loads), so a small GPU still fits Whisper and the translation model. Most errors were misheard words ("Krankenkarte" for "Krankheit", "Mammoskulpturen" for "Marmorskulpturen") or formatting ("18" for "eighteen", "sodass" for "so dass").

`large-v3` (full decoder, 3 GB download) on the same sets and GPU:

| `large-v3` | German clean / noisy | English clean / noisy | Time per 10 s | VRAM |
|---|---|---|---|---|
| float16, beam 5 (chosen with 7 GB free) | 3.90% / 11.37% | 4.27% / 7.57% | 0.66–0.69 s | 3.9 GB |
| float16, beam 1 | 4.04% / 12.29% | 4.34% / 8.42% | 0.58–0.61 s | 3.9 GB |
| int8_float16, beam 5 | 3.93% / 11.62% | 4.44% / 8.29% | 0.76–0.79 s | |
| float32, beam 5 | 3.90% / 11.30% | 4.31% / not run | 0.95–1.03 s | 4.2 GB |

`large-v3` made 11–19% fewer errors than `large-v3-turbo` in all four sets, at 2.3 times the time. Through the app's chunking, a 10 s dictation took 0.82 s instead of 0.33 s, and 89 s of audio at once 5.6 s instead of 2.5 s. Live transcription leaves at most one 30 s chunk for after the stop tap. `[model] gpu_model = "auto"` therefore picks by free VRAM when the model loads: `large-v3` float16 with at least 7 GB (Whisper and NLLB-1.3B then use about 5.5 GB), `large-v3` int8 with at least 4 GB, else `large-v3-turbo` int8. `large-v3` in int8 needs 2.1 GB, as much as the turbo in float16 (2.2 GB). It was slightly worse than `large-v3` float16 in all four sets but still better than the turbo float16 with beam 5 in all four (German 3.93 / 11.62 vs 4.49 / 14.11, English 4.44 / 8.29 vs 4.80 / 8.71). On the iGPU `large-v3` was 4 times slower at the same WER on the author's voice, so the iGPU keeps the turbo.

Without a GPU, faster-whisper took 14 s for 10 s of audio with `large-v3-turbo` and 17.5 s with `large-v3` (int8, 4 threads, Core i5-10600K). Both are slower than real time, so the CPU fallback stays `small` (3.4 s). The encoder is the same size in both models and dominates on the CPU, so `large-v3` costs only 25% more there. The distilled `distil-large-v3` models were not tested: they are trained on English only.

Audio preprocessing, `large-v3-turbo` float16 beam 5, same sets plus "quiet" (speech at -50 dBFS, white hiss at -68 dBFS, like a quiet laptop microphone). WER in %:

| Preprocessing | German clean / noisy / quiet | English clean / noisy / quiet | Mean |
|---|---|---|---|
| none | 4.49 / 14.11 / 6.67 | 4.80 / 8.71 / 5.48 | 7.38 |
| RMS normalization to -20 dBFS | 4.18 / 14.15 / 6.49 | 4.60 / 8.42 / 5.48 | 7.22 |
| 80 Hz high-pass + normalization | 4.14 / 14.04 / 6.77 | 4.67 / 8.12 / 5.48 | 7.20 |
| noise reduction + normalization | 4.14 / 13.86 / 6.67 | 5.06 / 8.29 / 5.81 | 7.31 |
| strong noise reduction + normalization | 4.07 / 12.99 / 6.53 | 5.02 / 7.67 / 5.74 | 7.00 |

The noise reduction is a Wiener filter on the spectrogram: the noise is the mean power of the quietest 20% of frames, the gain is floored at 0.15 (strong: noise counted twice, floor 0.08). It raised the signal-to-noise ratio of the noisy set from 10 to 16.6 dB. Only the strong variant helped, and only on the noisy sets (about 1 point). It made clean and quiet English slightly worse, and white noise is the easiest case for it. Normalization gained 0.2–0.3 points in 4 of 6 sets, about 6–9 words; on the author's own recordings it made no difference (see "iGPU: OpenVINO GenAI"). None of it is used: switching to `large-v3` reduced the errors on noisy audio 2–3 times more than the best preprocessing.

On the CPU (`small`, Core i5-10600K, 40 sentences per set, about 800 words), int8 scored 8.98% / 33.33% (German clean / noisy) and 5.32% / 13.36% (English), float32 8.46% / 33.46% and 5.06% / 13.10%. The gaps are 2–4 words, and float32 took 20% longer, so the CPU stays on int8. On the iGPU, int8 and fp16 had the same WER on the author's voice (see "iGPU: OpenVINO GenAI").

Smaller precision for the iGPU and the CPU, OpenVINO GenAI on the CPU device of the desktop (Core i5-10600K, 4 threads), same FLEURS sets with 30 sentences each (about 600 words, so differences below 1 point are noise). The accuracy carries over to the iGPU; the iGPU speed has to be measured on the laptop (`scripts/eval_quality.py --models turbo-int8 v3-int8 v3-int4 turbo-int4`):

| OpenVINO model | German clean / noisy | English clean / noisy | Mean | CPU time per 10 s | RAM |
|---|---|---|---|---|---|
| `large-v3-turbo` int8 (iGPU default) | 4.55% / 14.17% | 4.74% / 6.74% | 7.55% | 7.6–9.4 s | 1.9 GB |
| `large-v3-turbo` int4 | 5.56% / 17.03% | 3.83% / 8.74% | 8.79% | 6.4–8.0 s | 1.2 GB |
| `large-v3` int8 | 4.05% / 12.82% | 4.19% / 6.56% | 6.90% | 11.7–13.3 s | 4.0 GB |
| `large-v3` int4 | 4.22% / 13.83% | 4.92% / 8.93% | 7.97% | 9.0–10.4 s | 2.6 GB |

int4 weights cost accuracy for both models, mostly on noisy audio, and `large-v3` in int4 was no better than the turbo in int8. They are not used. `large-v3` in int8 had the fewest errors, 9% fewer than the turbo, at 1.5 times the CPU time and twice the RAM. On the author's voice on the iGPU it had the same WER as the turbo and took 4 times as long (see "iGPU: OpenVINO GenAI"), so the iGPU keeps the turbo; `[model] gpu_model = "large-v3"` switches it.

On the CPU, OpenVINO ran `large-v3-turbo` int8 in 7.6–9.4 s per 10 s, faster than faster-whisper (14 s) and at about half the errors of `small` (see below). It is still too slow for the CPU fallback: the last chunk of a dictation (up to 30 s) would take up to about 28 s after the stop tap, against about 10 s with `small`.

OpenVINO lists an NVIDIA card as `GPU` too (through OpenCL, seen with an RTX 4070). Its GPU plugin is made for Intel GPUs, so the iGPU path only takes a device whose name contains "Intel".

## Decoding

Three real dictations with pauses (German 63 s, English 60 s, mixed 33 s), faster-whisper `small` on the CPU:

| Variant | Result |
|---|---|
| A: timestamps | English starts a capital letter after every pause and has no punctuation. In the mixed clip the English part is translated into German. |
| B: no timestamps | Text near the 30 s window boundary is lost or garbled. The end contains hallucinated text. |
| C: timestamps and prompt | Better punctuation, still breaks at pauses. |
| D: no timestamps and prompt | Good punctuation, boundary errors beyond 30 s. |
| E: VAD chunks, no timestamps, prompt (chosen) | Complete and punctuated, languages kept. Same speed as A. |

Whisper was trained on subtitles. With timestamps it splits text at pauses and capitalizes each line. A punctuated bilingual prompt ("Hallo, das ist ein Test. Okay, so let's see how this works…") sets the punctuation style and keeps mixed audio from being translated (only for chunks in the prompt's languages, see below). A custom vocabulary is appended to the prompt. Faster-whisper's `hotwords` fixed the same words but dropped commas.

Prompt only for its own languages (2026-11-13): with the German+English sentence, Whisper translated other languages into English, even with the right language token forced. Since then the sentence is used only for chunks Whisper hears in one of `[model] prompt_languages` (default `["de", "en"]`). Other chunks get only the vocabulary as prompt. Whisper's own language decision picks the prompt, so there is no extra detection pass. The iGPU decodes a chunk with the prompt for the previous chunk's language (kept across dictations) and decodes it again only if the heard language needs the other prompt; the CPU and CUDA paths detect before decoding and only repeat the detection. FLEURS dev, 50 sentences per language, app path on the iGPU (turbo int8), word errors (Japanese: character errors), GPU time per audio minute:

| Set | Before | Prompt by language | GPU s/min before → after |
|---|---|---|---|
| French | 76.8% | 4.7% | 2.60 → 2.94 |
| Spanish | 54.6% | 2.8% | 2.44 → 2.48 |
| Japanese (CER) | 69.6% | 3.9% | 3.27 → 2.50 |
| German | 4.0% | 4.0% | 3.08 → 2.44 |
| English | 4.0% | 4.0% | 3.15 → 2.74 |
| German + 2 s pause + French, both orders (20 dictations) | 43.9% | 8.0% | 4.0 → 5.9 generate calls per dictation |

Earlier run (same rule): Portuguese 54.0% → 4.0%, Dutch 19.2% → 9.3%, Polish 16.2% → 5.4%, Italian 8.4% → 2.4%; the author's 24 recordings and the corpus unchanged. Timing differences on single-language sets are noise (one decode per chunk on both). Resetting the remembered language at every dictation doubled the time of every French dictation (6.3 s per audio minute), so it is kept. CPU path (faster-whisper `small`, 10 sentences each): small did not translate French with the sentence (12.0% → 11.2%), German unchanged (7.5%). The CUDA path uses the same rule but was not measured.

Short chunks (2026-11-13): German and English cut to 1-3 s (the author's recordings, 507 pieces) were heard as another language in 18 pieces (3.6%; 7% at 1 s, 2% at 3 s; mostly Russian and Korean, once each Dutch, Portuguese, French). With the sentence, Whisper still wrote them in Latin script, 34 of 63 words right; decoded again in the heard language, 0 of 46 right (Cyrillic, Hangul, or other words). So a chunk under 4 s of speech (`MIN_SWITCH_S`) heard outside `prompt_languages` counts as "language unknown" and keeps its first decode with the sentence, unless the previous chunk was heard in that language. Rejected: 3 s (the two 3 s pieces, Korean and Dutch, stay broken); keeping the first decode only when it is not in the heard language's script (fixes Russian and Korean, not the Dutch, Portuguese and French ones, which are Latin script either way). Cost: a short dictation in another language right after German or English is decoded as before this change; from its second chunk on the language is trusted. Short foreign speech is the reason for the cost: cut to the first 1-3 s of a sentence (8 FLEURS clips per language), forcing the heard language and dropping the sentence gives far more words of the output found in the reference than main (words found / words output):

| Language, length | Main | Forced language |
|---|---|---|
| French 1 s | 0/23 | 16/21 |
| French 2 s | 8/51 | 41/49 |
| French 3 s | 18/72 | 72/79 |
| Spanish 2 s | 16/40 | 38/42 |
| Russian 2 s | 5/48 | 33/37 |
| Russian 3 s | 17/58 | 53/58 |
| Korean 2 s | 18/104 | 91/99 |
| Korean 3 s | 64/151 | 117/126 |

The guard keeps main's output for a first short foreign chunk after German or English, so this gain applies only from the second chunk on or for chunks of 4 s and more.

Beam search on the iGPU (2026-11-12): OpenVINO GenAI 2026.4 refuses `num_beams > 1` on the GPU ("Not Implemented") and `num_return_sequences > 1`. Optimum Intel (`OVModelForSpeechSeq2Seq`, Hugging Face's own `generate()` on the same OpenVINO model) does run beam search on the iGPU: on the author's 24 recordings 131 errors greedy (app), 127 with 2 beams, 122 with 4 (−7%, fewer deletions and insertions, names and sound-alikes unchanged), but 2.7 times slower per chunk and with torch and transformers (1.4 GB) as dependencies. Not adopted. Details in [real-recordings.md](real-recordings.md#fourth-round-experiments-and-decisions-2026-11-12).

Prompt sentence (2026-11-12): 15 variants on the author's 24 recordings and the corpus; none beat the current one beyond the noise, and any prompt that leans towards one language makes Whisper translate chunks of the other. Keep it short and balanced. Details in [real-recordings.md](real-recordings.md#fifth-round-experiments-and-decisions-2026-11-12-evening).

Language guard (2026-11-14): after tiny input changes (80 Hz high-pass, -6 dB, -16 LUFS) a German chunk was sometimes written in English although Whisper's language token said German (p about 0.997): the English half of the prompt sentence translated it. One passage per recording, 32-46 more errors. So when Whisper heard a prompt language but the text is clearly in the other one (`text_language`: py3langid restricted to the prompt languages, at least 6 words, probability >= 0.9), the chunk is decoded again with the language forced and without the sentence (vocabulary only); the retry is kept unless it is still in the wrong language. Not applied with a fixed `[model] language`. iGPU and CUDA paths; the batched CPU path (faster-whisper `small`) has no per-chunk decode and was not changed. Real language switches pass: language token and text agree. 24 recordings, errors with the guard against without: clean 126 / 126, 80 Hz high-pass 129 / 177, -6 dB 130 / 184, +6 dB 124 / 124. A per-language sentence (German only for German chunks, English only for English ones) fixed the flips as well but cost 34 errors on the corpus: [real-recordings.md](real-recordings.md#eighth-round-experiments-and-decisions-2026-11-14).

## Chunking and quality

Corpus: 13 texts read by the author (about 25 s each; German, English, mixed, German with anglicisms), plus five long dictations joined from them with 2–3.5 s pauses and an extra 2.5 s pause in the middle of a sentence. `scripts/eval_quality.py` measures word error rate (WER), case errors, time and GPU time.

- Whisper picks one language per chunk, from the chunk's start. Merging voice activity detection (VAD) segments blindly into 30 s chunks translated German passages that followed English ones (runs of 12–48 wrong words in 3 of 5 long dictations). One chunk per segment avoided that but made Whisper invent words in short chunks. Segments are now merged only if their detected language matches. Detection costs about 0.26 s of GPU time per segment.
- A chunk that starts directly on a word made Whisper skip its first words. Of 130 random mid-sentence cuts, 13% lost 3 or more words. With 0.3–0.5 s of silence before the chunk, 5% did. Every chunk now starts with 0.5 s of silence.
- Result on the long dictations: WER from 10.0% to 4.0%, no translated passages. Short clips: 3.6% to 3.4%. GPU time rose from 1.5 to 2.3 s per audio minute. Short dictations stay at 0.38 s per 10 s.
- Prompt examples for the spoken commands made Whisper write "Command" on short unclear speech (3 of 66 test dictations). They were removed from the prompt. The commands still work and are off by default.
- Peak normalization of quiet recordings was tested and rejected. Whisper made no errors even at 1% volume.

## Live transcription

Every 5 s while recording, the audio so far is passed to the VAD. A chunk is transcribed once a later chunk has started and the chunk ends more than 5 s before the newest audio. On a 313 s clip, the time from the stop tap to the text fell from 8–13 s to 0.9 s on the iGPU and from 28 s to 6.6 s on the CPU. With language-aware grouping, a 5-minute dictation is pasted 1.3–2.4 s after the stop tap.

Language detection after the stop (2026-11-12): from about 30 s on, detecting each segment's language took 27–30% of the time from the stop tap to the text, because a segment was only detected when it was compared with a following one, so the last ended segment always waited for the stop. Detecting every ended segment during the dictation cut the stop time by 6–12% for 30 s and 90 s dictations, with identical text on all 24 recordings. OpenVINO GPU properties (`PERFORMANCE_HINT: LATENCY`, `GPU_HOST_TASK_PRIORITY`, `GPU_QUEUE_PRIORITY`, `NUM_STREAMS`) changed the time by 2–10%, within the noise; not adopted.

## Qwen3-ASR

A search for a better and faster model than Whisper turbo (Granite, Parakeet, Canary, voting) is in [model-search.md](model-search.md).

Evaluated on 2026-11-08 and rejected: 4 to 5 times slower than Whisper turbo on the iGPU for 17–20% fewer errors on English only. Models and code were removed again. Everything that was measured, solved and decided is in [qwen3-asr.md](qwen3-asr.md); do not repeat the evaluation without a new reason (a faster runtime, or an NVIDIA GPU path).

## Rejected: LLM cleanup

Five local models from 0.27B to 1.2B parameters (llama.cpp, CPU, 11 test cases) were tested for cleaning up the transcript. Models below 1B parameters answered questions in the text, translated mixed text or changed almost nothing. LFM2.5-1.2B fixed misheard words but also translated mixed text and added 1–4 s. Whisper with the prompt was kept without a cleanup step.

A second round (2026-11-01) used OpenVINO GenAI with the pre-converted `OpenVINO/*-int4-ov` models on the laptop's Arc 140V iGPU instead of llama.cpp on the CPU. 19 cases: cleanup, German/English/Spanish/French translation, lists, e-mails, paragraphs, rewriting and "don't answer this" traps. Models: LFM2.5-1.2B, Qwen3-1.7B, Qwen3.5-0.8B, 2B and 4B, Gemma-4-E2B and E4B. Nothing was added to the app, and the scripts and models were deleted. Outcome: not pursued.

- Speed: the iGPU is 3–10 times faster than the CPU (CPU: 2–7 s to the first token, unusable). Tokens per second on the iGPU: LFM2.5-1.2B 84, Qwen3-1.7B 40, Qwen3.5-2B 33, Qwen3.5-0.8B 31, Gemma-4-E2B 27, Qwen3.5-4B 22, Gemma-4-E4B 18. A typical call took 0.3–1.7 s, RAM 1–7.5 GB.
- Quality: without a language hint most models translated German cleanup input into English; telling them that the text is German and must not be translated (Whisper knows the language) fixed it. Translation is good from about 1.2B. Qwen3.5-4B was best overall (passed the traps, natural rewrites). LFM2.5-1.2B had the best speed for its quality (fixed misheard words) but went off script (stray tokens, chatty replies, "merged" became "gemergt"). Qwen3.5-2B and Gemma-4-E4B failed the traps (wrote the e-mail, invented a paragraph), and the 0.8B model answered the question. Any use would need a similarity guard that falls back to the raw text.
- Stability, the blocker: Qwen3.5 (exported as a vision-language model, `VLMPipeline`) and Gemma 4 repeatedly crashed the iGPU driver (`CL_OUT_OF_RESOURCES`, "could not execute a primitive"), once with a 12 minute hang that needed a force-kill and once with a full laptop freeze during a sustained-load test (Kernel-Power 41, LiveKernelEvents). Only LFM2.5 and Qwen3-1.7B (plain `LLMPipeline`) were stable. A driver crash is not acceptable, and the Whisper model shares the iGPU.
- Energy: one clean measurement (LFM2.5 on the iGPU: about 2.4 J per call, 0.1 J per token); the battery rate counter was too coarse for the rest. A few joules per call is estimated, so battery would not be the problem.
- Decision: small local LLMs are not reliable, fast and light enough to justify cleanup, formatting or translation in the app. Revisit only with a more stable GPU driver or OpenVINO release; the first candidates would be LFM2.5-1.2B and Qwen3.5-4B.
- Method notes: pass `enable_thinking=False` for Qwen3 (else about 350 tokens per call); `LLMPipeline.generate` needs a list as input to return performance metrics; avoid long GPU stress tests.

## Laptop measurements (2026-11-02)

Core Ultra 7 258V, 32 GB RAM, Arc 140V iGPU, Windows 11, on the author's corpus (13 recordings with 506 words, plus 5 joined long dictations with 2,285 words and about 267 s of audio each). `scripts\eval_quality.py --models turbo-int8 v3-int8 turbo-int4 v3-int4 --beams 1 --prompts app` on AC power; the iGPU decodes greedily, so only beam 1 applies. Errors are substituted, deleted and inserted words against the reference text.

| OpenVINO model | Errors, short clips | Errors, long dictations | Time per 10 s | Slowest short clip (23 s) | Time per long dictation | GPU s per audio min | RAM |
|---|---|---|---|---|---|---|---|
| `large-v3-turbo` int8 (chosen) | 19 (3.8%) | 84 (3.7%) | 0.30 s | 0.72 s | 12.0–12.4 s | 2.0 | 1.4 GB |
| `large-v3` int8 | 17 (3.4%) | 98 (4.3%) | 1.33 s | 3.05 s | 34–38 s | 7.4 | 2.9 GB |
| `large-v3-turbo` int4 | 24 (4.7%) | 104 (4.6%) | 0.39 s | 1.06 s | 12–13 s | 2.1 | 1.3 GB |
| `large-v3` int4 | 17 (3.4%) | 104 (4.6%) | 1.36 s | 3.24 s | 18–23 s, once 505 s | 4.9 | 2.4 GB |

- `large-v3` is not more accurate than the turbo on the author's voice: the 2-word gap on short clips is noise, and on the long dictations it made more errors (98 against 84). It is 4.4 times slower. With live transcription at most one 30 s chunk is left for after the stop tap, which would take about 3 s instead of 0.7 s. The iGPU keeps the turbo, as on the desktop; `[model] gpu_model = "large-v3"` switches it.
- int4 saves nothing on the iGPU (turbo int4 is slower than int8) and makes more errors. One `large-v3` int4 run took 505 s for a 268 s clip and then 18 s for the same kind of clip, so it stalled once. int4 is not used.
- Idle with the model loaded and no translation model: 1–2.5% of one core in this script, 0% GPU.

Translation on the laptop CPU (CTranslate2 int8, beam 2, median of 5 warm runs; `[translation] model` and `cpu_threads`):

| Model, threads | 1 short phrase | 3 sentences | 10 sentences | CPU time, 3 sentences | RAM | Load |
|---|---|---|---|---|---|---|
| NLLB-600M, 4 | 0.22 s | 0.39–0.45 s | 0.60 s | 1.5–1.7 s | +0.8 GB | 3.8 s |
| NLLB-1.3B, 4 (default) | 0.34 s | 0.61–0.70 s | 1.54 s | 2.3–2.6 s | +1.55 GB | 5.6 s |
| NLLB-600M, 8 | 0.23 s | 0.28–0.35 s | 0.68 s | 2.1–2.7 s | | |
| NLLB-1.3B, 8 | 0.32 s | 0.48–0.64 s | 1.33 s | 3.5–4.5 s | | |

The laptop CPU is 2–3 times faster than the desktop's Core i5-10600K for this (1.3B: 0.6 s for 3 sentences against 1.78 s). Eight threads gain at most 0.1 s and double the CPU time, so 4 stay. In a spot check of 6 short texts (German, English, French, Spanish) both models were correct and differed only in wording ("fertigstellen" against "beenden", "Te escribiré después" against "Después te escribiré brevemente"). The 1.3B is the default when memory allows because of its 1 COMET point on the desktop test (see [translation.md](translation.md)).

The whole app (Whisper turbo on the iGPU plus NLLB in one process) uses 2.2 GB of RAM with the 600M model and 2.9 GB with the 1.3B. Cold start with an empty OpenVINO cache: UI, tray and hotkey after 1.6 s, Whisper ready after 12.6 s, the translation model 4 s later; during the model load the UI thread blocks for about 9 s (cold-disk DLL loading holds the GIL, logged as a warning). With a warm cache Whisper loads in 2.4–3.9 s. The status pill costs 8.6% (gradient), 10.9% (classic) and 13.1% (wave) of one core while it animates, at 200% display scaling; it only animates while recording, transcribing or translating.

Battery: the Windows battery rate counter was not usable for single dictations. It stayed on one value for several seconds and drifted with other programs, so idle readings varied between 4.3 and 5.3 W (once 9.5 W) and a loop of dictations gave a value below idle. The energy figures above stay estimates; CPU and GPU seconds per audio minute are the comparable numbers.

### Idle CPU with the iGPU backend

While idle, the app used 6–9% of one core (of 8) in every one of 12 runs of the real app, not the 0.3–0.6% measured before. It is kernel time in a single thread. With `device = "cpu"` the idle load is 0.07%, and the translation model is not involved (7.6% without it). An app that had run since the evening before showed the same 7.5% as a fresh start, so it does not grow with uptime.

The thread belongs to Intel's OpenCL runtime (`igdrcl64.dll`, loaded by OpenVINO's GPU plugin; Arc 140V driver 32.0.101.8132). Sampling the busy thread (300 samples) showed it in a timed wait loop of that DLL, and 14% of the samples were inside `timeBeginPeriod`, `timeEndPeriod` and `ZwSetTimerResolution`: the runtime changes the system timer resolution every few milliseconds while idle. The cost depends on the timer state of the system. While a second process held `timeBeginPeriod(1)`, the app's idle load fell from 8.4% to 2.7% and went back to 7% afterwards. Processes that only load the model burned 0% in five runs in a row at one time and 8–10% at another; the real app burned every time.

Tried without effect:

- Intel runtime debug keys in the environment (`NEOReadDebugKeys=1` with `DirectSubmissionControllerTimeout=100000`, `EnableDirectSubmissionController=0` or `EnableDirectSubmission=0`): 3 interleaved rounds each on the real app gave 6.5–8.7% in all 12 runs, baseline included. The runtime printed none of its settings, so it ignores the environment; on Windows the keys are read from the registry and need administrator rights.
- Deleting the pipeline: the load stayed in 2 of 3 runs.
- Loading the model on a worker thread, opening the microphone, a Tk window or running an inference: no systematic change.

Not tried: a separate process for the iGPU model that is stopped when idle (ends the load, costs a reload of about 3 s and is a larger change), holding `timeBeginPeriod(1)` in the app (cut the load by two thirds in the test above, but forces a 1 ms system timer, which costs power itself), other driver or OpenVINO versions. The power cost of 7% of one core could not be measured (the battery counter varies by ±0.5 W, and this laptop has no package power counters); the load is about 0.9% of the CPU's capacity.

Still open: a quiet microphone. The dictation errors seen in real use may come from the low input level. Compare the corpus WER at the current Windows input level and at a higher one.
