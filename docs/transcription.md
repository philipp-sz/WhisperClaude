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

Whisper was trained on subtitles. With timestamps it splits text at pauses and capitalizes each line. A punctuated bilingual prompt ("Hallo, das ist ein Test. Okay, so let's see how this works…") sets the punctuation style and keeps mixed audio from being translated. A custom vocabulary is appended to the prompt. Faster-whisper's `hotwords` fixed the same words but dropped commas.

## Chunking and quality

Corpus: 13 texts read by the author (about 25 s each; German, English, mixed, German with anglicisms), plus five long dictations joined from them with 2–3.5 s pauses and an extra 2.5 s pause in the middle of a sentence. `scripts/eval_quality.py` measures word error rate (WER), case errors, time and GPU time.

- Whisper picks one language per chunk, from the chunk's start. Merging voice activity detection (VAD) segments blindly into 30 s chunks translated German passages that followed English ones (runs of 12–48 wrong words in 3 of 5 long dictations). One chunk per segment avoided that but made Whisper invent words in short chunks. Segments are now merged only if their detected language matches. Detection costs about 0.26 s of GPU time per segment.
- A chunk that starts directly on a word made Whisper skip its first words. Of 130 random mid-sentence cuts, 13% lost 3 or more words. With 0.3–0.5 s of silence before the chunk, 5% did. Every chunk now starts with 0.5 s of silence.
- Result on the long dictations: WER from 10.0% to 4.0%, no translated passages. Short clips: 3.6% to 3.4%. GPU time rose from 1.5 to 2.3 s per audio minute. Short dictations stay at 0.38 s per 10 s.
- Prompt examples for the spoken commands made Whisper write "Command" on short unclear speech (3 of 66 test dictations). They were removed from the prompt. The commands still work and are off by default.
- Peak normalization of quiet recordings was tested and rejected. Whisper made no errors even at 1% volume.

## Live transcription

Every 5 s while recording, the audio so far is passed to the VAD. A chunk is transcribed once a later chunk has started and the chunk ends more than 5 s before the newest audio. On a 313 s clip, the time from the stop tap to the text fell from 8–13 s to 0.9 s on the iGPU and from 28 s to 6.6 s on the CPU. With language-aware grouping, a 5-minute dictation is pasted 1.3–2.4 s after the stop tap.

## Rejected: LLM cleanup

Five local models from 0.27B to 1.2B parameters (llama.cpp, CPU, 11 test cases) were tested for cleaning up the transcript. Models below 1B parameters answered questions in the text, translated mixed text or changed almost nothing. LFM2.5-1.2B fixed misheard words but also translated mixed text and added 1–4 s. Whisper with the prompt was kept without a cleanup step.
