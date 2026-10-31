# Technical decisions

All measurements were taken on an Intel Core Ultra 7 258V (8 cores) with an Intel Arc 140V iGPU, on battery unless noted. Timings vary by about ±20% with the power state.

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
