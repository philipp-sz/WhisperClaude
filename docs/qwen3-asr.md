# Qwen3-ASR (evaluated and removed, 2026-11-08)

Qwen3-ASR (Alibaba, Apache-2.0, 1.7B and 0.6B parameters, 30 languages including German and English) tops the Hugging Face Open ASR leaderboard of 2026-10-09 with an English average WER of 5.40% (1.7B) and 6.24% (0.6B), against 7.16% for Whisper large-v3. The leaderboard is English only and measured on data-center GPUs. The question was whether this carries over to German and English dictation on the laptop. It was evaluated on 2026-11-08 and added as an opt-in engine (`[model] engine = "qwen"`, `whisperclaude/qwen_asr.py`). Whisper stayed the default.

## Status

Removed from the app on 2026-11-08. Reasons, in order of weight: the speed (the author dictates mostly in one language with a few English words, so the mixed-language weakness below matters less, but 3–6 s between the stop tap and the text does), the RAM (4–7 GB), and a gain that exists only for English. The downloaded models (5.9 GB), their compiled-kernel caches (8.4 GB) and the code were deleted. If it is ever needed again: the ONNX export `andrewleech/qwen3-asr-{1.7b,0.6b}-onnx` (an individual's repository, see the rules in the general knowledge base; no official export exists), the OpenVINO `MakeStateful` trick, the f32 encoder and the other rules below are all that the implementation consisted of (about 350 lines).

The full implementation (`whisperclaude/qwen_asr.py`, its tests, the `engine`/`qwen_model` config keys, the factory in `transcriber.py`, the chunk-length parameter and the `qwen-*` keys of `scripts/eval_quality.py`) is kept in a local git stash, not in the history: `git stash list` (message "Qwen3-ASR engine, removed 2026-11-08"), `git stash show -p` for the tracked part and `git show 'stash@{N}^3'` for the new files. It has the 350-line runtime and 270 lines of tests. `scripts/eval_fleurs.py` stayed (it works for any model).

## Runtimes

| Runtime | Result |
|---|---|
| ONNX export by andrewleech (`qwen3-asr-1.7b-onnx`, `qwen3-asr-0.6b-onnx`: FP32 encoder, int4 RTN decoders), run by `qwen_asr.py` on ONNX Runtime (CPU) | used |
| The same ONNX files read by OpenVINO (CPU, iGPU) | used; the only way found to run the models on the Arc iGPU |
| llama.cpp with the official GGUF files (`ggml-org/Qwen3-ASR-*-GGUF`, builds with Vulkan, SYCL and OpenVINO) | not tested: the downloaded binaries (b12202, checksums verified) were not run, because running downloaded executables was not permitted in this session |
| PyTorch with the `qwen-asr` package | not used: large install on Python 3.14; the ONNX export was checked token for token against it by its author |
| NPU | not tried |

`qwen_asr.py` implements the rest itself: Whisper's log-mel spectrogram (faster-whisper's extractor), the chat prompt with one `<|audio_pad|>` per encoder feature, greedy decoding, and the language header (`language German<asr_text>text`). The app's chunking, language-aware grouping and live transcription are unchanged (`QwenTranscriber` extends the iGPU transcriber). The model writes its own punctuation and casing, so there is no prompt.

## What had to be solved

- OpenVINO reads the ONNX graphs including the int4 `MatMulNBits` operators. The decoder step takes the whole KV cache (28 layers, 8 heads, about 80 MB each for keys and values) as an input and returns it extended, which cost 245 ms per token on the iGPU. `MakeStateful` turns the cache into model state: 58 ms (0.6B) and 65 ms (1.7B) per token. The rest is the 28 layers, each with many small kernels.
- The encoder in f16, the iGPU default, crashed the driver (`CL_OUT_OF_RESOURCES`) on real 28–30 s chunks, every time, on the 0.6B. In f32 the same audio ran through (encoder 0.04–0.2 s per chunk). The encoder runs in f32.
- The stateful decoder failed rarely with "Layout mismatch: variable ... assign output ..." in the iGPU plugin after earlier utterances of other lengths (1.7B, FLEURS German sentence 44 of 150). A fresh infer request did not help. The decoder is recompiled (kernels are cached) and the chunk is decoded again; `QwenAsr.retries` counts these.
- The 0.6B on the iGPU wrote garbage for 2 of 13 clips (`en_message` became one character, `cmd_de` became "ology") with f16 prefill. f32 prefill gave correct text but took 45–80 s per clip. The 0.6B therefore always runs on the CPU.
- The model sometimes starts with `<|im_start|>` instead of `language ...` and then garbles the text. That token is excluded from the first step.
- A word list as `context` (the system message), the Whisper prompt equivalent, made the 0.6B print Arabic characters for 2 clips and did not help the 1.7B (6.5% against 6.3%). Context is off. Four bilingual contexts (the Whisper prompt, an instruction not to translate, a mixed sample, both) were tested on the mixed clips and two joined dictations of the 1.7B: 11.6–16.9% WER against 8.7% without context.
- A decoder that never emits EOS repeated a phrase (320 words from 30 s). The token budget follows the audio length (0.75 per feature), and a periodic tail of at least 24 tokens is cut.

## Corpus results

Author's corpus as in the laptop measurements above (13 clips with 506 words, 5 joined dictations with 2,285 words), on battery, `scripts\eval_quality.py --models qwen-... --qwen-variants '[{"ctx": false}]'`. Whisper turbo was run in the same session (0.41 s per 10 s here, against 0.30 s on AC power). "Single language" are the 9 German and English clips (437 words), "mixed" the 2 clips with German and English in one sentence (69 words). The 1.7B iGPU row uses the app's 20 s chunk limit; the CPU rows were measured with 30 s chunks.

| Engine and device | Single language | Mixed | Joined | Time per 10 s | Slowest clip | CPU s / GPU s per audio min | RAM | Load |
|---|---|---|---|---|---|---|---|---|
| Whisper large-v3-turbo int8, iGPU | 13 errors | 6 | 84 (3.7%) | 0.41 s | 0.9 s | 2.2 / 2.5 | 1.4 GB | 4.5 s |
| Qwen3-ASR 1.7B int4, iGPU (20 s chunks) | 9 | 25 | 118 (5.2%) | 1.9 s | 5.0 s | 9.8 / 8.1 | 4.3–7 GB | 7–24 s |
| Qwen3-ASR 1.7B int4, CPU (ONNX Runtime) | 9 | 26 | 171 (7.5%) | 12.3 s | 34.8 s | 278 / 0 | 3.7 GB | 19 s |
| Qwen3-ASR 0.6B int4, CPU (ONNX Runtime) | 26 | 12 | 247 (10.8%) | 5.4 s | 15.4 s | 154 / 0 | 1.7 GB | 9 s |
| Qwen3-ASR 0.6B int4, iGPU (not used) | 92 | 12 | 329 (14.4%) | 3.2 s | 10.2 s | 13.3 / 13.6 | 2.3 GB | 16 s |

The timings of the iGPU rows vary between runs: the 1.7B took 1.7–1.9 s per 10 s in its quiet runs and 2.3–3.5 s in earlier ones (other load on the machine, power state), and the 0.6B rows come from those earlier runs. This table shows the 1.7B at its quiet value and the rest as measured; rows should be compared by factors of about 2 or more. The first start compiles the iGPU kernels: 68 s (1.7B) and 79 s (0.6B), once. The RAM of the 1.7B on the iGPU is 4.3 GB after loading and settles at 6.5–7.5 GB during use (no growth over 80 transcriptions, so not a leak). Idle load is 0% GPU and 0.5–2.6% of one core, as with Whisper. Download: 3.9 GB (1.7B) and 1.9 GB (0.6B).

- On single-language clips the 1.7B made 9 errors against 13 for Whisper turbo (437 words, a small sample; FLEURS below). It also had fewer casing errors (41 against 58, 30 s chunks).
- On mixed clips and joined dictations it was worse. Qwen3-ASR assigns one language per utterance and translates or drops the other one: `mix_claude` ("Ich habe gestern mit Claude an meinem Projekt gearbeitet, and honestly it went much better ...") came out as an English translation with 22 of 37 words wrong. Whisper is kept from this by its bilingual prompt. Qwen3-ASR has no equivalent: none of the contexts above helped.
- The 1.7B on the iGPU is 4 to 5 times slower than Whisper turbo: every token needs 28 decoder layers (turbo: 4) and about 1 GB of int4 weights. With live transcription, only the last chunk (up to 20 s) is left after the stop tap: 3–6 s with the 1.7B against 1.0–1.1 s with Whisper (268 s test dictations). CPU and GPU seconds per audio minute are about 4 and 3 times those of Whisper. Energy was not measured (the battery counter is unusable, see above).
- Chunk length: with 30 s chunks, live transcription of one test dictation dropped a whole passage (63 errors, 35 deleted words, against 17 with one-shot transcription; Whisper had 16 against 17). Limits of 20 s and 15 s gave 32 and 28 errors for that clip and left the other clip unchanged (17–21). Over the 5 joined dictations the total was 123 errors (30 s), 129 (one chunk per segment), 118 (20 s) and 119 (15 s), with 42, 38, 29 and 28 deleted words. The app uses 20 s for Qwen.
- A remaining deletion risk: a long dictation can still lose a sentence (Qwen3-ASR sometimes ends early); Whisper turbo deleted 19 words in the same 5 dictations, the 1.7B 29.

## FLEURS

Method as in "NVIDIA GPU: faster-whisper on CUDA": the first sentences without digits of FLEURS dev, "noisy" with white noise 10 dB below the speech, through the app's transcription path (`scripts\eval_fleurs.py`). WER in %. iGPU: 150 sentences per language (about 2,850 words), CPU: 60 (about 1,140 words), so differences below about 1 point are noise on the CPU rows.

| Engine | German clean / noisy | English clean / noisy | Time per 10 s | RAM |
|---|---|---|---|---|
| Whisper large-v3-turbo int8, iGPU | 4.58 / 13.41 | 4.55 / 8.60 | 0.44–0.50 s | 1.5 GB |
| Qwen3-ASR 1.7B int4, iGPU | 4.79 / 13.62 | 3.77 / 6.88 | 2.1–2.3 s | 6.4–7.7 GB |
| Whisper small int8, CPU (60 sentences) | 12.28 / – | 7.37 / – | 3.6–4.2 s | 0.5 GB |
| Qwen3-ASR 0.6B int4, CPU (60 sentences) | 9.56 / – | 7.46 / – | 4.2–4.3 s | 2.8–3.0 GB |

- English: the 1.7B made 17% fewer errors than turbo on clean speech (107 against 129) and 20% fewer on noisy speech (195 against 244). It also beats `large-v3` float16 on the desktop (4.27 / 7.57). The leaderboard result carries over to English read speech.
- German: tie with turbo (137 against 131 errors clean, 390 against 384 noisy). `large-v3` made 3.90 / 11.37 on the desktop.
- The 0.6B on the CPU made 22% fewer errors than `small` on German (109 against 140) and the same on English (85 against 84), at about the same speed and 6 times the RAM.
- The stateful decoder recompiled 5 times in the 600 transcriptions of the 1.7B (see above); every one recovered.

## Decision

Rejected and removed (2026-11-08). Whisper turbo stays the default, and the CPU fallback stays `small`.

- The author's dictation is mostly in one language with a few English words. For that, the 1.7B is as good as Whisper on German (a tie on FLEURS) and better on English (17–20% fewer errors). It is clearly worse only when the language switches inside a sentence.
- The deciding factor is speed: 4 to 5 times slower on the iGPU, 3–6 s (not 1 s) between the stop tap and the text, and 3 to 5 times the RAM. The accuracy gain does not pay for that.
- The 0.6B is better than `small` on German (22% fewer errors) at the same speed, but with 6 times the RAM, and it runs on the CPU only.
- The removed code had no NVIDIA GPU path (ONNX Runtime has no CUDA build here), so there was nothing to keep for a desktop with an NVIDIA card. On such a machine Whisper `large-v3` on CUDA already reaches 3.90 / 11.37 (German) and 4.27 / 7.57 (English) on FLEURS.
- Not done: llama.cpp (the GGUF files and Windows builds with Vulkan, SYCL and OpenVINO exist; running the downloaded binaries was not permitted), the NPU, and a fixed-shape or fused decoder that would cut the 65 ms per token. A faster runtime would lift the speed objection but not the mixed-language one.
- What came out of the search for a better model instead: [model-search.md](model-search.md).
