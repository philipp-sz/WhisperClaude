# Search for a better and faster model (2026-11-08)

Goal: fewer dictation mistakes than Whisper large-v3-turbo at the same or a shorter transcription time (or the same accuracy, but faster). Preprocessing and LLM cleanup had not helped (see [transcription.md](transcription.md)), and Qwen3-ASR was too slow ([qwen3-asr.md](qwen3-asr.md)), so the models were the remaining lever. The scripts are in [scripts/model_search/](../scripts/model_search/README.md). All numbers are from the laptop (Core Ultra 7 258V, Arc 140V iGPU), on battery, Whisper measured in the same session.

## Candidates

| Model | Source (all official) | Type | Languages | Result |
|---|---|---|---|---|
| Granite Speech 5.0 470M TurboCTC | `ibm-granite/granite-speech-5.0-470m-turboctc`, Apache-2.0, 946 MB safetensors | encoder only, CTC | 7 (en, de, fr, es, pt, it, ja) | **best candidate**: faster than Whisper by 4 to 10 times, more accurate on FLEURS, not on the author's clips |
| Granite 5.0 470M TurboCTC `-nc` | `ibm-granite/...-turboctc-nc`, CC-BY-NC-SA-4.0 | same, more training data | English-focused | rejected: 79% WER on German, 6.6% on English (worse than the normal one) |
| Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3`, CC-BY-4.0, 2.5 GB | FastConformer transducer | 25 European | about Whisper's level, not better; useful as a voting partner |
| Canary-1B-v2 | `nvidia/canary-1b-v2`, CC-BY-4.0, 3.9 GB | attention encoder-decoder | 25, language must be given | not better than Whisper, slower, no mixed language |
| Canary-1B-Flash | `nvidia/canary-1b-flash` | attention encoder-decoder | 4 | not tested: no transformers support, needs NVIDIA's NeMo |
| Granite Speech 5.0 2B | `ibm-granite/granite-speech-5.0-2b`, 4.6 GB, ungated | speech LLM, autoregressive | 6 (en, fr, de, es, pt, ja) | tested on the CPU (accuracy only): English better than Whisper, German not, translates mixed sentences, best casing, **keyword biasing for names works**; for the desktop |
| Cohere Transcribe 2B (gated repository: needs an account), Canary-Qwen 2.5B, Voxtral | official | LLM-style, autoregressive | | not tested: the token-by-token cost that made Qwen3-ASR 4 to 5 times slower on the iGPU; Cohere's repository needs an account and agreement, not done without the author's OK |
| German fine-tunes of Whisper or Parakeet (community) | individuals' repositories | | | not tested: individuals' repositories need approval first (rules in the general knowledge base) |

## Results

Same method as before: FLEURS dev (first 150 sentences without digits per language, clean and with white noise 10 dB below the speech), and the author's corpus (13 clips with 506 words; the 5 joined dictations repeat the same 11 texts, so they add almost no independent evidence). WER in %. The models ran on the CPU in float32 (4 threads) through transformers, Whisper through its iGPU backend; the accuracy does not depend on the device (the iGPU conversion of Granite gave the same tokens).

| Model | German clean / noisy | English clean / noisy | All four | Corpus errors: single-language (437 words) / mixed (69) | Casing errors (corpus) |
|---|---|---|---|---|---|
| Whisper large-v3-turbo int8 | 4.58 / 13.41 | 4.55 / 8.60 | 7.79 | 13 / 6 | 58 |
| Granite 470M TurboCTC, spelling fixed | 4.26 / 12.19 | 4.34 / 7.05 | 6.97 | 20 / 6 | 99 |
| Parakeet TDT 0.6B v3 | 5.06 / 12.78 | 4.76 / 7.79 | 7.61 | 14 / 7 | 69 |
| Qwen3-ASR 1.7B (removed) | 4.79 / 13.62 | 3.77 / 6.88 | | 9 / 25 | 41 |
| Canary-1B-v2 (first 60 sentences, clean only) | 4.74 | 6.15 | | 13 / 30 (language forced to German) | 21 |
| Granite Speech 5.0 2B (first 60 sentences, clean only, CPU) | 5.44 | 4.21 | | 13 / 26 | 27 |
| Whisper turbo on the same first 60 sentences | 5.09 | 5.00 | | | |
| Granite on the same first 60 sentences | 4.47 | 4.74 | | | |

- Granite makes 10% fewer errors than Whisper turbo on FLEURS (6.97% against 7.79%), most clearly on noisy audio (English noisy 7.05 against 8.60). On the author's clips it makes more errors (20 against 13 on 437 words, a sample too small to be sure: the difference of 7 errors is about 1.2 standard deviations).
- Granite keeps both languages in a mixed sentence ("Ich habe gestern mit Claude an meinem Projekt gearbeitet and honestly it went much better than I expected. Der Code ist jetzt viel übersichtlicher und the tests finally pass"): 6 errors on the mixed clips, as many as Whisper. Qwen3-ASR and Canary translate. A CTC model decides per frame and has no language per utterance.
- Granite writes punctuation, but gets noun capitalization wrong more often (99 casing errors on the corpus against 58 for Whisper, 21 for Canary), writes some words in the spelling from before 1996 ("muß", "paßt", "daß": 3 of the 24 single-language errors and 13 of the 158 errors on the joined dictations, fixed by a list of 12 words, `fix_old_spelling`), and sometimes drops the first word of a chunk ("Hey").
- Parakeet is not better than Whisper on German (5.06 against 4.58 on FLEURS) and equal on the clips.
- Canary-1B-v2 has the best casing and equals Whisper on the clips, but is not more accurate on FLEURS, needs the language per call and was 3.2 to 5 s per 10 s on the CPU.

Granite 2B differs from the other models in one respect: **keyword biasing**. The prompt takes a list of words ("transcribe the speech to text. Keywords: a, b, ..."), and names are the weak spot of every model (about 13% wrong on English FLEURS). Test on the first 50 English FLEURS sentences (101 names: capitalized words inside a sentence), CPU, `scripts/model_search/names_test.py`:

| Keywords | WER | Names wrong |
|---|---|---|
| none | 4.67% | 11 of 101 |
| one shared list of the 50 names in the set (like a vocabulary list in config.toml) | **4.15%** | **5** |
| only the names of that sentence (1 to 4 words per call) | 7.27% | 6 |

A medium list helps names without hurting the rest; a very short list per call made the model insert and change words (45 → 70 errors). The sample is small. Without keywords the 2B reads like Whisper on the clips (13 errors), is better on English FLEURS (4.21 against 5.00), worse on German (5.44 against 5.09), has the best casing (27 casing errors on the clips, Whisper 58) and translates English parts of mixed sentences into German, like Qwen3-ASR ("die Tests finally pass" stayed, "it went much better than I expected" became German). On the CPU it takes 9 to 11 s per 10 s of speech; the speed on an NVIDIA GPU is not measured.

## Speed

| | Time per 10 s of speech |
|---|---|
| Whisper large-v3-turbo int8, iGPU (today) | 0.41–0.50 s |
| Granite, iGPU, OpenVINO f16 (encoder 65–120 ms per chunk of any length up to 30.5 s, numpy front end 15–35 ms) | about 0.10–0.15 s per chunk |
| Granite, CPU, float32, 4 threads (transformers) | 0.8–1.4 s |
| Parakeet, CPU, float32 | 1.6–1.7 s |
| Canary-1B-v2, CPU, float32 | 3.2–5.1 s |

With live transcription only the last chunk is left after the stop tap: about 0.15 s with Granite against 0.7 s with Whisper. The encoder time on the iGPU is flat (the graph is launch-bound), so one fixed 30.5 s graph serves every chunk length. bfloat16 on this CPU was 20 to 30 times slower than float32 (36–51 s for a 17–20 s clip) and was not used.

## Granite on the iGPU without PyTorch

The engine ([granite.py](../whisperclaude/granite.py), [granite_frontend.py](../whisperclaude/granite_frontend.py), converter [convert_granite.py](../scripts/convert_granite.py)) is built on a prototype that was checked against PyTorch:

- The encoder and CTC head convert with `openvino.convert_model` (needs torch, torchaudio and transformers once). The Hugging Face code reads the sequence length as a Python int, so the traced graph has a fixed input length: 30.5 s (a chunk is at most 30 s plus the 0.5 s lead-in), shorter audio is padded and masked. The converted graph gave the same tokens as PyTorch on the CPU, on the iGPU in f32 and on the iGPU in f16.
- The front end (80 HTK mel bins with n_fft 512, floor 8 dB below the maximum, deltas, two frames stacked) is reproduced in numpy within 1e-5 of the Hugging Face one, 29 ms per chunk. With it the runtime needs only numpy, OpenVINO and `tokenizers`; the transcripts of 12 of 13 clips equal the PyTorch ones, the 13th differs in one word.
- Greedy CTC decoding is: argmax per frame, collapse repeats, drop blanks (id 0), `tokenizers` decode.
- Sizes: float32 IR 2 GB (the iGPU compiles it to f16), compile 10–12 s per start (cacheable).

## Combining models

Different models make different mistakes. Per sentence, an oracle that picks the better of Whisper and Granite would reach 5.45% on FLEURS (all four sets) against 6.97% for Granite alone. Ways to realize that gain:

| Method | FLEURS all four | Cost | Verdict |
|---|---|---|---|
| Whisper alone | 7.79% | 0.4 s | baseline |
| Granite alone | 6.97% | 0.1 s | |
| Whisper for the sentences Granite is least sure of (mean, minimum or share of low token probabilities), 5–50% of sentences | 6.78–7.30% | Granite + a share of Whisper | not worth it: the confidence predicts errors too weakly (correlation of the error rate with mean confidence −0.75, but only 0.2 points gained) |
| Whisper's words, replaced by Granite's where Granite's word probability is at least 0.8 | 6.82% | Granite + Whisper | about Granite alone |
| Granite on five speed-perturbed copies (0.9 to 1.1), vote of three | 6.76–6.80% | 3 × Granite | not worth it: the copies make the same mistakes (−3%) |
| **Vote of Granite, Whisper and Parakeet, pivot Granite** | **6.16%** (German clean 3.63, noisy 10.16) | all three | −12% against Granite, −21% against Whisper |
| Vote, pivot Whisper | 6.54% | all three | |
| Vote, pivot Parakeet | 6.33% | all three | |

On the author's clips the vote with Whisper as pivot makes 10 errors against Whisper's 13 and Granite's 20. The vote ([vote.py](../whisperclaude/vote.py)) keeps the pivot's words unless both others agree on another word. Its cost is the sum of the three models: Granite 0.1 s, Whisper 0.4 s and Parakeet (not converted for the iGPU; 1.7 s per 10 s on the CPU), so about 0.7 s or more per 10 s against 0.4 s now. It also needs a Parakeet runtime (a transducer decoding loop on top of the encoder). Not built.

## Did not help

- Input level: the corpus is at −26 dBFS; normalizing to −26 or −16 dBFS changed nothing for Granite.
- The lead-in silence before a chunk (0, 0.2, 0.5, 1.0 s) and the chunk length (15 or 30 s): all within noise for Granite (single-language 23–26, joined 151–166 errors).
- Time-stretched copies, confidence-based fallback and the confidence hybrid (see above).
- Granite-nc, the larger candidates and the other Whisper ideas of earlier sessions (beam search, `large-v3`, preprocessing; see [transcription.md](transcription.md)).

## Problems met

- transformers' Granite feature extractor needs `torchaudio`; the newest torchaudio (2.11.1) needs torch 2.11, so the environment pins torch 2.11.0, not the newest 2.15.
- Parakeet's feature extractor needs `librosa`; its scipy DLL (`_rotation_cy`) was blocked by Windows application control ("Anwendungssteuerungsrichtlinie"). That is a security feature and is not worked around: [librosa_shim.py](../scripts/model_search/librosa_shim.py) supplies the one function used (the Slaney mel filters, identical to faster-whisper's).
- `from_pretrained(device_map=...)` needs `accelerate`; loading without it is enough on the CPU.
- Parakeet's `generate` returns an object with `.sequences`, not the ids.
- Model files came from the publishers' repositories; only `model.safetensors` and the json files were downloaded (the `.nemo` files contain a pickled checkpoint). The GGUF/ONNX conversions of individuals were not used.

## What was built afterwards (2026-11-08)

- **Granite engine** (`[model] engine = "granite"`): [granite.py](../whisperclaude/granite.py) (engine, CTC decoding, spelling fix), [granite_frontend.py](../whisperclaude/granite_frontend.py) (numpy front end), [convert_granite.py](../scripts/convert_granite.py) and [install_granite.py](../scripts/install_granite.py) (one-time conversion in a temporary environment with pinned wheels; downloads the weights from `ibm-granite`, writes about 1 GB of fp16 OpenVINO files to `%LOCALAPPDATA%\WhisperClaude\granite`, deletes the environment). It plugs into the app like the other backends: `create_transcriber(engine="granite")` returns an `OpenVinoTranscriber` subclass, so chunking, live transcription and the status pill are shared. On the author's corpus through the app's own path: 21 / 6 / 145 errors (single-language / mixed / joined), 0.10 s per 10 s on short clips and 0.06 s per 10 s on 268 s dictations (Whisper turbo 0.41 s), 7.4 s from start to ready. Without the model files, or if it cannot load, the factory falls back to Whisper and tells the user. Not on NVIDIA GPUs (OpenVINO has no CUDA device).
- **One interface for all engines:** every backend is a transcriber with `transcribe`, `transcribe_chunks` and `live`, created by `create_transcriber`. Models that need PyTorch (Parakeet, Granite 2B) run through [transcribe_cli.py](../scripts/model_search/transcribe_cli.py) in the separate environment; [real_engines.py](../scripts/real_engines.py) puts both kinds behind `engine.transcribe(audio)`.
- **Own recordings:** [compare_gui.py](../scripts/compare_gui.py) records free speech, runs every engine, and highlights wrong (red), invented (orange) and missing words against the corrected reference; names are marked `[[Name]]` and counted separately. [real_data.py](../scripts/real_data.py) stores the recordings, references and per-engine outputs in `bench_audio/real/` (git-ignored, it is the author's voice); [real_eval.py](../scripts/real_eval.py) runs any engine over all recordings later and prints the table, so a new model is compared on the same speech.
- **Vote** ([vote.py](../whisperclaude/vote.py)): keeps the pivot's punctuation, casing and line breaks, never replaces words with digits, hands a dropped sentence-final word's period to the previous word. On FLEURS: 6.17% with Granite as pivot (the experiment: 6.16%), 6.71% with Whisper as pivot (the experiment 6.54%: the experiment also turned Whisper's digits into number words, which the reference spells out). It is not part of the app: the third system, Parakeet, has no OpenVINO runtime yet; the vote runs in the comparison tools from cached outputs.

## Improving Granite without a second model

| Idea | Result |
|---|---|
| Replace pre-1996 spellings (`muß`, `daß`) | in the engine: 3 of 24 errors on the clips, 13 of 158 on the joined dictations |
| Vocabulary snap: replace decoded words that are near-misses of listed names | not a clear win: with a noisy stand-in list (79 capitalized English words from FLEURS) names wrong 29 → 23 of 231 at edit distance 1, but 6 more errors elsewhere, and far worse at distance 2 and 3. Needs a curated list: test it on recordings with `[[names]]` marked |
| Fix the casing | 11 of Granite's 26 casing errors on the clips are English loanwords used as German nouns ("Feedback", "Request", "Feature", "Release", "Update" written lowercase; Whisper: 2); 4 are lowercase sentence starts (Whisper 3). No safe general rule: the same words are lowercase in English sentences |
| Time-stretched copies, confidence fallbacks, lead-in, chunk length | no gain (see above) |
| CTC beam search with a word list as a bonus | not built: needs the top-k probabilities of every frame out of the graph (about 100 lines of decoding); the principled fix for names, worth building if names stay the main error on real recordings |
| Granite Speech 5.0 2B | keyword biasing works (names wrong 11 → 5 of 101 with a shared list, see above); autoregressive, so too slow for the laptop iGPU, meant for the desktop with the NVIDIA GPU; weak on mixed sentences |
| Fine-tuning on the author's recordings | possible once the recordings hold a few hours with references; the comparison tool collects exactly that |

## State and open decisions

- **Update 2026-11-11, real recordings** ([real-recordings.md](real-recordings.md)): on 24 free-speech recordings Whisper turbo is the most accurate (4.7%), Parakeet 5.3%, Granite 7.8%; the vote gains only 4 errors (143 against 147). The Parakeet runtime and the vote are not worth building; the points below are superseded where they disagree.
- **Removed 2026-11-11:** the Granite engine (`granite.py`, `granite_frontend.py`, its install and conversion scripts, tests, the `[model] engine` key) and `vote.py` left the code: Granite was worse than turbo on the author's speech (7.8% against 4.7%), and on the NPU it would only beat the CPU fallback slightly while needing its own 1 GB install. Everything is in the git stash "Granite engine and vote, removed 2026-11-11" (`git stash list`, `git stash apply stash@{N}`). Granite and Parakeet can still be compared through the PyTorch bridge (`scripts/real_eval.py run granite parakeet`).

- **Built:** Granite as an opt-in engine, the recording and comparison tools, the vote (library and tools).
- **Default:** still Whisper turbo. Granite is not shown to be better on the author's speech (20 against 13 errors on 437 words); the comparison tool exists to settle that on real recordings (`scripts/compare_gui.py`, then `scripts/real_eval.py`).
- **Vote in the app:** needs a Parakeet runtime for OpenVINO (encoder conversion, transducer decoding loop, a numpy front end; about 200 lines) and about 1.8 times the time. Decide after the real recordings show whether the vote helps on spontaneous speech (FLEURS is read speech).
- **Names:** no engine solves them with the laptop models. Granite 2B with a keyword list halves the wrong names in the test above and is the candidate for the NVIDIA desktop (measure its speed there). On the laptop, a CTC beam search with a word-list bonus for Granite 470M is the open idea.
- **Whisper's invented words:** on noise Whisper writes its own prompt sentence ("Okay, so let's see how this works.") and "you" on silence; a guard that discards output matching the prompt is not built (a user who says "Hallo, das ist ein Test" would be cut).
- **Data needed:** real recordings with references. A first set of 10 to 20 free-speech recordings, with names marked `[[like this]]`, answers most of the open questions.
