# Translation

## Translate key

Tapping Left Alt alone translates. With text selected, the selection is replaced by its translation. While recording, the tap stops the recording, and the dictation is pasted translated. A list opens to pick the target language. It has a search box, and recently used languages come first. The list opens at the stop tap, so the transcription runs while the language is picked.

The source language can be picked too (Tab switches the list). It is "Auto" by default, and the picker shows what auto-detection found ("Auto · German"). For a dictation the guess appears once the transcript is ready. NLLB does not detect the language itself: the source language is a tag at the start of its input. A wrong tag produced nonsense on short German phrases translated into French with NLLB-600M ("Danke" became "Dans le monde entier", "Kein Problem" became "Le problème de Kein", "Na klar" became "Je ne sais pas"). With the right tag all 10 phrases were translated correctly. A picked source applies to every sentence, except sentences detected as the target language with at least 90% probability, which stay as they are. Mixed-language text therefore needs "Auto".

- A tap of Alt alone opens the focused app's menu bar (Win32 menus, VS Code) or its key tips (Office). Windows does this on the key-up, and only if no other key came between press and release. The keyboard hook therefore swallows the key-up of a tap and replays it behind an unassigned key (vkE8). A WinForms window with a native menu entered menu mode after a synthetic Alt tap without the hook. With the hook it did not. Alt combinations and long presses are not touched.
- The selection is copied with Ctrl+Insert. Ctrl+C would do the same in most apps, but in a terminal without a selection it interrupts the running program. If the clipboard does not change within 0.6 s, nothing was selected. VS Code copies the whole line when nothing is selected and marks that in its clipboard data, so this also counts as no selection. The previous clipboard is restored right after the copy.
- NLLB is tokenized with the `tokenizers` library and the model's `tokenizer.json`. On 2,100 FLORES-200 sentences in 7 scripts the tokens and the decoded text were identical to the transformers tokenizer, so neither transformers nor sentencepiece is needed.

## Translation model

The translation runs on CTranslate2, which faster-whisper already uses. It runs on CUDA or the CPU. CTranslate2 has no Intel GPU backend, so on the laptop the model has to be fast on the CPU. Small multilingual LLMs were ruled out by earlier tests (see [Rejected: LLM cleanup](transcription.md#rejected-llm-cleanup)). The newer small translation models found (GemmaX2-2B, Tower-Plus-2B, HY-MT1.5-1.8B) are such LLMs. HY-MT's license also excludes the EU. The candidates were therefore encoder-decoder models with pre-converted int8 CTranslate2 weights on Hugging Face.

Test: 200 sentences of FLORES-200 devtest in each of 12 directions (German, English, French, Spanish, Italian, Polish, Japanese, Chinese, Ukrainian), plus 25 dictation-style German and English sentences with references written for this test. Scores: COMET-22 (`wmt22-comet-da`) and chrF++. Latency: one 3-sentence German dictation into English on a Core i5-10600K (4 threads, int8) and an RTX 4070 (int8_float16), median of 5 runs. Some CPU runs overlapped with downloads: NLLB-600M at beam 2 took 0.88 s then and 0.77 s on an idle CPU.

| Model (int8) | Size | FLORES COMET / chrF++ | Dictation COMET | Latency CPU | Latency RTX 4070 | RAM (CPU) |
|---|---|---|---|---|---|---|
| M2M100-418M, beam 4 | 0.5 GB | 81.7 / 49.4 | 85.7 | 0.84 s | | 0.52 GB |
| NLLB-200-distilled-600M, beam 1 | 0.6 GB | 84.5 / 51.5 | 88.8 | 0.72 s | | 0.66 GB |
| NLLB-200-distilled-600M, beam 2 (little memory) | 0.6 GB | 85.2 / 52.4 | 89.0 | 0.77–0.88 s | 0.19 s | 0.66 GB |
| NLLB-200-distilled-600M, beam 4 | 0.6 GB | 85.5 / 52.7 | 88.6 | 1.20 s | | 0.66 GB |
| NLLB-200-distilled-1.3B, beam 2 (default) | 1.4 GB | 86.7 / 54.5 | 90.0 | 1.78 s | 0.35 s | 1.44 GB |
| NLLB-200-distilled-1.3B, beam 4 | 1.4 GB | 86.8 / 54.8 | 89.8 | 2.87 s | | 1.44 GB |
| NLLB-600M, beam 2, repetition penalty 1.1 | | 85.2 / 51.8 | 88.4 | | | |
| NLLB-600M, beam 2, wrong source language tag | | 78.3 / 45.9 | 78.2 | | | |

`[translation] model = "auto"` uses NLLB-1.3B, for about 1 COMET point more than NLLB-600M, unless memory is short: with less than 2 GB of free VRAM on an NVIDIA GPU (after Whisper), or with less than 12 GiB of RAM on the CPU (`translator.auto_model`; a 16 GB PC reports about 15.5 GiB, an 8 GB one about 7.5 GiB). On the desktop CPU the 1.3B model takes 2.3 times as long as the 600M (1.78 s against 0.77 s for 3 sentences), on the laptop 1.5 times (0.6 s against 0.4 s, see [laptop measurements](transcription.md#laptop-measurements-2026-11-02)). Either one can be set in `config.toml`. The first start downloads the chosen model (1.4 GB or 0.6 GB), and the Whisper model comes on top (0.5 GB for `small` on the CPU, 0.8 GB for the turbo on the iGPU, 3 GB for `large-v3` on an NVIDIA GPU). Beam 2 scores like beam 4 at lower cost. The repetition penalty made no gain and was dropped. A wrong source language costs 7–11 COMET points, so the source language is detected per sentence (below). M2M100-1.2B was not tested: it is larger than NLLB-600M and scored below it in the NLLB paper.

- NLLB-600M scored higher than M2M100-418M in every direction (chrF++), at a similar size and speed.
- Typical errors: NLLB-600M sometimes drops a clause ("Kein Stress, das hat Zeit bis Freitag." became "Don't worry. It's Friday."). M2M100 translates word by word ("we want to go on a walk?"). All three turned "halb acht" (7:30) into 8:30.
- NLLB adds words to inputs of one or two words ("Ja" became "Yes, I did.") with every beam size and repetition penalty tested. Its subtitle-style artifacts on such inputs (a leading "- ", spaces before punctuation, "I 'm") are removed after the translation.
- Language detection: py3langid answered Serbian for "ok" and Italian for "Hallo", each below 5% probability. For the whole text, the user's own languages therefore count 10 times. Sentences of 25 characters or more get their own language if detected with at least 90% probability. This handles mixed German and English text.
- The own languages come from `[translation] my_languages`, or, if it is empty, from Windows (display language, keyboard layouts, English). Windows does not always know them: with English Windows and a US keyboard, only English counts. Test: 20 short phrases each (1–4 words, "Bis morgen", "Gracias", "Sounds good"), correct source language:

| Own languages | German | Spanish | English |
|---|---|---|---|
| none | 14 | 16 | 15 |
| English (Windows) | 11 | 15 | 20 |
| German, English | 19 | 15 | 18 |
| Spanish, English | 11 | 20 | 16 |
| German only | 20 | 15 | 15 |

Configured languages fix the short phrases in that language. Leaving out English costs English phrases, so the config comment asks to include it.

## Committed memory and MKL (2026-11-11)

On Intel CPUs CTranslate2 uses Intel's MKL for its matrix products. With int8 weights MKL commits memory that it never touches: NLLB-1.3B shows 5.3 GB of private memory but uses 1.4 GB (working set), NLLB-600M 2.3 against 0.7 GB, Whisper `small` on the CPU 2.7 against 0.6 GB. Committed memory is neither RAM nor disk, but it counts against the commit limit (RAM + page file), so on a small machine other programs could fail to get memory. `CT2_USE_MKL=0` switches CTranslate2 to oneDNN: commit equals use (NLLB-1.3B 1.4 GB), the translations were identical (8 of 8 sentences), but translating took 2.2 times as long (0.98 s against 0.44 s per sentence) and Whisper `small` 1.2 times (6.2 s against 5.1 s for 58 s of audio). The app therefore turns MKL off only below 12 GiB of RAM (`lean_cpu_math` in `__main__`, the same threshold that picks NLLB-600M); an explicit `CT2_USE_MKL` in the environment wins.
