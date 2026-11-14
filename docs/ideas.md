# Ideas (collected 2026-11-14, not yet discussed)

The author's ideas after the eighth round, written down as he described them. None is decided or built: each needs a discussion first (feasible? worth it? fits an efficient, reliable tool?). Many depend on the first one working.

## 1. Text-to-text improvement model (the central idea)

- Goal: transcribe reliably, make bullet points where useful, format for notes or e-mails, rewrite based on tone, fix heavy grammar issues, remove filler words, fix self-corrections ("at five … no, at six").
- Thought: maybe the transcription models are not the limit, but understanding the whole context is.
- Candidates: models made to take text in and give improved text out, encoder-decoder models or models tuned for exactly this, unlike the small chat-tuned decoder-only LLMs that were rejected before ([transcription.md](transcription.md#rejected-llm-cleanup)): mEdIT, CoEdIT, Flan-T5 (large) or mT5, S1-mini (English only).
- Could also help translations, and enable new features (tone, notes, e-mails, lists, self-corrections).
- If it works: the formal/informal switch (branch `formality`) may become unnecessary, and the spoken commands (`commands.py`) could be removed (the author finds they interrupt his thinking while speaking and are not reliable enough).

## 2. Features that build on it

- **Pill animation for the stages:** "Transcribing" leaves the pill downwards, "Fixing language" comes in from above, then e.g. "Improving tone". Needs testing.
- **Tone from the window context:** detect the app in focus (e-mail program, chat window, …) and pick the tone from it.
- **Voice instructions on selected text:** select text and say "make this more friendly", "make this more formal", "translate it into German", instead of a separate translation model and tone choice. The author doubts it is feasible without a reliable LLM.
- **Send to a chatbot:** a key or a spoken start phrase sends the transcribed message directly to Claude, Gemini or ChatGPT. Unsure whether possible or sensible.

## 3. Speed and model choice

- **Faster engines plus correction:** newer approaches use VAD and dynamic windows instead of fixed 30 s windows (the app already chunks by VAD), and faster architectures (encoder-only/CTC, TDT such as Parakeet). If the text-to-text step fixes their extra errors, a faster model (already tested: Parakeet, fluid-speech-pro, Granite) could become the main engine: the goal is the best pipeline (model + correction) and the most efficient one, not the best model alone.
- **Live text while speaking** (as in the Gemini and Claude apps, words appear right after they are said): streaming models such as TDT could make it possible. (The author's note on this point was cut off mid-sentence: "In that case of using this kind of different model that transcribes each word as soon as it's spoken, we could actually modify th…")
- **Harness for several models:** instead of one fixed model, let users pick from a chosen set of models (others may prefer something other than Whisper). The pipeline is only tuned and tested for Whisper so far, so it would need generalizing. But keep the number of models small: five transcription, five translation and five text models would be too much for an efficient tool (memory, battery, install size).
- **Runtimes:** not every model runs on OpenVINO, which blocks a multi-model harness. Check Vulkan or SYCL (via llama.cpp/GGUF) as a more general and maybe faster path; whether GGUF models are generally faster.
- **Windows hardware:** optimize further for Intel and NVIDIA (speed, efficiency and quality); test the new features on the desktop's NVIDIA GPU, which hasn't happened yet. Later maybe AMD laptops and PCs.

## 4. Recognition

- **Phoneme-based hotword injection:** could help with unknown or internal words (Claude Code, CUDA, names). The author asks for an explanation of how it works.
- **Very quiet speech / whispering:** currently not recognized; investigate how to handle it.

## 5. Recording and interface

- **Temporary mute while recording:** for a phone call or a quick word with someone, hold a key to mute (seen elsewhere with the space bar; maybe Left Ctrl or another key here).
- **Pill that follows the text field:** a round pill that moves to the text field or the caret (coordinates via Windows UI Automation, if possible), more animated. A random idea; the author is unsure it makes sense.

## 6. Distribution

- Move from a Python app to one that installs with a single command; decide whether to keep the Python version. Question: would a non-Python app (e.g. C++) be more efficient? Explain the options.

## Requests for explanation

- A technical deep dive: how the tool works in detail, how preprocessing and the different model types work, what is worth trying.
- Phoneme-based hotword injection (above).
- Non-Python app options (above).
