# WhisperClaude

WhisperClaude is a dictation tool for Windows that runs OpenAI's Whisper locally. A tap on Left Ctrl starts recording. A second tap stops it and pastes the transcript into the focused text field. Whisper detects the language of each passage, so one dictation can mix several languages. A tap on Left Alt translates the selected text or the dictation into one of 201 languages. Audio and text never leave the computer. On an Intel Core Ultra 7 258V with its Arc 140V iGPU, 10 s of speech is transcribed in 0.38 s.

<p align="center">
  <img src="docs/images/demo.gif" alt="Status pill: loading, ready, recording with waves that follow the voice, transcribing, pasted" width="300">
</p>

## Usage

Requirements: Windows 11, Python 3.12 or newer and 3–5 GB of disk space for the models. A GPU is optional: an NVIDIA GPU (CUDA) or an Intel GPU or iGPU (Core Ultra, Arc). Without one, the CPU is used.

```powershell
git clone https://github.com/philipp-sz/WhisperClaude.git
cd WhisperClaude
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m whisperclaude
```

For an NVIDIA GPU, install the CUDA libraries as well (about 1.3 GB): `.venv\Scripts\python.exe -m pip install -e .[cuda]`. The first run downloads the models. On an iGPU it also compiles the model once, which takes about 15 s. Autostart registers two Task Scheduler tasks without admin rights. One starts the app at login. The other restarts it after an unlock or a wake from sleep if Windows closed it.

```powershell
.venv\Scripts\python.exe scripts\install_autostart.py            # install and start
.venv\Scripts\python.exe scripts\install_autostart.py --remove   # uninstall
```

### Configuration

All settings are in [`config.toml`](config.toml) in the project folder. Each key is commented and optional. The tray menu entry "Settings (config.toml)" opens the file. "Restart to apply settings" applies changes.

The default prompt is written for German and English. For other languages, three keys should be set before the first dictation:

- `[model] initial_prompt`: one short, correctly punctuated sentence per dictation language. Whisper copies its punctuation and casing. With two languages it also keeps mixed speech from being translated into one language.
- `[translation] my_languages`: the languages the user writes in, most used first. They decide the source language of short texts such as "Gracias". If the list is empty, the app uses the Windows display language, the keyboard layouts and English.
- `[model] vocabulary`: names and terms Whisper should spell exactly.

Example for Spanish and English:

```toml
[model]
initial_prompt = "Hola, esto es una prueba. Okay, so let's see how this works, and then we'll decide."
vocabulary = ["GitHub", "Kubernetes"]

[translation]
my_languages = ["es", "en"]
```

The other settings work with their defaults: hotkeys, device, models, a fixed dictation language, auto-stop, history, tray icon and status pill style, spoken commands and translation.

### Dictating

1. Click into a text field and tap Left Ctrl. A status pill shows waves that follow the voice.
2. Speak, then tap Left Ctrl again. On the Arc 140V iGPU the text is pasted 0.5 s later for a short dictation and 1.3–2.4 s later for a 5-minute dictation.
3. Esc during recording stops it without pasting. The text is kept in "Recent dictations" and marked as cancelled. The pill offers to copy it. After 60 s without speech the recording stops by itself and is kept the same way.
4. After each paste the pill shows a Copy button for 5 s. The tray menu entry "Recent dictations" keeps the last 5 dictations.

The app writes the clipboard only to paste. It restores the previous content afterwards.

### Translating

1. Select text in any app and tap Left Alt. A picker opens with "From" and "To" fields and a list of target languages. Recently used languages come first. Typing filters by name, native name or code.
2. Enter (or Left Alt again) translates. The selection is replaced by the translation. Esc cancels.
3. While dictating, tap Left Alt instead of the second Left Ctrl tap. The picker opens while the dictation is transcribed. Enter pastes the translation, Esc the untranslated dictation.
4. "From" is "Auto" by default and shows the detected language, for example "Auto · German". Tab switches the list to the source language. Enter takes it and goes back to the target. The user's own languages come first in that list.

"Recent dictations" keeps a translated dictation or selection as one entry. Its submenu copies either the original or the translation.

The tray icon shows the state: loading, ready, recording, transcribing and pasted. Its menu has status, recent dictations, settings, restart, log and quit.

<p align="center">
  <img src="docs/images/tray-icon.gif" alt="Tray icon: grey while loading, green when ready, red with sound bars while recording, amber while transcribing" width="200">
</p>

### Formatting by voice

Spoken commands are off by default. Set `enabled = true` in the `[commands]` section of `config.toml` to use them. A command is the trigger word "command" followed by a command word, always in English:

| Spoken | Result |
|---|---|
| `command new line` | line break |
| `command new paragraph` | blank line |
| `command bullet` | new line starting with `- ` |
| `command separator` | `---` with a blank line above and below |
| `command open quote` … `command close quote` | `"…"` |

A bullet item ends at the end of its sentence. Sentences that only contain "new line" or "bullet" stay unchanged. So does "command" after a determiner ("the command bullet"). A misheard command stays in the text.

## Features

- System-wide hotkey. Left Ctrl alone toggles recording. Ctrl+C, Ctrl+click and Ctrl+scroll do not.
- Dictation in Whisper's 100 languages, also mixed, with automatic language detection.
- Translation of selected text or a dictation into 201 languages with NLLB-200, on the GPU or CPU. The source language is detected or picked.
- Custom vocabulary for names and technical terms.
- Live transcription. Finished 30 s chunks are transcribed while recording continues.
- The previous clipboard is restored after pasting, including images and files. Dictations are never synced to the cloud clipboard.
- Status pill with three styles (`[pill] style`): color waves, gradient bars or plain bars. Quiet microphones get more gain, so the waves stay visible.
- Local history of the last 5 dictations in `%LOCALAPPDATA%\WhisperClaude\history.json`, readable only by the Windows user. `[history] keep = 0` turns it off and deletes the file.
- Works offline after the first model download. Logs never contain transcript text.
- 0% GPU while idle. CPU: 0.07% of one core with the CPU backend, 6–9% with the Intel iGPU backend (a thread in Intel's OpenCL runtime, see [docs](docs/transcription.md#idle-cpu-with-the-igpu-backend)).

## How it works

Voice activity detection splits the recording at pauses into chunks of up to 30 s. Finished chunks are transcribed while recording continues. The Whisper backend depends on the hardware: `large-v3` or `large-v3-turbo` on an NVIDIA GPU (by free memory), `large-v3-turbo` on an Intel GPU, `small` on the CPU.

The technical overview and measurements are in [docs/how-it-works.md](docs/how-it-works.md). The experiments behind each choice are in [docs/](docs/README.md).

## Development

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m pytest          # fast tests (no model, no hardware)
.venv\Scripts\python.exe -m pytest -m slow  # real models on CPU and (if present) iGPU / NVIDIA GPU
```

The slow tests need two short recordings of the user's voice, made with `scripts/record_clip.py`. Recordings are git-ignored. Without them the slow tests are skipped.

## Limitations

- Windows only.
- No space is inserted before a dictation that follows existing text. The app cannot read the text field.
- The first 0.2–0.4 s after the start tap can be lost while the microphone opens.
- NLLB adds words to inputs of one or two words ("Ja" becomes "Yes, I did.").
- While the app runs, a tap of Left Alt alone no longer opens an app's menu bar. `[translation] key` sets another key.
- The NLLB-200 weights are licensed CC-BY-NC 4.0, which allows non-commercial use only.

Built with Claude Code. The author specified features, tested and reported bugs. The name refers to Claude Code.

## License

MIT, see [LICENSE.md](LICENSE.md). Dependencies are installed with pip and models are downloaded at runtime. They keep their own licenses:

| Component | License |
|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper), CTranslate2, ONNX Runtime, sounddevice, Whisper models ([OpenAI](https://github.com/openai/whisper), [Systran](https://huggingface.co/Systran/faster-whisper-small) and [OpenVINO](https://huggingface.co/OpenVINO) conversions), Silero VAD | MIT |
| OpenVINO (+ GenAI, Tokenizers), Hugging Face tokenizers / huggingface_hub | Apache-2.0 |
| NumPy, PyAV, psutil, py3langid; Pillow | BSD; MIT-CMU |
| [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) translation model (Meta), [CTranslate2 conversion](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8) | CC-BY-NC-4.0 |
| CUDA libraries (`[cuda]` extra, cuBLAS, cuDNN) | NVIDIA license |
| [pynput](https://github.com/moses-palmer/pynput), [pystray](https://github.com/moses-palmer/pystray) | LGPL-3.0 (unmodified, installed separately) |

This project is not affiliated with Wispr Flow.
