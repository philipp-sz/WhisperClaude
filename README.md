# WhisperClaude

Local, private push-to-talk dictation for Windows, inspired by Wispr Flow.
Tap **Left Ctrl**, speak (German, English or both mixed), tap again, and the text is pasted
into whatever text field has focus: Word, the browser, chat apps, editors.

Everything runs on your machine. Audio never leaves the laptop.

## Features

- **One key:** tap Left Ctrl alone to start/stop. Ctrl+C, Ctrl+click or Ctrl+scroll don't trigger it.
- **Fast on laptops:** uses the Intel iGPU via OpenVINO (Whisper large-v3-turbo) when available,
  otherwise faster-whisper on the CPU (Whisper small). Picked automatically at startup.
- **German + English:** auto-detects the language, keeps mixed recordings as spoken.
- **Good punctuation:** a bilingual example prompt sets the style; long recordings are cut at
  speech pauses, so there are no stray capitals after pauses and no words lost at 30 s boundaries.
- **Custom vocabulary:** names and terms Whisper should spell exactly (e.g. "Claude", "VS Code").
- **Floating status pill:** level bars that follow your voice while recording, a wave while
  transcribing, start/stop animations. It never takes focus from the text field.
- **Tray icon:** status, open config, restart (apply config), open log, quit.
- **Clipboard-safe:** your previous clipboard text is restored after pasting.
- **Starts at login** without a console window (Task Scheduler logon task).

## How it works

```
Left Ctrl tap ─► record mic (16 kHz) ─► Left Ctrl tap ─► VAD: cut at pauses into ≤ 30 s chunks
   ─► Whisper (iGPU via OpenVINO, or CPU via faster-whisper) with prompt + vocabulary
   ─► clipboard ─► Ctrl+V into the focused app ─► restore old clipboard
```

## Requirements

- Windows 11 (tested; Windows 10 should work but is untested). Uses Win32 APIs for hotkey,
  overlay and pasting.
- Python 3.12 or newer (developed on 3.14).
- Optional: an Intel GPU/iGPU supported by OpenVINO (e.g. Intel Arc, Core Ultra). Without one,
  the CPU path is used.
- About 1–2 GB free disk space for the models (downloaded on first start).

## Installation

```powershell
git clone https://github.com/philipp-sz/WhisperClaude.git
cd WhisperClaude
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

Run it once with a console to watch the first start (model download; on an iGPU also a
one-time compile of ~15 s):

```powershell
.venv\Scripts\python.exe -m whisperclaude
```

Then install the autostart. It registers a logon task and starts the app right away:

```powershell
.venv\Scripts\python.exe scripts\install_autostart.py            # install
.venv\Scripts\python.exe scripts\install_autostart.py --remove   # uninstall
```

## Usage

1. Click into any text field.
2. Tap **Left Ctrl** (alone, briefly). The pill shows "● Recording".
3. Speak. Tap **Left Ctrl** again. The text appears a moment later.

A tap during transcription is ignored. Recordings under 0.3 s are discarded.

The tray icon (under the **^** arrow next to the clock) shows the state by color: grey = loading,
green = ready, red = recording, amber = transcribing. Right-click for the menu.

## Configuration

Settings live in [`config.toml`](config.toml); every key is optional. After editing, use
tray → **Restart (apply config)**. Invalid values are reported in the log and the defaults are used.

| Key | Default | Meaning |
|---|---|---|
| `hotkey.key` | `"ctrl_l"` | Key to tap: `ctrl_l`, `ctrl_r`, `f9`, `pause`, … |
| `model.device` | `"auto"` | `auto` = iGPU if available, else CPU; `gpu`; `cpu` |
| `model.gpu_model` | `"large-v3-turbo"` | OpenVINO Whisper model on the iGPU |
| `model.cpu_model` | `"small"` | faster-whisper model on the CPU |
| `model.language` | `""` | `""` = auto-detect, or lock to `"de"` / `"en"` |
| `model.initial_prompt` | bilingual example | Punctuated text whose style Whisper copies |
| `model.vocabulary` | `["Claude", "cloud", …]` | Words to spell exactly. Keep the list short. |
| `paste.restore_delay_s` | `0.3` | Wait before restoring the old clipboard |

Logs: `%LOCALAPPDATA%\WhisperClaude\log.txt`. Transcribed text is never written to the log.

## Performance

Measured on an Intel Core Ultra 7 258V (Arc 140V iGPU), battery power:

| Backend | 10 s audio | 60 s audio | Extra energy per 60 s dictation |
|---|---|---|---|
| faster-whisper small, CPU | 3.6 s | 6.9 s | ~38 J |
| **OpenVINO large-v3-turbo, iGPU** | **0.33 s** | **1.5 s** | **~19 J** |

Idle: ~0.05–1 % of one CPU core; no measurable effect on battery draw. RAM: ~1.2 GB with the
iGPU model loaded. Details, including the benchmarks behind each decision, are in
[`docs/PLAN.md`](docs/PLAN.md).

## Privacy

- Audio is recorded only while the pill shows "Recording" and is processed in memory only.
- No cloud services. The only network access is downloading the models from Hugging Face on the
  first start; after that the app runs offline.
- Transcripts are not logged or stored.

## Limitations

- Windows only.
- Only text clipboard content is restored; an image on the clipboard is lost after dictating.
- The app can't see the text field, so dictating directly after existing text doesn't add a
  leading space.
- The NPU isn't used (OpenVINO's Whisper pipeline failed on it in testing).

## Development

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m pytest          # fast tests (no model, no hardware)
.venv\Scripts\python.exe -m pytest -m slow  # real models on CPU and (if present) iGPU
```

Helper scripts in `scripts/`: `benchmark.py` (model speed), `record_clip.py` and
`compare_decoding.py` (decoding quality on your own recordings), `check_env.py` (sanity check).

The two short clips in `tests/data/` were generated with the Windows built-in text-to-speech
voices, so the tests don't contain anyone's real voice.

## Third-party licenses

This repository contains only its own code (MIT, see [LICENSE.md](LICENSE.md)). Dependencies are
installed via pip and models are downloaded at runtime; they keep their own licenses:

| Component | License |
|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper), CTranslate2, ONNX Runtime, sounddevice | MIT |
| Whisper models ([OpenAI](https://github.com/openai/whisper); [Systran](https://huggingface.co/Systran/faster-whisper-small) and [OpenVINO](https://huggingface.co/OpenVINO) conversions) | MIT |
| Silero VAD (bundled with faster-whisper) | MIT |
| OpenVINO, OpenVINO GenAI, OpenVINO Tokenizers, Hugging Face tokenizers / huggingface_hub | Apache-2.0 |
| NumPy, PyAV, psutil, pyperclip | BSD |
| Pillow | MIT-CMU |
| [pynput](https://github.com/moses-palmer/pynput), [pystray](https://github.com/moses-palmer/pystray) | LGPL-3.0 (used unmodified as separately installed libraries) |

"Wispr Flow" is mentioned only to describe the idea; this project is not affiliated with it.
