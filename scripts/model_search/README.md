# Model search scripts (2026-11-08)

Code from the search for a better and faster model than Whisper turbo. Results and decisions: [docs/model-search.md](../../docs/model-search.md).

| File | Purpose |
|---|---|
| `setup_env.py` | Creates the PyTorch environment and downloads Parakeet into `%LOCALAPPDATA%\WhisperClaude\model_search` (about 4 GB, `--remove` deletes it) |
| `names_test.py` | Keyword-list test of Granite 2B on English FLEURS names |
| `eval_models.py` | Corpus and FLEURS evaluation of Granite, Granite 2B, Parakeet and Canary through transformers (CPU) |
| `transcribe_cli.py` | Transcribes WAV files with one of these models; the bridge for `scripts/real_eval.py` and the compare tool |
| `librosa_shim.py` | The one librosa function Parakeet's extractor needs (librosa's scipy DLL is blocked by Windows application control) |

## Environment for the evaluation

`setup_env.py` builds it (a separate virtual environment outside the project, wheels only, Python 3.14); by hand:

```
pip install torch==2.11.0 torchaudio==2.11.0 transformers==5.22.2 faster-whisper==1.2.1 openvino==2026.4.0 psutil soxr
```

torchaudio 2.11 is the newest one, so torch must be 2.11.0. Models come from the original publishers only (`ibm-granite/granite-speech-5.0-470m-turboctc`, `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-v2`). Download `model.safetensors` and the json files with `snapshot_download(allow_patterns=...)` into `$MODEL_SEARCH_DIR/hf` (set `HF_HOME` there; default `%LOCALAPPDATA%\WhisperClaude\model_search`). FLEURS dev (`google/fleurs`, `data/<lang>/dev.tsv` and `audio/dev.tar.gz`) goes flat into `$MODEL_SEARCH_DIR/fleurs/de_de` and `en_us`. Delete the folder when done.
