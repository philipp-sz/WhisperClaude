"""Creates the PyTorch environment for the model search tools and downloads Parakeet (for the vote).

.venv\Scripts\python.exe scripts\model_search\setup_env.py [--yes] [--remove]

Everything goes into %LOCALAPPDATA%\WhisperClaude\model_search (about 4 GB): a virtual environment with pinned
wheels from PyPI (torch, torchaudio, transformers, faster-whisper, openvino) and nvidia/parakeet-tdt-0.6b-v3
(model.safetensors and the json files, from the publisher). Afterwards set WC_TORCH_PYTHON to the printed
python.exe, and scripts\real_eval.py run parakeet works. --remove deletes the folder.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperClaude" / "model_search"
PACKAGES = ["torch==2.11.0", "torchaudio==2.11.0", "transformers==5.22.2", "faster-whisper==1.2.1",
            "openvino==2026.4.0", "psutil==7.2.2", "soxr==1.1.0"]
DOWNLOAD = ("from huggingface_hub import snapshot_download; "
            "snapshot_download('nvidia/parakeet-tdt-0.6b-v3', allow_patterns=['model.safetensors', '*.json'])")


def run(args: list[str], env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def main() -> None:
    if "--remove" in sys.argv:
        shutil.rmtree(HOME, ignore_errors=True)
        print(f"Removed {HOME}")
        return
    print(__doc__)
    if "--yes" not in sys.argv and input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")
    HOME.mkdir(parents=True, exist_ok=True)
    python = HOME / "venv" / "Scripts" / "python.exe"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(HOME / "venv")])
    run([str(python), "-m", "pip", "install", "--only-binary=:all:", "--disable-pip-version-check", *PACKAGES])
    run([str(python), "-c", DOWNLOAD], env={**os.environ, "HF_HOME": str(HOME / "hf")})
    print(f"\nDone. Set WC_TORCH_PYTHON={python}\n(The model search scripts keep their data in {HOME}.)")


if __name__ == "__main__":
    main()
