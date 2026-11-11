"""Transcribes WAV files with a PyTorch model of eval_models.py (the bridge for scripts/real_eval.py and the compare tool).

    python transcribe_cli.py <granite|parakeet|granite2b> <out.json> [--keywords "Claude,Gemini"] <wav> [<wav> ...]

Writes {file stem: {"text", "s"}} to out.json after every file. Run it with the Python of the separate environment
described in README.md; the model is loaded once per call.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import eval_models as em
from scripts.real_data import read_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["granite", "parakeet", "granite2b"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--keywords", default="")
    parser.add_argument("wavs", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = em.MODELS[args.model](torch.float32)
    if args.keywords:
        model.keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    model(np.zeros(em.SR, np.float32))
    results: dict[str, dict] = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for wav in args.wavs:
        start = time.perf_counter()
        text = em.transcribe(model, read_wav(wav))
        results[wav.stem] = {"text": text, "s": round(time.perf_counter() - start, 3)}
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(wav.stem, results[wav.stem]["s"], flush=True)


if __name__ == "__main__":
    main()
