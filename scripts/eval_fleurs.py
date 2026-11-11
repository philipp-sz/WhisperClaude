"""WER on FLEURS dev (read speech, German and English), through the app's transcription path.

Method of docs/transcription.md: the first N sentences without digits (only number formatting
differs there), reference = FLEURS' raw transcription, WER as in eval_quality.py (case and
punctuation ignored). "noisy" adds white noise 10 dB below the speech. The data is not part of
the repository: FLEURS (CC-BY-4.0) from huggingface.co/datasets/google/fleurs, `data/<lang>/dev.tsv`
and `data/<lang>/audio/dev.tar.gz`, unpacked flat into <dir>/de_de and <dir>/en_us.

.venv\\Scripts\\python.exe scripts\\eval_fleurs.py --dir D:\\fleurs --models turbo-int8 small-cpu
Models are the Whisper OpenVINO keys of eval_quality.py and small-cpu (faster-whisper small on the CPU).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_quality import MODELS, score # noqa: E402

SR = 16_000
SETS = {"de_de": "German", "en_us": "English"}
RESULTS = Path(os.environ.get("EVAL_RESULTS", ROOT / "bench_audio" / "eval_fleurs.json"))


def sentences(folder: Path, n: int) -> list[tuple[str, str]]:
    """[(wav path, reference text)] of the first n sentences without digits."""
    out = []
    with open(folder / "dev.tsv", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) > 3 and not any(c.isdigit() for c in row[2]) and (folder / row[1]).exists():
                out.append((str(folder / row[1]), row[2]))
            if len(out) == n:
                break
    return out


def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono WAV as float32; FLEURS stores 32-bit float, which `wave` can't read."""
    raw = path.read_bytes()
    pos, fmt = 12, None
    while pos + 8 <= len(raw):
        name, size = raw[pos:pos + 4], int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + size]
        if name == b"fmt ":
            fmt = (int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little"),
                   int.from_bytes(body[4:8], "little"), int.from_bytes(body[14:16], "little"))
        elif name == b"data":
            tag, channels, rate, bits = fmt
            assert rate == SR and channels == 1, f"{path.name}: {rate} Hz, {channels} channels"
            if tag == 3 and bits == 32:
                return np.frombuffer(body, "<f4").astype(np.float32)
            if tag == 1 and bits == 16:
                return np.frombuffer(body, "<i2").astype(np.float32) / 32768
            raise ValueError(f"{path.name}: unsupported WAV format {tag}/{bits}")
        pos += 8 + size + size % 2
    raise ValueError(f"{path.name}: no data chunk")


def noisy(audio: np.ndarray, snr_db: float = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(audio ** 2))) or 1e-4
    return (audio + rng.normal(0, rms * 10 ** (-snr_db / 20), len(audio))).astype(np.float32)


def worker(model_key: str, root: Path, n: int, noise: bool) -> None:
    import psutil

    from whisperclaude import transcriber as t

    if model_key == "small-cpu": # the app's CPU fallback (faster-whisper, int8, 4 threads)
        tr = t.Transcriber("small")
    else:
        t.ov_repo = lambda m: MODELS[model_key]
        tr = t.OpenVinoTranscriber(model=model_key)
    tr.transcribe(np.zeros(SR, np.float32))
    for lang in SETS:
        data = [(read_wav(Path(p)), ref) for p, ref in sentences(root / lang, n)]
        for kind in ("clean", "noisy") if noise else ("clean",):
            errors = words = 0
            wall = audio_s = 0.0
            slowest = 0.0
            worst = []
            for audio, ref in data:
                audio = noisy(audio) if kind == "noisy" else audio
                t0 = time.perf_counter()
                text = tr.transcribe(audio)
                dt = time.perf_counter() - t0
                wall, audio_s, slowest = wall + dt, audio_s + len(audio) / SR, max(slowest, dt)
                sc = score(ref, text)
                errors, words = errors + sc["errors"], words + sc["ref_words"]
                worst.append((sc["errors"], ref[:70], text[:70]))
            worst.sort(reverse=True)
            print("RESULT " + json.dumps({
                "model": model_key, "set": lang, "kind": kind, "sentences": len(data),
                "words": words, "errors": errors, "wer": 100 * errors / words,
                "s_per_10s": 10 * wall / audio_s, "max_s": slowest,
                "rss_gb": psutil.Process().memory_info().rss / 2 ** 30, "worst": worst[:3]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path, help="folder with de_de/ and en_us/")
    ap.add_argument("--models", nargs="*", default=["turbo-int8"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--noise", action="store_true", help="also at 10 dB SNR")
    ap.add_argument("--worker")
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.dir, args.n, args.noise)
        sys.stdout.flush()
        os._exit(0) # OpenVINO threads block a normal exit
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else []
    for m in args.models:
        cmd = [sys.executable, __file__, "--worker", m, "--dir", str(args.dir), "--n", str(args.n)]
        p = subprocess.run(cmd + (["--noise"] if args.noise else []), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=ROOT)
        got = [json.loads(line[7:]) for line in p.stdout.splitlines() if line.startswith("RESULT ")]
        if not got:
            print(p.stderr[-1500:])
        results = [r for r in results if r["model"] != m or (r["set"], r["kind"]) not in
                   {(g["set"], g["kind"]) for g in got}] + got
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'model':18s} {'set':8s} {'kind':6s} {'words':>6s} {'errors':>6s} {'WER':>7s} {'s/10s':>6s} {'max s':>6s} {'RAM':>6s}")
    for r in sorted(results, key=lambda r: (r["set"], r["kind"], r["model"])):
        print(f"{r['model']:18s} {SETS[r['set']]:8s} {r['kind']:6s} {r['words']:6d} {r['errors']:6d} "
              f"{r['wer']:6.2f}% {r['s_per_10s']:5.2f}s {r['max_s']:5.1f}s {r['rss_gb']:5.2f}G")


if __name__ == "__main__":
    main()
