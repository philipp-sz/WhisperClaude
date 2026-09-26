"""Benchmark small vs large-v3-turbo (CPU, int8) on a German and an English clip.

Run in a terminal:  .venv\\Scripts\\python.exe scripts\\benchmark.py
- If bench_audio/de.wav or en.wav is missing, it records it from the mic (10 s, countdown).
- --rerecord forces new recordings.
- Each model runs in its own subprocess so RAM numbers aren't mixed up.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "bench_audio"
RESULTS = ROOT / "benchmark_results.json"
SR = 16_000
CLIP_SECONDS = 10
MODELS = ["small", "large-v3-turbo"]
RUNS = 3  # timed runs per clip, after one warm-up

PROMPTS = {
    "de": "Hallo, ich teste gerade meine Diktiersoftware. Morgen früh muss ich in die "
          "Stadt fahren, und danach möchte ich noch einkaufen gehen: Äpfel, Käse und Brötchen.",
    "en": "Hi, this is a quick test of my dictation app. Tomorrow I want to finish the "
          "benchmark, compare both models, and then decide which one feels fast enough.",
}


# ---------- audio I/O ----------

def save_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, f"{path}: need 16 kHz mono"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768


def record(lang):
    import sounddevice as sd

    print(f"\n=== Recording {lang.upper()} clip ({CLIP_SECONDS} s) ===")
    print("Read this aloud at normal speed:\n")
    print("   " + PROMPTS[lang] + "\n")
    input("Press Enter when ready...")
    for i in (3, 2, 1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  >>> SPEAK NOW <<<", flush=True)
    audio = sd.rec(CLIP_SECONDS * SR, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    peak = float(np.abs(audio).max())
    print(f"  done. peak level {peak:.2f}" + ("  (very quiet! check mic)" if peak < 0.05 else ""))
    AUDIO_DIR.mkdir(exist_ok=True)
    save_wav(AUDIO_DIR / f"{lang}.wav", audio)


# ---------- worker (one model per process) ----------

class Sampler:
    """Samples this process's CPU % (normalised to all cores) and RSS every 100 ms."""

    def __init__(self):
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count()
        self.cpu, self.rss = [], []
        self._stop = threading.Event()
        self.proc.cpu_percent(None)  # prime
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(0.1):
            self.cpu.append(self.proc.cpu_percent(None) / self.ncpu)
            self.rss.append(self.proc.memory_info().rss)

    def stop(self):
        self._stop.set()
        self._t.join()
        # fraction of samples where CPU was >= 90 % of all cores
        pinned = sum(c >= 90 for c in self.cpu) / len(self.cpu) if self.cpu else 0.0
        return {
            "cpu_avg": statistics.mean(self.cpu) if self.cpu else 0.0,
            "cpu_peak": max(self.cpu, default=0.0),
            "cpu_pinned_frac": pinned,
            "rss_peak_mb": max(self.rss, default=0) / 2**20,
        }


def worker(model_name, threads):
    sys.path.insert(0, str(ROOT))
    from whisperclaude.transcriber import Transcriber

    clips = {lang: load_wav(AUDIO_DIR / f"{lang}.wav") for lang in PROMPTS}
    rss_before = psutil.Process().memory_info().rss / 2**20

    t0 = time.perf_counter()
    tr = Transcriber(model=model_name, cpu_threads=threads)
    load_s = time.perf_counter() - t0

    tr.transcribe(clips["de"])  # warm-up (first call has one-time costs)

    out = {"model": model_name, "threads": threads, "load_s": load_s,
           "rss_before_mb": rss_before, "clips": {}}
    for lang, audio in clips.items():
        dur = len(audio) / SR
        lat, samplers = [], []
        for _ in range(RUNS):
            s = Sampler()
            t0 = time.perf_counter()
            text, detected = tr.transcribe_detailed(audio)
            lat.append(time.perf_counter() - t0)
            samplers.append(s.stop())
        med = statistics.median(lat)
        out["clips"][lang] = {
            "duration_s": dur,
            "latency_s": med,
            "latency_all": lat,
            "rtf": med / dur,
            "cpu_avg": statistics.mean(x["cpu_avg"] for x in samplers),
            "cpu_peak": max(x["cpu_peak"] for x in samplers),
            "cpu_pinned_frac": statistics.mean(x["cpu_pinned_frac"] for x in samplers),
            "rss_peak_mb": max(x["rss_peak_mb"] for x in samplers),
            "detected": detected,
            "text": text,
        }
    # Windows reports the true peak working set of the whole process lifetime.
    mi = psutil.Process().memory_info()
    out["peak_wset_mb"] = getattr(mi, "peak_wset", mi.rss) / 2**20
    print("RESULT_JSON " + json.dumps(out, ensure_ascii=False))


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerecord", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    # faster-whisper defaults to 4 threads; use all cores unless told otherwise
    ap.add_argument("--threads", type=int, default=psutil.cpu_count())
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return worker(args.worker, args.threads)

    for lang in PROMPTS:
        if args.rerecord or not (AUDIO_DIR / f"{lang}.wav").exists():
            record(lang)

    from faster_whisper import download_model

    results = []
    for m in args.models:
        print(f"\n=== {m}: downloading (if needed) ===", flush=True)
        download_model(m)  # keep download time out of load time
        print(f"=== {m}: benchmarking, {args.threads} threads "
              f"({RUNS} runs per clip after warm-up) ===", flush=True)
        p = subprocess.run(
            # -X utf8: piped stdout is cp1252 on Windows otherwise; umlauts break decoding
            [sys.executable, "-X", "utf8", __file__, "--worker", m,
             "--threads", str(args.threads)],
            capture_output=True, text=True, encoding="utf-8",
        )
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if p.returncode or not line:
            print(p.stdout, p.stderr, sep="\n")
            sys.exit(f"worker for {m} failed")
        results.append(json.loads(line[len("RESULT_JSON "):]))

    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"{'model':<16}{'clip':<5}{'latency':>9}{'RTF':>7}{'CPU avg':>9}{'CPU peak':>10}"
          f"{'>=90%':>7}{'RAM peak':>10}{'load':>7}  lang")
    for r in results:
        for lang, c in r["clips"].items():
            print(f"{r['model']:<16}{lang:<5}{c['latency_s']:>8.2f}s{c['rtf']:>7.2f}"
                  f"{c['cpu_avg']:>8.0f}%{c['cpu_peak']:>9.0f}%{c['cpu_pinned_frac']*100:>6.0f}%"
                  f"{r['peak_wset_mb']:>8.0f}MB{r['load_s']:>6.1f}s  {c['detected']}")
    print("\nTranscripts:")
    for r in results:
        for lang, c in r["clips"].items():
            print(f"  [{r['model']} / {lang}] {c['text']}")
    print(f"\nSaved to {RESULTS.name}")
    print("Rule: turbo only if latency < 2 s per 10 s clip AND CPU not pinned near 100 % for long.")


if __name__ == "__main__":
    main()
