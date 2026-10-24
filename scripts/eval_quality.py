"""Transcription quality vs. speed/efficiency on the recorded corpus (scripts/record_corpus.py).

.venv\\Scripts\\python.exe scripts\\eval_quality.py                        # default matrix
.venv\\Scripts\\python.exe scripts\\eval_quality.py --models turbo-int8 --pre none hp

Each model runs in its own process (clean RAM / idle numbers) and goes through the app's own
transcription path (VAD, chunking, prompt), with the variant's tweaks patched in. Reports word
error rate (WER = (substituted + deleted + inserted words) / reference words, ignoring case and
punctuation), invented "command" words, time to text, CPU + GPU seconds per minute of audio,
RAM and idle load. Transcripts are written to bench_audio/eval_results.json (git-ignored).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.record_corpus import CORPUS, CORPUS_DIR  # noqa: E402

SR = 16_000
RESULTS = ROOT / "bench_audio" / "eval_results.json"
MODELS = {
    "turbo-int8": "OpenVINO/whisper-large-v3-turbo-int8-ov",  # today's default
    "turbo-fp16": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "v3-int8": "OpenVINO/whisper-large-v3-int8-ov",
    "v3-fp16": "OpenVINO/whisper-large-v3-fp16-ov",
}
REFS = {cid: text.replace("[Pause] ", "").replace(" [Pause]", "") for cid, text in CORPUS}
# One long dictation switching German <-> English with pauses (chunk borders between languages:
# where "German turned into English" was seen).
# Five orders and pause lengths, so the chunk borders fall differently each time.
_SPEECH = ["de_einigen", "en_message", "de_email", "mix_claude", "de_anglizismen", "en_prompt",
           "de_uni", "mix_paper", "de_alltag", "en_notes", "en_email"]
_rng = np.random.default_rng(7)
JOINED = {f"joined_{k}": ([str(x) for x in _rng.permutation(_SPEECH)],
                          [float(g) for g in _rng.uniform(2.1, 3.5, len(_SPEECH))])
          for k in range(1, 6)}
for _name, (_order, _) in JOINED.items():
    REFS[_name] = " ".join(REFS[o] for o in _order)


# ---------- scoring ----------

NUMBERS = {w: str(i) for i, ws in enumerate(
    [("null", "zero"), ("eins", "one"), ("zwei", "two"), ("drei", "three"), ("vier", "four"),
     ("fünf", "five"), ("sechs", "six"), ("sieben", "seven"), ("acht", "eight"),
     ("neun", "nine"), ("zehn", "ten")]) for w in ws}


def words(text: str) -> list[str]:
    """Lowercase words without punctuation; number words = digits ("two" and "2" both count)."""
    text = text.lower().replace("-", " ").replace("’", "'")
    return [NUMBERS.get(w, w) for w in re.findall(r"[\w']+", text)]


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str, str]]:
    """Levenshtein alignment -> [(op, ref_word, hyp_word)], op in = sub del ins."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            out.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            out.append(("ins", "", hyp[j - 1]))
            j -= 1
    return out[::-1]


def cased(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.replace("-", " ").replace("’", "'"))


def score(ref_text: str, hyp_text: str) -> dict:
    ops = align(words(ref_text), words(hyp_text))
    errors = [o for o in ops if o[0] != "="]
    # words that are right but wrongly capitalized ("Und" mid-sentence, "wir" for a noun, ...)
    ref_c, hyp_c, i, j, case = cased(ref_text), cased(hyp_text), 0, 0, 0
    for op, _, _ in ops:
        if op == "=":
            case += ref_c[i][:1] != hyp_c[j][:1]
        i += op != "ins"
        j += op != "del"
    return {"ref_words": len(words(ref_text)), "errors": len(errors), "case_errors": case,
            "invented_command": max(0, words(hyp_text).count("command")
                                    - words(ref_text).count("command")),
            "diffs": [f"{op}: {r} -> {h}" for op, r, h in errors]}


# ---------- audio ----------

def load(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def clips() -> dict[str, np.ndarray]:
    found = {cid: load(CORPUS_DIR / f"{cid}.wav") for cid, _ in CORPUS
             if (CORPUS_DIR / f"{cid}.wav").exists()}
    for k, (name, (order, gaps)) in enumerate(JOINED.items()):
        if all(o in found for o in order):
            rng = np.random.default_rng(100 + k)
            found[name] = np.concatenate(
                [x for o, g in zip(order, gaps)
                 for x in (*pause_inside(found[o], rng), np.zeros(int(g * SR), np.float32))])
    return found


def pause_inside(audio: np.ndarray, rng, pause_s: float = 2.5) -> tuple[np.ndarray, ...]:
    """Split a clip at a random gap between words and put a pause there: a mid-sentence
    thinking pause, where the VAD cuts a chunk (the hard case in real dictation)."""
    frames = audio[: len(audio) // 320 * 320].reshape(-1, 320)
    e = np.sqrt((frames ** 2).mean(1))
    quiet = [i * 320 for i in range(len(e))
             if 3 * SR < i * 320 < len(audio) - 3 * SR and e[i] < np.percentile(e, 25)]
    if not quiet:
        return (audio,)
    cut = int(rng.choice(quiet))
    return audio[:cut], np.zeros(int(pause_s * SR), np.float32), audio[cut:]


def highpass(audio: np.ndarray, hz: float = 80) -> np.ndarray:
    """Remove rumble below `hz` (desk knocks, fans, handling noise), smooth roll-off."""
    spec = np.fft.rfft(audio)
    f = np.fft.rfftfreq(len(audio), 1 / SR)
    spec *= np.clip((f - hz / 2) / (hz / 2), 0, 1)  # 0 below hz/2, ramp to 1 at hz
    return np.fft.irfft(spec, len(audio)).astype(np.float32)


def normalize(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2))) or 1.0
    return np.clip(audio * (target_rms / rms), -1, 1).astype(np.float32)


PRE = {"none": lambda a: a, "hp": highpass, "norm": normalize}


# ---------- worker: one model, several variants ----------

def worker(model_key: str, variants: list[dict]) -> None:
    import psutil

    from scripts.power import gpu_seconds
    from whisperclaude import transcriber as t
    from whisperclaude.commands import COMMANDS
    from whisperclaude.constants import DEFAULT_PROMPT, DEFAULT_VOCABULARY

    proc = psutil.Process()
    t.ov_repo = lambda m: MODELS[model_key]
    t0, c0 = time.perf_counter(), time.process_time()
    tr = t.OpenVinoTranscriber(model=model_key)
    load_s = time.perf_counter() - t0
    rss = proc.memory_info().rss / 2**30
    # idle cost with the model loaded (what the app costs while you don't dictate)
    g0, c0 = gpu_seconds(), time.process_time()
    time.sleep(30)
    idle = {"cpu_pct": (time.process_time() - c0) / 30 * 100,
            "gpu_pct": (gpu_seconds() - g0) / 30 * 100}
    emit({"model": model_key, "load_s": load_s, "rss_gb": rss, "idle": idle})

    examples = " " + " ".join(f"Command {c}." for c in COMMANDS)  # the app's prompt until 10-24
    prompts = {"app": t.build_prompt(DEFAULT_PROMPT, DEFAULT_VOCABULARY),
               "cmd_examples": t.build_prompt(DEFAULT_PROMPT + examples, DEFAULT_VOCABULARY),
               "none": None}
    real = tr.pipeline
    data = clips()
    vad_default = t.VadOptions
    app_groups = t._speech_groups  # the app's: merged within one language (with detect)

    def merge_groups(audio, *args, **kwargs):  # previous app behavior: merged blindly
        return app_groups(audio)

    def segment_groups(audio, *args, **kwargs):  # one chunk per passage
        return [[s] for g in app_groups(audio) for s in g]

    chunkers = {"app": app_groups, "merge": merge_groups, "segment": segment_groups}
    for v in variants:
        t._speech_groups = chunkers[v.get("chunk", "app")]
        t.LEAD_IN_S = v.get("lead", 0.5)
        class Pipe:  # the app's _generate with this variant's generation settings
            def get_generation_config(self, _v=v):
                cfg = real.get_generation_config()
                cfg.num_beams = _v.get("beams", 1)
                return cfg

            def generate(self, *a, **k):
                return real.generate(*a, **k)

        tr.pipeline = Pipe()
        tr.initial_prompt = prompts[v.get("prompt", "app")]
        pad = v.get("pad_ms")
        t.VadOptions = (vad_default if pad is None else
                        (lambda pad=pad, **k: vad_default(speech_pad_ms=pad, **k)))
        pre = PRE[v.get("pre", "none")]
        out, wall, cpu = {}, 0.0, 0.0
        g0 = gpu_seconds()
        try:
            tr.transcribe(data[next(iter(data))][:SR])
        except Exception as e:  # e.g. beam search isn't implemented on the GPU
            emit({"model": model_key, "variant_error": v, "error": str(e).splitlines()[-1]})
            continue
        for cid, audio in data.items():
            audio = pre(audio)
            t0, c0 = time.perf_counter(), time.process_time()
            text = tr.transcribe(audio)
            wall += time.perf_counter() - t0
            cpu += time.process_time() - c0
            out[cid] = {"text": text, "s": time.perf_counter() - t0, "audio_s": len(audio) / SR}
        audio_min = sum(len(a) for a in data.values()) / SR / 60
        emit({"model": model_key, "variant": v, "clips": out, "wall_s": wall,
              "cpu_s_per_min": cpu / audio_min, "gpu_s_per_min": (gpu_seconds() - g0) / audio_min})


def emit(obj: dict) -> None:
    print("RESULT " + json.dumps(obj), flush=True)  # ASCII: the pipe is not UTF-8


# ---------- report ----------

def report(results: list[dict]) -> None:
    models = {r["model"]: r for r in results if "load_s" in r}
    print(f"\n{'model':11s} {'load':>6s} {'RAM':>6s} {'idle CPU':>9s} {'idle GPU':>9s}")
    for m, r in models.items():
        print(f"{m:11s} {r['load_s']:5.1f}s {r['rss_gb']:5.2f}G {r['idle']['cpu_pct']:8.2f}% "
              f"{r['idle']['gpu_pct']:8.2f}%")
    print(f"\n{'model':11s} {'variant':28s} {'WER all':>8s} {'short':>6s} {'long':>6s} "
          f"{'joined':>7s} {'case':>5s} {'cmd+':>5s} {'s/10s':>6s} {'max s':>6s} {'CPU/min':>8s} "
          f"{'GPU/min':>8s}")
    for r in results:
        if "variant_error" in r:
            print(f"{r['model']:11s} {json.dumps(r['variant_error']):28s} FAILED: {r['error']}")
        if "variant" not in r:
            continue
        sc = {cid: score(REFS[cid], c["text"]) for cid, c in r["clips"].items()}
        r["scores"] = sc

        def wer(ids):
            ids = [i for i in ids if i in sc]
            n = sum(sc[i]["ref_words"] for i in ids)
            return 100 * sum(sc[i]["errors"] for i in ids) / n if n else float("nan")

        short = [c for c in sc if not c.startswith(("long", "joined"))]
        short_s = [r["clips"][c] for c in short]
        per10 = 10 * sum(c["s"] for c in short_s) / sum(c["audio_s"] for c in short_s)
        v = r["variant"]
        name = f"{v.get('prompt', 'app')} beam{v.get('beams', 1)} {v.get('pre', 'none')}" + (
            f" pad{v['pad_ms']}" if v.get("pad_ms") is not None else "") + (
            f" {v['chunk']}" if v.get("chunk") else "") + (
            f" lead{v['lead']}" if "lead" in v else "")
        print(f"{r['model']:11s} {name:28s} {wer([c for c in sc if not c.startswith('joined')]):7.1f}% "
              f"{wer(short):5.1f}% {wer(['long_de', 'long_mix']):5.1f}% "
              f"{wer([c for c in sc if c.startswith('joined')]):6.1f}% {sum(s['case_errors'] for s in sc.values()):5d} {sum(s['invented_command'] for s in sc.values()):5d} "
              f"{per10:5.2f}s {max(c['s'] for c in short_s):5.2f}s "
              f"{r['cpu_s_per_min']:7.2f}s {r['gpu_s_per_min']:7.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--variants", default="[]")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--beams", nargs="*", type=int, default=[1, 5])
    ap.add_argument("--prompts", nargs="*", default=["app", "cmd_examples"])
    ap.add_argument("--pre", nargs="*", default=["none"])
    ap.add_argument("--pad", nargs="*", type=int, default=[])
    ap.add_argument("--chunk", nargs="*", default=[], help="chunking variants: app merge segment")
    ap.add_argument("--lead", nargs="*", type=float, default=[], help="silence before each chunk")
    ap.add_argument("--report", action="store_true", help="only re-print the saved results")
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, json.loads(args.variants))
        sys.stdout.flush()
        os._exit(0)  # OpenVINO threads block a normal exit
    if args.report:
        report(json.loads(RESULTS.read_text(encoding="utf-8")))
        return
    variants = [{"prompt": p, "beams": b, "pre": pre} for p in args.prompts for b in args.beams
                for pre in args.pre] + [{"pad_ms": pad} for pad in args.pad] + [
        {"chunk": c, **({"pad_ms": pad} if pad else {})} for c in args.chunk
        for pad in (args.pad or [None])]
    if args.lead:  # cross every variant with each lead-in
        variants = [{**v, "lead": lead} for v in variants for lead in args.lead]
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else []
    for m in args.models:
        print(f"--- {m}: {len(variants)} variants", flush=True)
        p = subprocess.run([sys.executable, __file__, "--worker", m, "--variants",
                            json.dumps(variants)], capture_output=True, text=True, errors="replace",
                           encoding="utf-8", cwd=ROOT)
        got = [json.loads(line[7:]) for line in p.stdout.splitlines() if line.startswith("RESULT ")]
        if not got:
            print(p.stderr[-2000:])
        keys = {(g["model"], json.dumps(g.get("variant"), sort_keys=True)) for g in got}
        results = [r for r in results if (r["model"], json.dumps(r.get("variant"), sort_keys=True))
                   not in keys] + got
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report(results)


if __name__ == "__main__":
    main()
