"""Runs engines over your own recordings and compares them against your corrected texts.

.venv\\Scripts\\python.exe scripts\\real_eval.py run whisper                  # transcribe what is missing
.venv\\Scripts\\python.exe scripts\\real_eval.py run parakeet --torch-python C:\\path\\to\\venv\\Scripts\\python.exe
.venv\\Scripts\\python.exe scripts\\real_eval.py report [--show]             # the table (and the differences)

Recordings and references come from scripts/compare_gui.py (see scripts/real_data.py); only recordings with a
reference count. Outputs are cached per engine, so the report is instant and old engines stay comparable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import real_engines as engines # noqa: E402
from scripts.real_data import REAL_DIR, Recording, evaluate, load_hyps, load_recordings, markup, read_wav # noqa: E402

TABLE = f"{'engine':16s} {'recs':>4s} {'words':>6s} {'errors':>6s} {'WER':>6s} {'sub':>4s} {'del':>4s} {'ins':>4s} {'names':>9s} {'time/audio':>10s}"


def with_reference(directory: Path = REAL_DIR) -> list[Recording]:
    """The recordings that have a reference text."""
    return [r for r in load_recordings(directory) if r.reference.strip()]


def audio_seconds(rec: Recording) -> float:
    return len(read_wav(rec.wav)) / 16_000


def summarize(name: str, recordings: list[Recording], directory: Path = REAL_DIR) -> str | None:
    """One table row: the engine's errors over the recordings it has output for."""
    hyps = load_hyps(name, directory)
    rows = [(r, evaluate(r.reference, hyps[r.id]["text"])) for r in recordings if r.id in hyps]
    if not rows:
        return None
    total = {k: sum(e[k] for _, e in rows) for k in ("words", "errors", "sub", "del", "ins", "names", "names_wrong")}
    seconds = sum(hyps[r.id]["s"] for r, _ in rows)
    audio = sum(audio_seconds(r) for r, _ in rows)
    names = f"{total['names_wrong']}/{total['names']}"
    return (f"{name:16s} {len(rows):4d} {total['words']:6d} {total['errors']:6d} {100 * total['errors'] / max(total['words'], 1):5.1f}% "
            f"{total['sub']:4d} {total['del']:4d} {total['ins']:4d} {names:>9s} {seconds / audio:9.2f}x")


def report(recordings: list[Recording], show: bool, directory: Path = REAL_DIR) -> None:
    """Prints the comparison table, and with `show` the highlighted differences per recording."""
    names = sorted(p.stem for p in (directory / "hyp").glob("*.json")) if (directory / "hyp").exists() else []
    print(f"{len(recordings)} recordings with a reference ({sum(audio_seconds(r) for r in recordings):.0f} s of audio)\n{TABLE}")
    for name in names:
        row = summarize(name, recordings, directory)
        if row:
            print(row)
    if show:
        for rec in recordings:
            print(f"\n--- {rec.id}  {rec.note}\nREFERENCE: {rec.reference}")
            for name in names:
                hyp = load_hyps(name, directory).get(rec.id)
                if hyp:
                    pieces, missing = markup(rec.reference, hyp["text"])
                    text = " ".join(f"*{t.strip()}*" if kind != "ok" else t.strip() for t, kind in pieces)
                    print(f"{name:14s}: {text}" + (f"   (missing: {' '.join(missing)})" if missing else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("engines", nargs="*")
    parser.add_argument("--force", action="store_true", help="transcribe again even if cached")
    parser.add_argument("--torch-python")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    recordings = with_reference()
    if args.command == "run":
        for name in args.engines:
            engines.fill_cache(name, recordings, args.force, args.torch_python)
    report(recordings, args.show)


if __name__ == "__main__":
    main()
