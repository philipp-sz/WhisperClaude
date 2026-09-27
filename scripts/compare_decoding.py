"""Compare decoding options (timestamps, initial_prompt, batched) on recorded clips.

.venv\\Scripts\\python.exe scripts\\compare_decoding.py long_de_1 long_en_1
Prints each variant's time and full text so they can be compared side by side.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark import AUDIO_DIR, SR, load_wav  # noqa: E402
from whisperclaude.transcriber import DEFAULT_PROMPT, Transcriber  # noqa: E402

# (label, use batched pipeline, extra kwargs)
VARIANTS = [
    ("A timestamps", False, {}),
    ("B without_timestamps", False, {"without_timestamps": True}),
    ("C timestamps + prompt", False, {"initial_prompt": DEFAULT_PROMPT}),
    ("D without_ts + prompt", False, {"without_timestamps": True, "initial_prompt": DEFAULT_PROMPT}),
    ("E batched + prompt (current)", True, {"without_timestamps": True,
                                            "initial_prompt": DEFAULT_PROMPT, "batch_size": 4}),
]


def main():
    names = sys.argv[1:] or sorted(p.stem for p in AUDIO_DIR.glob("long_*.wav"))
    tr = Transcriber()
    tr.transcribe(load_wav(AUDIO_DIR / f"{names[0]}.wav"))  # warm-up

    for name in names:
        audio = load_wav(AUDIO_DIR / f"{name}.wav")
        print(f"\n{'=' * 100}\n{name}  ({len(audio) / SR:.1f} s)")
        for label, batched, extra in VARIANTS:
            engine = tr.pipeline if batched else tr.model
            t0 = time.perf_counter()
            segments, info = engine.transcribe(
                audio, beam_size=tr.beam_size, vad_filter=True, language=None, **extra
            )
            segs = list(segments)
            dt = time.perf_counter() - t0
            print(f"\n--- {label}: {dt:.2f} s, lang {info.language}, {len(segs)} segments")
            for s in segs:
                print(f"  [{s.start:5.1f}-{s.end:5.1f}] {s.text.strip()}")


if __name__ == "__main__":
    main()
