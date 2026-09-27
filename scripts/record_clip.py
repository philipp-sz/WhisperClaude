"""Record a free-length test clip: Enter to start, Enter to stop.

.venv\\Scripts\\python.exe scripts\\record_clip.py long_de_1
-> bench_audio/long_de_1.wav
.venv\\Scripts\\python.exe scripts\\record_clip.py speech_de --dir tests/data --say "Guten Morgen."
-> tests/data/speech_de.wav (clip for the slow tests)

All *.wav files are git-ignored, so your voice is never committed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark import AUDIO_DIR, SR, save_wav  # noqa: E402
from whisperclaude.recorder import Recorder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name", help="file name without .wav")
ap.add_argument("--dir", type=Path, default=AUDIO_DIR, help="target folder (default bench_audio)")
ap.add_argument("--say", help="sentence to show for reading aloud")
args = ap.parse_args()

path = args.dir / f"{args.name}.wav"
if path.exists() and input(f"{path.name} exists. Overwrite? [y/N] ").lower() != "y":
    sys.exit("aborted")
if args.say:
    print(f"\nRead this aloud:\n\n   {args.say}\n")

rec = Recorder()
input("Press Enter to START recording...")
rec.start()
input(">>> RECORDING. Speak, then press Enter to STOP.")
audio = rec.stop()

peak = float(np.abs(audio).max()) if len(audio) else 0.0
path.parent.mkdir(parents=True, exist_ok=True)
save_wav(path, audio)
print(f"Saved {path}: {len(audio) / SR:.1f} s, peak {peak:.2f}"
      + ("  (very quiet! check mic)" if peak < 0.05 else ""))
