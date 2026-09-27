"""Record a free-length test clip: Enter to start, Enter to stop.

.venv\\Scripts\\python.exe scripts\\record_clip.py long_de_1
-> bench_audio/long_de_1.wav (git-ignored)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark import AUDIO_DIR, SR, save_wav  # noqa: E402
from whisperclaude.recorder import Recorder  # noqa: E402

if len(sys.argv) != 2:
    sys.exit("usage: record_clip.py NAME")
path = AUDIO_DIR / f"{sys.argv[1]}.wav"
if path.exists() and input(f"{path.name} exists. Overwrite? [y/N] ").lower() != "y":
    sys.exit("aborted")

rec = Recorder()
input("Press Enter to START recording...")
rec.start()
t0 = time.monotonic()
input(">>> RECORDING. Speak, then press Enter to STOP.")
audio = rec.stop()

peak = float(np.abs(audio).max()) if len(audio) else 0.0
AUDIO_DIR.mkdir(exist_ok=True)
save_wav(path, audio)
print(f"Saved {path.name}: {len(audio) / SR:.1f} s, peak {peak:.2f}"
      + ("  (very quiet! check mic)" if peak < 0.05 else ""))
