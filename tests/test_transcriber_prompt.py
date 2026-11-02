import pytest

from whisperclaude.overlay import LevelMeter
from whisperclaude.transcriber import build_prompt


def test_build_prompt_appends_vocabulary():
    assert build_prompt("Hallo, Test.", ["Claude", " VS Code "]) == "Hallo, Test. Claude, VS Code."
    assert build_prompt("Hallo, Test.", []) == "Hallo, Test."
    assert build_prompt(None, ["Claude"]) == "Claude."
    assert build_prompt("", []) is None


def _speak(meter, speech_db, noise_db=-66.0, seconds=6.0):
    """Syllables (4 per second, louder in their middle) in phrases with pauses, over room
    noise; returns the meter's output per frame."""
    import math
    import random

    rng, out = random.Random(1), []
    for frame in range(int(seconds * 30)):
        t = frame / 30
        in_phrase = (t % 2.0) < 1.4
        syllable = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 4)
        db = speech_db - 14 * (1 - syllable) if in_phrase else noise_db
        out.append(meter(10 ** ((db + rng.uniform(-1.5, 1.5)) / 20)))
    return out


def test_level_meter_lifts_quiet_speech_but_keeps_loud_louder():
    fixed = lambda db: min(1.0, max(0.0, (db + 55) / 35))  # the old fixed scale
    quiet = max(_speak(LevelMeter(), -52)[30:])
    medium = max(_speak(LevelMeter(), -45)[30:])
    loud = max(_speak(LevelMeter(), -22)[30:])
    assert fixed(-52) < 0.1 and quiet > 0.4  # was nearly flat
    assert fixed(-45) < 0.3 and medium > 0.6
    assert loud > 0.95 and quiet < medium < loud  # still louder = higher


def test_level_meter_keeps_silence_flat():
    meter = LevelMeter()
    silence = [meter(10 ** ((-66 + d) / 20)) for d in (0, 2, -2, 1, -1) * 60]
    assert max(silence) < 0.15
    assert LevelMeter()(0.0) == 0.0


@pytest.mark.parametrize("text, kind", [
    ("Error: no microphone", "error"), ("Could not load model – see log", "error"),
    ("Copy failed – clipboard busy, try again", "error"),
    ("Config error – using defaults (see log)", "error"),
    ("Copied – paste with Ctrl+V", "ok"),
    ("Cancelled", "stopped"), ("Too short", "info"), ("Nothing recognized", "info"),
    ("Not pasted (the focus moved)", "info"), ("Translation failed: model broken", "error"),
])
def test_message_badges(text, kind):
    from whisperclaude.overlay import message_kind
    assert message_kind(text) == kind
