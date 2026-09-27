from whisperclaude.overlay import level_to_unit
from whisperclaude.transcriber import build_prompt


def test_build_prompt_appends_vocabulary():
    assert build_prompt("Hallo, Test.", ["Claude", " VS Code "]) == "Hallo, Test. Claude, VS Code."
    assert build_prompt("Hallo, Test.", []) == "Hallo, Test."
    assert build_prompt(None, ["Claude"]) == "Claude."
    assert build_prompt("", []) is None


def test_level_to_unit_db_scale():
    assert level_to_unit(0.0) == 0.0            # silence
    assert level_to_unit(10 ** (-55 / 20)) == 0.0
    assert level_to_unit(0.1) == 1.0            # -20 dB and louder = full bar
    assert 0.3 < level_to_unit(0.01) < 0.6      # quiet laptop-mic speech ~ -40 dB
