import json
import time

from whisperclaude.history import Entry, History


def test_keeps_the_newest_entries_and_survives_a_restart(tmp_path):
    h = History(tmp_path, keep=3)
    for text in ("eins", "zwei", "drei", "vier"):
        h.add(text)
    assert [e.text for e in h.entries()] == ["vier", "drei", "zwei"]
    assert h.latest().text == "vier"
    again = History(tmp_path, keep=3)  # e.g. after an app restart
    assert [e.text for e in again.entries()] == ["vier", "drei", "zwei"]


def test_keep_zero_turns_it_off_and_deletes_old_dictations(tmp_path):
    History(tmp_path, keep=5).add("geheim")
    assert (tmp_path / "history.json").exists()
    off = History(tmp_path, keep=0)
    off.add("nicht speichern")
    assert off.entries() == [] and not (tmp_path / "history.json").exists()


def test_clear_and_empty_text(tmp_path):
    h = History(tmp_path)
    h.add("   ")
    assert h.entries() == []
    h.add("text")
    h.clear()
    assert h.entries() == [] and not (tmp_path / "history.json").exists()


def test_broken_file_starts_fresh(tmp_path):
    (tmp_path / "history.json").write_text("{not json", encoding="utf-8")
    h = History(tmp_path)
    assert h.entries() == []
    h.add("neu")
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))[0]["text"] == "neu"


def test_menu_labels():
    now = time.time()
    short = Entry(now, "Kurzer Text.")
    assert short.label().endswith("· Kurzer Text.")
    long = Entry(now, "Ich glaube, ich würde es jetzt auch ein bisschen anders machen.\n" * 200)
    label = long.label()
    assert "…" in label and "\n" not in label and "(12,800 characters)" in label
    old = Entry(now - 3 * 86400, "Alt")
    assert time.strftime("%d.%m.", time.localtime(old.time)) in old.label()
