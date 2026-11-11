"""Automatic vocabulary (vocabulary.py): candidates, weighting, caps and the token budget, with a fake tokenizer."""
import os
import time

import pytest

from whisperclaude import vocabulary as voc

COMMON = {"the", "and", "results", "test", "names", "translation"}


def pieces(text):
    """Fake tokenizer: common words are one token, other words one token per 3 letters, plus 1 per space."""
    count = 0
    for word in text.split():
        count += 1 if word.lower().strip(".,") in COMMON else max(1, len(word) // 3)
    return count


def write(path, text, age_days=0.0):
    path.write_text(text, encoding="utf-8")
    old = time.time() - age_days * 86_400
    os.utime(path, (old, old))


def test_candidates_skip_code_links_and_acronyms(tmp_path):
    """Test that only capitalized words in prose count, not code, link targets or all-caps acronyms."""
    write(tmp_path / "a.md", "Claude Code and OpenVINO. `Hidden` [link](Target) https://Example.com GPU VAD")
    counts = voc.weighted_counts([tmp_path / "a.md"])
    assert set(counts) == {"Claude Code", "OpenVINO"}


def test_old_files_count_less(tmp_path):
    """Test that a file's count halves every HALF_LIFE_DAYS."""
    write(tmp_path / "new.md", "Parakeet")
    write(tmp_path / "old.md", "Granite", age_days=voc.HALF_LIFE_DAYS + 1)
    counts = voc.weighted_counts(sorted(tmp_path.glob("*.md")))
    assert counts["Parakeet"] == pytest.approx(1.0, abs=0.01)
    assert counts["Granite"] == pytest.approx(0.5, abs=0.01)


def test_files_and_skipped_folders(tmp_path):
    """Test that only .md/.txt outside tool folders are read and missing sources are ignored."""
    (tmp_path / ".venv").mkdir()
    (tmp_path / "notes").mkdir()
    for name in ("notes/a.md", "notes/b.txt", "notes/c.py", ".venv/d.md"):
        (tmp_path / name).write_text("x")
    found = {p.name for p in voc.text_files([tmp_path, tmp_path / "missing"])}
    assert found == {"a.md", "b.txt"}


def test_file_caps(tmp_path, monkeypatch):
    """Test that reading stops at MAX_FILES and skips files above MAX_FILE_BYTES."""
    monkeypatch.setattr(voc, "MAX_FILES", 2)
    monkeypatch.setattr(voc, "MAX_FILE_BYTES", 100)
    write(tmp_path / "1.md", "Bijektion " * 20) # too big
    write(tmp_path / "2.md", "Hoare")
    write(tmp_path / "3.md", "Gemini") # beyond MAX_FILES
    counts = voc.weighted_counts(sorted(tmp_path.glob("*.md")))
    assert set(counts) == {"Hoare"}


def test_rank_keeps_frequent_unknown_words():
    """Test that rare-to-Whisper, frequent words rank first and common or rare-in-text words are dropped."""
    counts = {"Teilaufgabe": 10.0, "Results": 50.0, "Bijektion": 4.0, "Hoare": 2.0, "OpenVINO": 6.0}
    assert voc.rank(counts, pieces) == ["Teilaufgabe", "Bijektion", "OpenVINO"] # count x extra pieces: 20, 8, 6


def test_rank_caps_the_number_of_words():
    """Test that at most max_words come back however many qualify."""
    counts = {f"Wordnumber{i:03d}": 5.0 for i in range(100)}
    assert len(voc.rank(counts, pieces, max_words=40)) == 40


def test_budget_drops_the_weakest_words():
    """Test that the prompt never exceeds the token budget and manual words are not repeated."""
    auto = [f"Specialname{i:02d}" for i in range(50)]
    words = voc.fit_budget("Hallo, das ist ein Test.", ["Claude", "specialname00"], auto, pieces, budget=60)
    prompt = f"Hallo, das ist ein Test. {', '.join(['Claude', 'specialname00', *words])}."
    assert pieces(prompt) <= 60 and "Specialname00" not in words and words == auto[1:len(words) + 1]


def test_auto_vocabulary_is_off_without_sources_and_never_raises(tmp_path):
    """Test that no sources means no words and a failing tokenizer only logs."""
    assert voc.auto_vocabulary([], "", [], pieces) == []
    write(tmp_path / "a.md", "Bijektion. Bijektion. Bijektion.")
    def broken(text):
        raise RuntimeError("tokenizer missing")
    assert voc.auto_vocabulary([tmp_path], "", [], broken) == []
    assert voc.auto_vocabulary([tmp_path], "", [], pieces) == ["Bijektion"]
