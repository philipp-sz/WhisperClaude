"""Recordings with references, name marks, scoring and the engine caches of scripts/real_data.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import real_data as rd # noqa: E402


def test_name_marks_are_removed_and_the_words_flagged():
    text, flags = rd.parse_reference("Ich habe mit [[Claude Code]] am [[Projekt]] gearbeitet.")
    assert text == "Ich habe mit Claude Code am Projekt gearbeitet."
    assert flags == [False, False, False, True, True, False, True, False]


def test_evaluate_counts_substitutions_deletions_insertions_and_names():
    result = rd.evaluate("Ich habe mit [[Claude]] gearbeitet", "Ich habe mit Cloud sehr gearbeitet")
    assert (result["sub"], result["del"], result["ins"], result["errors"]) == (1, 0, 1, 2)
    assert (result["names"], result["names_wrong"], result["words"]) == (1, 1, 5)
    deleted = rd.evaluate("Ich habe mit [[Claude]] gearbeitet", "Ich habe mit gearbeitet")
    assert (deleted["del"], deleted["names_wrong"]) == (1, 1)


def test_case_punctuation_hyphens_and_number_words_do_not_count():
    result = rd.evaluate("Das ist zwei Mal gut.", "das IST 2 mal gut")
    assert result["errors"] == 0
    assert rd.evaluate("Ein Rück-Ruf", "ein rück ruf")["errors"] == 0


def test_a_perfect_name_is_counted_but_not_wrong():
    result = rd.evaluate("Hallo [[Müller]]", "Hallo Müller.")
    assert (result["names"], result["names_wrong"], result["errors"]) == (1, 0, 0)


def test_markup_marks_a_wrong_word_an_invented_word_and_lists_a_missing_one():
    reference = "Wir gehen heute ins Kino"
    assert rd.markup(reference, "Wir gehen morgen ins Kino")[0][2] == (" morgen", "sub")
    assert rd.markup(reference, "Wir gehen heute ins Kino ja")[0][-1] == (" ja", "ins")
    pieces, missing = rd.markup(reference, "Wir gehen ins Kino")
    assert [kind for _, kind in pieces] == ["ok"] * 4 and missing == ["heute"]


def test_markup_keeps_the_original_text_pieces():
    pieces, _ = rd.markup("Hallo Welt", "Hallo, Welt!\nNeu")
    assert "".join(text for text, _ in pieces) == "Hallo, Welt!\nNeu"
    assert pieces[-1] == ("\nNeu", "ins")


def test_recordings_are_saved_listed_and_get_their_reference(tmp_path):
    audio = (0.3 * np.sin(np.arange(16_000) / 10)).astype(np.float32)
    rec = rd.save_recording(audio, tmp_path)
    assert rec.wav.exists() and rec.meta.exists() and rec.reference == ""
    assert np.allclose(rd.read_wav(rec.wav), audio, atol=1e-4)
    rd.save_reference(rec, "Das ist [[Claude]].", note="Notiz")
    (loaded,) = rd.load_recordings(tmp_path)
    assert (loaded.id, loaded.reference, loaded.note) == (rec.id, "Das ist [[Claude]].", "Notiz")


def test_a_wav_without_a_json_is_still_listed(tmp_path):
    rd.write_wav(tmp_path / "x.wav", np.zeros(100, np.float32))
    (rec,) = rd.load_recordings(tmp_path)
    assert (rec.id, rec.reference) == ("x", "")


def test_engine_caches_are_kept_per_engine_and_recording(tmp_path):
    rd.save_hyp("whisper", "a", "Text eins", 0.4, tmp_path)
    rd.save_hyp("whisper", "b", "Text zwei", 0.5, tmp_path)
    rd.save_hyp("vote:whisper", "a", "Vote", 0.0, tmp_path)
    assert rd.load_hyps("whisper", tmp_path) == {"a": {"text": "Text eins", "s": 0.4}, "b": {"text": "Text zwei", "s": 0.5}}
    assert rd.load_hyps("vote:whisper", tmp_path)["a"]["text"] == "Vote" # ":" is fine in an engine name
    assert rd.load_hyps("granite", tmp_path) == {}


@pytest.mark.parametrize("reference, hypothesis, errors", [("", "", 0), ("", "Hallo", 1), ("Hallo", "", 1)])
def test_empty_texts(reference, hypothesis, errors):
    assert rd.evaluate(reference, hypothesis)["errors"] == errors


def test_a_reader_never_sees_a_half_written_cache(tmp_path):
    import threading
    import time

    stop, errors = threading.Event(), []

    def write():
        for i in range(60):
            rd.save_hyp("whisper", f"r{i}", "text " * 200, 0.1, tmp_path)
        stop.set()

    writer = threading.Thread(target=write)
    writer.start()
    while not stop.is_set():
        try:
            rd.load_hyps("whisper", tmp_path)
        except ValueError as e: # JSONDecodeError
            errors.append(e)
        time.sleep(0.001)
    writer.join()
    assert not errors
