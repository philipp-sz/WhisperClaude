"""LiveTranscription (transcribing while recording) with a fake VAD and a fake model."""
import numpy as np
import pytest

from whisperclaude import transcriber as t

SR = t.SAMPLE_RATE


def fake_vad(audio, options=None):
    """Speech = runs of non-zero samples (the real VAD adds padding; not needed here)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (audio != 0).astype(int), [0]])))
    return [{"start": int(a), "end": int(b)} for a, b in zip(edges[::2], edges[1::2])]


def fake_model(chunks):
    """'Transcribes' each burst as w<id>: its samples all carry the burst's id."""
    return " ".join(" ".join(f"w{round(v)}" for v in np.unique(c) if v) for c in chunks)


def dictation(n_bursts=12, speech_s=8, pause_s=3):
    """Bursts of 'speech' with pauses: 3 bursts fit in one 30 s chunk."""
    parts = []
    for k in range(1, n_bursts + 1):
        parts += [np.full(speech_s * SR, k, np.float32), np.zeros(pause_s * SR, np.float32)]
    return np.concatenate(parts)


@pytest.fixture(autouse=True)
def vad(monkeypatch):
    monkeypatch.setattr(t, "get_speech_timestamps", fake_vad)


def test_speech_chunks_merge_passages_up_to_30_s():
    chunks = t.speech_chunks(dictation())
    assert [len(c) / SR for c in chunks] == [24 + t.LEAD_IN_S] * 4
    lead = int(t.LEAD_IN_S * SR)  # each chunk starts with silence, then the speech
    assert all(not c[:lead].any() and c[lead:].all() for c in chunks)


def language_of_burst(audio):
    """Fake language detection: odd bursts are German, even ones English."""
    return "de" if round(audio.max()) % 2 else "en"


def test_passages_in_different_languages_are_never_merged():
    """Whisper uses one language per chunk: a merged German + English chunk got translated."""
    calls = []
    detect = lambda a: calls.append(1) or language_of_burst(a)  # noqa: E731
    chunks = t.speech_chunks(dictation(n_bursts=4), detect)
    assert [len(c) / SR for c in chunks] == [8 + t.LEAD_IN_S] * 4  # de / en / de / en

    same = dictation(n_bursts=4) * 0 + np.where(dictation(n_bursts=4) > 0, 1, 0)  # all "de"
    assert [len(c) / SR for c in t.speech_chunks(same.astype(np.float32), detect)] == [
        24 + t.LEAD_IN_S, 8 + t.LEAD_IN_S]  # same language: merged up to 30 s as before


def test_a_single_passage_needs_no_language_detection():
    calls = []
    t.speech_chunks(dictation(n_bursts=1), lambda a: calls.append(1) or "de")
    assert calls == []  # short dictations cost no extra GPU pass


def test_live_detects_each_passage_only_once():
    audio = dictation()
    detected = []
    live = t.LiveTranscription(fake_model, lambda a: detected.append(round(a.max())) or
                               language_of_burst(a))
    for end in range(5 * SR, len(audio), 5 * SR):
        live.feed(audio[:end])
    assert live.finish(audio) == " ".join(f"w{k}" for k in range(1, 13))
    assert len(detected) == len(set(detected)) == 12


def test_live_gives_the_same_text_and_leaves_little_for_the_stop():
    audio = dictation()
    calls = []
    live = t.LiveTranscription(lambda chunks: calls.append(len(chunks)) or fake_model(chunks))
    for end in range(5 * SR, len(audio), 5 * SR):  # what the feeder sees while recording
        live.feed(audio[:end])
    assert live.done_s > 90  # most of the 132 s were transcribed before the stop
    n_before_stop = len(calls)
    text = live.finish(audio)
    assert text == fake_model(t.speech_chunks(audio)) == " ".join(f"w{k}" for k in range(1, 13))
    assert calls[n_before_stop:] == [1]  # the stop only had the last chunk left


def test_live_short_recording_does_everything_at_the_stop():
    audio = dictation(n_bursts=2, speech_s=4)  # 14 s: below LIVE_MIN_TAIL_S
    live = t.LiveTranscription(fake_model)
    live.feed(audio)
    assert live.done_s == 0
    assert live.finish(audio) == "w1 w2"


def test_live_never_takes_the_newest_audio():
    """Nothing is decided about a passage that is still being spoken."""
    def burst(k, s):
        return np.full(s * SR, k, np.float32)

    def pause(s):
        return np.zeros(s * SR, np.float32)

    # chunk 1 = bursts 1+2 (28 s of speech), ending at 33 s; burst 3: 34-44 s, burst 4: 47-52 s
    audio = np.concatenate([burst(1, 20), pause(5), burst(2, 8), pause(1), burst(3, 10),
                            pause(3), burst(4, 5)])
    live = t.LiveTranscription(fake_model)
    live.feed(audio[:40 * SR])
    assert live.done_s == 0  # burst 3 is still being spoken: it might still join chunk 1
    live.feed(audio[:50 * SR])
    assert live.done_s == 33  # burst 3 is finished and doesn't fit: chunk 1 is final
    assert live.finish(audio) == "w1 w2 w3 w4"


def test_live_failed_chunk_is_retried_at_the_stop():
    audio = dictation(n_bursts=6)
    fail = [True]

    def flaky(chunks):
        if fail[0]:
            raise RuntimeError("GPU hiccup")
        return fake_model(chunks)

    live = t.LiveTranscription(flaky)
    with pytest.raises(RuntimeError):
        live.feed(audio)
    fail[0] = False
    assert live.finish(audio) == " ".join(f"w{k}" for k in range(1, 7))


def test_silence_gives_empty_text():
    live = t.LiveTranscription(fake_model)
    silence = np.zeros(60 * SR, np.float32)
    live.feed(silence)
    assert live.finish(silence) == ""
