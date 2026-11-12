"""Context-aware insertion: the pure decision, the last-paste fallback, the paste wrapper (all
with fakes), and the UIA reader's thread handling (UIA itself replaced by a stub)."""
import threading
import time

import pytest
from pynput.keyboard import Key, KeyCode

from whisperclaude import uia
from whisperclaude.spacing import (PasteTracker, SmartPaster, adapt, is_modifier, lowercase_first,
                                   mid_sentence, needs_space)

# ---------- needs_space ----------


@pytest.mark.parametrize("before, text, expected", [
    ("works.", "Next sentence", True), # the reported case
    ("and", "then", True),
    ("works.", "Next", True),
    ("3", "apples", True),
    ('said "', "x", True), # straight quotes can't be told apart: closing is the common case
    ("end)", "Next", True),
    ("hello,", "world", True),
    ("works. ", "Next", False), # already a space
    ("works.\n", "Next", False), # line start
    ("works.\r", "Next", False),
    ("a\t", "b", False),
    ("a ", "b", False),
    ("", "First", False), # start of the text
    (None, "First", False), # unknown: paste as is
    ("(", "note", False), # inside a bracket
    ("see [", "x", False),
    ("a/", "b", False),
    ("works.", ".", False), # the dictation starts with closing punctuation
    ("works", ", and", False),
    ("works", "!", False),
    ("works", "...", False),
    ("works", " Next", False), # already starts with a space
    ("works", "\nNext", False),
    ("works", "", False),
])
def test_needs_space(before, text, expected):
    assert needs_space(before, text) is expected


def test_only_the_last_character_counts():
    assert needs_space("some words and then", "x") is True
    assert needs_space("some words and then ", "x") is False


# ---------- case ----------

@pytest.mark.parametrize("before, expected", [
    ("so,", True), ("and", True), ("and ", True), ("and  \t", True), ("ünd", True),
    ("works.", False), ("works. ", False), ("works?", False), ("works:", False),
    ("works\n", False), ("and\n", False), ("and\n  ", False), # a line break ends the walk back
    ("", False), (None, False), ("   ", False), ("2024", False), ('"', False),
])
def test_mid_sentence(before, expected):
    assert mid_sentence(before) is expected


@pytest.mark.parametrize("text, expected", [
    ("The house is big.", "the house is big."),
    ("Und dann gingen wir.", "und dann gingen wir."),
    ("Dass es klappt.", "dass es klappt."),
    ("Haus und Garten.", "Haus und Garten."), # German noun
    ("I think so.", "I think so."),
    ("IT is great.", "IT is great."), # acronym
    ("Anna and Bob.", "Anna and Bob."), # name
    ("It's fine.", "it's fine."),
    ("the house", "the house"), # already lowercase
    ("123 apples", "123 apples"),
    ("", ""),
    ("...", "..."),
])
def test_lowercase_first(text, expected):
    assert lowercase_first(text) == expected


def test_adapt_combines_both_and_case_is_opt_in():
    assert adapt("The end.", "and") == " The end."
    assert adapt("The end.", "and", adapt_case=True) == " the end."
    assert adapt("The end.", "works.", adapt_case=True) == " The end."
    assert adapt("The end.", "", adapt_case=True) == "The end."
    assert adapt("The end.", None, adapt_case=True) == "The end."
    assert adapt("Haus.", "und ", adapt_case=True) == "Haus."


# ---------- fallback: the last paste ----------

class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def test_modifiers():
    for key in (Key.ctrl_l, Key.ctrl, Key.shift_r, Key.alt_l, Key.alt_gr, Key.cmd, Key.cmd_r):
        assert is_modifier(key)
    for key in (Key.esc, Key.enter, Key.backspace, Key.f9, Key.left, KeyCode.from_char("a"), None):
        assert not is_modifier(key)


def test_tracker_same_window_no_input():
    clock = Clock()
    tracker = PasteTracker(clock)
    assert tracker.guess_before(5) is None # nothing pasted yet
    tracker.record(5, "It works.")
    assert tracker.guess_before(5) == "It works."[-6:]
    assert tracker.guess_before(6) is None # another window
    assert tracker.guess_before(0) is None # no window known


def test_tracker_input_since_the_paste_forgets():
    tracker = PasteTracker(Clock())
    tracker.note_input() # before the paste: irrelevant
    tracker.record(5, "It works.")
    assert tracker.guess_before(5) is not None
    tracker.note_input() # a key or a click: the caret may have moved
    assert tracker.guess_before(5) is None
    tracker.record(5, "More.") # a new paste starts afresh
    assert tracker.guess_before(5) == "More."


def test_tracker_expires():
    clock = Clock()
    tracker = PasteTracker(clock)
    tracker.record(5, "x.")
    clock.now += PasteTracker.MAX_AGE_S - 1
    assert tracker.guess_before(5) == "x."
    clock.now += 2
    assert tracker.guess_before(5) is None


def test_tracker_empty_paste_is_nothing():
    tracker = PasteTracker(Clock())
    tracker.record(5, "")
    assert tracker.guess_before(5) is None


# ---------- SmartPaster ----------

class Rig:
    """A paste function that records, a reader with a scripted answer, a window."""

    def __init__(self, before=None, window=7, adapt_case=False):
        self.pasted, self.asked, self.window = [], [], window
        self.before = before
        self.tracker = PasteTracker(Clock())
        self.paster = SmartPaster(self.pasted.append, self.read, self.tracker, adapt_case,
                                  lambda: self.window)

    def read(self, n):
        self.asked.append(n)
        if isinstance(self.before, Exception):
            raise self.before
        return self.before


def test_paster_uses_uia_when_it_answers():
    rig = Rig(before="d")
    rig.paster("then")
    assert rig.pasted == [" then"]
    rig.before = " "
    rig.paster("then")
    assert rig.pasted[-1] == "then"
    rig.before = "" # start of the text: the answer counts, the last paste does not
    rig.paster("Start")
    assert rig.pasted[-1] == "Start"


def test_paster_falls_back_to_the_last_paste():
    rig = Rig(before=None)
    rig.paster("First sentence.") # nothing known: as is
    rig.paster("Second sentence.") # same window, no input: right behind the first
    assert rig.pasted == ["First sentence.", " Second sentence."]
    rig.tracker.note_input() # the user clicked or typed
    rig.paster("Third.")
    assert rig.pasted[-1] == "Third."
    rig.paster("Fourth.")
    rig.window = 8 # another window
    rig.paster("Fifth.")
    assert rig.pasted[-1] == "Fifth."


def test_fallback_respects_a_trailing_line_break():
    rig = Rig(before=None)
    rig.paster("Item one.\n")
    rig.paster("Item two.")
    assert rig.pasted[-1] == "Item two."


def test_uia_answer_wins_over_the_last_paste():
    rig = Rig(before=None)
    rig.paster("First.")
    rig.before = " " # the user's caret is after a space, the field says so
    rig.paster("Second.")
    assert rig.pasted[-1] == "Second."


def test_reads_only_as_much_as_needed():
    rig = Rig(before="x")
    rig.paster("a")
    assert rig.asked == [1]
    rig = Rig(before="x", adapt_case=True)
    rig.paster("a")
    assert rig.asked[0] > 1


def test_case_adaptation_through_the_paster():
    rig = Rig(before="so, ", adapt_case=True)
    rig.paster("The plan works.")
    assert rig.pasted == ["the plan works."]


def test_any_failure_pastes_unchanged():
    rig = Rig(before=RuntimeError("boom"))
    rig.paster("Text.")
    assert rig.pasted == ["Text."]
    rig = Rig(before="x")
    rig.paster.window = lambda: 1 / 0
    rig.paster("Text.")
    assert rig.pasted == ["Text."]


def test_a_failing_paste_is_raised_and_not_remembered():
    rig = Rig(before=None)

    def fail(text):
        raise OSError("clipboard busy")

    rig.paster.paste = fail
    with pytest.raises(OSError):
        rig.paster("Text.")
    assert rig.tracker.guess_before(7) is None


# ---------- the UIA reader's thread handling ----------

@pytest.fixture
def stub_uia(monkeypatch):
    """No real UIA: the read is a function we control."""
    monkeypatch.setattr(uia._ole32, "CoInitializeEx", lambda *a: 0)
    monkeypatch.setattr(uia, "_out_ptr", lambda call, *args: 1)
    calls = {"n": 0, "answer": "x", "gate": None}

    def read(automation, n):
        calls["n"] += 1
        if calls["gate"] is not None:
            calls["gate"].wait(5)
        if isinstance(calls["answer"], Exception):
            raise calls["answer"]
        return calls["answer"]

    monkeypatch.setattr(uia, "_range_before_caret", read)
    return calls


def test_reader_returns_the_answer(stub_uia):
    reader = uia.CaretReader()
    assert reader.text_before(1, timeout=2) == "x"
    stub_uia["answer"] = None
    assert reader.text_before(1, timeout=2) is None


def test_reader_survives_an_exception(stub_uia):
    reader = uia.CaretReader()
    stub_uia["answer"] = RuntimeError("boom")
    assert reader.text_before(1, timeout=2) is None
    stub_uia["answer"] = "y"
    assert reader.text_before(1, timeout=2) == "y"


def test_a_hanging_app_costs_one_timeout_and_requests_do_not_pile_up(stub_uia):
    reader = uia.CaretReader()
    gate = stub_uia["gate"] = threading.Event()
    t0 = time.perf_counter()
    assert reader.text_before(1, timeout=0.05) is None # the app doesn't answer
    assert time.perf_counter() - t0 < 1
    t0 = time.perf_counter()
    assert reader.text_before(1, timeout=0.05) is None # refused at once: still busy
    assert time.perf_counter() - t0 < 0.03
    assert stub_uia["n"] == 1 # the second request never reached the thread
    gate.set() # the app answers at last
    deadline = time.time() + 2
    while reader._outstanding and time.time() < deadline:
        time.sleep(0.01)
    stub_uia["gate"] = None
    assert reader.text_before(1, timeout=2) == "x" # and the reader works again
