"""The translation model's on-demand loading and idle unload (fake model, fake clock)."""
import threading
import time

import pytest

from whisperclaude.translator import LazyModel, Translator


class FakeCT2:
    """The part of ctranslate2.Translator that LazyModel uses."""

    def __init__(self, gate=None):
        self.gate = gate
        self.events = []

    def load_model(self):
        self.events.append("load")

    def unload_model(self):
        self.events.append("unload")

    def translate_batch(self, batch, **kwargs):
        class Result:
            hypotheses = [["deu_Latn", "x"]]
        return [Result() for _ in batch]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(unload_after_s=600.0, gate=None):
    """(LazyModel, the factory's call log, clock); the watcher thread is off."""
    calls, clock = [], Clock()

    def factory():
        if gate is not None:
            gate.wait(5)
        calls.append("build")
        return FakeCT2()

    return LazyModel(factory, unload_after_s, clock, watch=False), calls, clock


def use(lazy):
    with lazy.use() as model:
        return model


def test_first_use_builds_once_later_uses_do_not_load_again():
    lazy, calls, _ = make()
    assert not lazy.loaded
    m = use(lazy)
    assert use(lazy) is m and calls == ["build"] and lazy.loaded


@pytest.mark.parametrize("unload_after_s, idle_s, unloaded", [
    (600, 599, False), (600, 600, True), (600, 3600, True), (0, 10**6, False), # 0 = never
])
def test_unloads_only_after_the_idle_time(unload_after_s, idle_s, unloaded):
    lazy, calls, clock = make(unload_after_s)
    model = use(lazy)
    clock.now += idle_s
    assert lazy.unload_if_idle() is unloaded
    assert lazy.loaded is not unloaded
    assert model.events == (["unload"] if unloaded else [])


def test_use_and_a_started_load_restart_the_idle_time():
    lazy, _, clock = make(600)
    use(lazy)
    clock.now += 500
    use(lazy)
    clock.now += 500
    assert not lazy.unload_if_idle() # 500 s since the last use
    clock.now += 99
    lazy.start_load() # the translate key was tapped
    clock.now += 500
    assert not lazy.unload_if_idle()


def test_no_unload_during_a_translation():
    lazy, _, clock = make(600)
    with lazy.use() as model:
        clock.now += 3600 # a very long translation
        t = threading.Thread(target=lazy.unload_if_idle)
        t.start()
        t.join(0.2)
        assert t.is_alive() and lazy.loaded and model.events == [] # waits for the lock
    t.join(5)
    assert lazy.loaded and model.events == [] # the translation just ended: not idle


def test_use_after_an_unload_reloads_the_same_model():
    lazy, calls, clock = make(600)
    model = use(lazy)
    clock.now += 600
    assert lazy.unload_if_idle()
    assert use(lazy) is model and lazy.loaded
    assert model.events == ["unload", "load"] and calls == ["build"]


def test_concurrent_start_load_loads_once():
    gate = threading.Event()
    lazy, calls, _ = make(gate=gate)
    started = []
    threads = [threading.Thread(target=lambda: started.append(lazy.start_load()))
               for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert started.count(True) == 1 # the others saw it loading
    waiter = threading.Thread(target=use, args=(lazy,)) # translate() while it loads
    waiter.start()
    time.sleep(0.1)
    assert waiter.is_alive() and not lazy.loaded
    gate.set()
    waiter.join(5)
    assert calls == ["build"] and lazy.loaded
    assert lazy.start_load() is False # loaded: nothing to start


def test_a_failed_load_is_logged_in_the_background_and_raised_to_the_translation():
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("disk gone")
        return FakeCT2()

    lazy = LazyModel(factory, 600, Clock(), watch=False)
    lazy.start_load() # fails on its thread (logged), not raised
    deadline = time.monotonic() + 5
    while not attempts or lazy._loading:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with pytest.raises(OSError, match="disk gone"):
        use(lazy) # translate() sees the error
    assert use(lazy) is not None and len(attempts) == 3 # and the next one retries


def test_idle_watcher_unloads_by_itself():
    clock, model = Clock(), FakeCT2()
    lazy = LazyModel(lambda: model, 0.2, time.monotonic)
    use(lazy)
    deadline = time.monotonic() + 5
    while lazy.loaded and time.monotonic() < deadline:
        time.sleep(0.05)
    lazy.close()
    assert not lazy.loaded and model.events == ["unload"]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("E", (), {"ids": [5]})()

    def id_to_token(self, i):
        return "tok"

    def token_to_id(self, t):
        return 7

    def decode(self, ids, skip_special_tokens=True):
        return "Hello"


def fake_translator(lazy):
    from whisperclaude.translator import LanguageDetector
    t = Translator.__new__(Translator) # no model files, no downloads
    t.tokenizer, t.beam_size, t._lazy = FakeTokenizer(), 2, lazy
    t.detector = LanguageDetector(["de", "en"])
    return t


def test_translate_after_an_unload_works_and_prepare_reloads():
    lazy, calls, clock = make(600)
    t = fake_translator(lazy)
    assert t.translate("Guten Morgen, wie geht es dir heute?", "eng_Latn") == "Hello"
    clock.now += 601
    assert lazy.unload_if_idle() and not lazy.loaded
    t.prepare() # the key tap
    t.translate("Guten Morgen, wie geht es dir heute?", "eng_Latn") # waits for that load
    assert lazy.loaded and calls == ["build"] and use(lazy).events == ["unload", "load"]
    lazy.unload_if_idle() # not idle yet
    assert lazy.loaded
