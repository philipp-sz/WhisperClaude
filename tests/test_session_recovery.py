"""The Session's GPU recovery wiring (__main__): first load, upgrade when idle, notice, background reload."""
from types import SimpleNamespace

import pytest

from whisperclaude import __main__ as main
from whisperclaude import recovery
from whisperclaude.app import State


class Recorder:
    """Records method calls on a stand-in for the tray or the overlay."""

    def __init__(self):
        self.calls, self.engine = [], ""

    def __getattr__(self, name):
        return lambda *args, **kw: self.calls.append((name, args))


def session(tmp_path, state=State.LOADING, last="igpu"):
    s = main.Session.__new__(main.Session)
    s.data_dir, s.config_error, s.hotkey, s.events = tmp_path, "", "Left Ctrl", None
    s.cfg = SimpleNamespace(hotkey=SimpleNamespace(key="ctrl_l"), model=None)
    s.tray, s.overlay, s.upgrade, s.watch = Recorder(), Recorder(), None, recovery.GpuWatch()
    swapped = []
    s.app = SimpleNamespace(state=state, set_transcriber=swapped.append)
    if last:
        recovery.remember_device(tmp_path, last)
    return s, swapped


def model(device):
    return SimpleNamespace(device=device)


def test_cpu_start_after_gpu_is_watched(tmp_path):
    """Test that a CPU load on a machine that last ran on the iGPU starts the watch, keeping the remembered GPU."""
    s, swapped = session(tmp_path)
    s.on_loaded(model("CPU"))
    assert swapped and s.watch.missing == "igpu"
    assert recovery.last_device(tmp_path) == "igpu" # not overwritten by the fallback
    assert s.tray.engine == "CPU"


def test_gpu_start_is_remembered(tmp_path):
    """Test that a GPU load is remembered and nothing is watched."""
    s, _ = session(tmp_path, last=None)
    s.on_loaded(model("GPU"))
    assert recovery.last_device(tmp_path) == "igpu" and s.watch.missing is None


def test_upgrade_waits_until_idle(tmp_path):
    """Test that a model loaded on the returned GPU replaces the CPU one only when nothing runs."""
    s, swapped = session(tmp_path, state=State.RECORDING)
    gpu = model("GPU")
    s.on_loaded(gpu)
    assert swapped == [] and s.upgrade is gpu
    s.app.state = State.IDLE
    s._apply_upgrade()
    assert swapped == [gpu] and s.upgrade is None and s.tray.engine == "iGPU"


def test_upgrade_that_landed_on_cpu_is_dropped(tmp_path):
    """Test that a reload that fell back to the CPU anyway is not swapped in and the watch goes on."""
    s, swapped = session(tmp_path, state=State.IDLE)
    s.on_loaded(model("CPU"))
    assert swapped == [] and s.watch.missing == "igpu"


@pytest.mark.parametrize("found", [False, True])
def test_probe_result(tmp_path, monkeypatch, found):
    """Test that a found GPU starts a quiet background load and repeated misses tell the user once."""
    s, _ = session(tmp_path)
    s.on_loaded(model("CPU"))
    started = []
    monkeypatch.setattr(main.threading, "Thread", lambda target, args, **kw: SimpleNamespace(
        start=lambda: started.append(args)))
    for _ in range(recovery.NOTICE_AFTER):
        assert s.watch.due(resumed=True)
        s.on_probed(found)
        if found:
            break
    messages = [c for c in s.overlay.calls if c[0] == "show_message"]
    assert (len(started), len(messages)) == ((1, 0) if found else (0, 1))
    if found:
        assert started[0][2] is True # quiet: no status texts on the pill
