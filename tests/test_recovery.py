"""GpuWatch scheduling and the remembered device (recovery.py), with a fake clock."""
import pytest

from whisperclaude import recovery
from whisperclaude.recovery import FIRST_PROBE_S, NOTICE_AFTER, PROBE_EVERY_S, GpuWatch


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("device, kind", [("CUDA", "cuda"), ("GPU", "igpu"), ("GPU.1", "igpu"), ("CPU", "cpu")])
def test_device_kind(device, kind):
    """Test that transcriber devices map to the remembered kinds."""
    assert recovery.device_kind(device) == kind


def test_remembered_device(tmp_path):
    """Test that the device kind survives a restart and junk is ignored."""
    assert recovery.last_device(tmp_path) is None
    recovery.remember_device(tmp_path, "igpu")
    assert recovery.last_device(tmp_path) == "igpu"
    (tmp_path / recovery.DEVICE_FILE).write_text("garbage")
    assert recovery.last_device(tmp_path) is None


@pytest.mark.parametrize("kind, expected", [("igpu", "igpu"), ("cpu", None), ("cpu", "cpu"), ("cuda", "igpu")])
def test_no_watch_unless_degraded(kind, expected):
    """Test that only a CPU load after a GPU load is watched."""
    clock = Clock()
    watch = GpuWatch(clock)
    watch.loaded(kind, expected)
    clock.now += 10 * PROBE_EVERY_S
    assert watch.due(resumed=True) is False


def test_probes_after_start_on_resume_and_periodically():
    """Test the schedule: first check after FIRST_PROBE_S, at a resume, then every PROBE_EVERY_S."""
    clock = Clock()
    watch = GpuWatch(clock)
    watch.loaded("cpu", "igpu")
    assert watch.missing == "igpu" and not watch.due()
    clock.now += FIRST_PROBE_S
    assert watch.due()
    assert not watch.due(resumed=True) # one check at a time
    watch.probe_done(False)
    assert watch.due(resumed=True) # a wake checks at once
    watch.probe_done(False)
    clock.now += PROBE_EVERY_S - 1
    assert not watch.due()
    clock.now += 1
    assert watch.due()


def test_notice_once_after_failed_checks():
    """Test that the user is told once, after NOTICE_AFTER failed checks."""
    watch = GpuWatch(Clock())
    watch.loaded("cpu", "cuda")
    told = []
    for _ in range(NOTICE_AFTER + 3):
        assert watch.due(resumed=True)
        told.append(watch.probe_done(False))
    assert told == [False] * (NOTICE_AFTER - 1) + [True] + [False] * 3


def test_found_gpu_stops_watching():
    """Test that a found GPU ends the checks (the reload's loaded() decides what comes next)."""
    watch = GpuWatch(Clock())
    watch.loaded("cpu", "igpu")
    assert watch.due(resumed=True)
    assert watch.probe_done(True) is False
    assert watch.missing is None and not watch.due(resumed=True)
