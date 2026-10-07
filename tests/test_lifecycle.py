import os
import subprocess
import sys
import time

from whisperclaude import lifecycle


def dead_pid() -> int:
    """A pid that certainly isn't running: a child that has already exited."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_no_marker_means_nothing_to_report(tmp_path):
    assert lifecycle.previous_instance_died(tmp_path) is None


def test_clean_exit_leaves_no_trace(tmp_path):
    lifecycle.mark_running(tmp_path)
    lifecycle.mark_clean_exit(tmp_path)
    assert lifecycle.previous_instance_died(tmp_path) is None
    lifecycle.mark_clean_exit(tmp_path)  # idempotent


def test_marker_of_a_dead_instance_is_reported_with_time(tmp_path):
    marker = tmp_path / lifecycle.RUNNING
    marker.write_text(str(dead_pid()), encoding="utf-8")
    last_seen = time.time() - 3 * 3600
    os.utime(marker, (last_seen, last_seen))
    msg = lifecycle.previous_instance_died(tmp_path)
    assert msg and "did not shut down cleanly" in msg
    assert "180 min ago" in msg or "179 min ago" in msg
    assert time.strftime("%H:%M:%S", time.localtime(last_seen)) in msg


def test_marker_of_a_running_python_process_is_not_reported(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / lifecycle.RUNNING).write_text(str(p.pid), encoding="utf-8")
        assert lifecycle.previous_instance_died(tmp_path) is None
    finally:
        p.kill()
        p.wait()


def test_marker_with_a_reused_pid_of_another_program_is_reported(tmp_path):
    """Windows reuses pids: a live pid that isn't python can't be the previous instance."""
    import psutil

    other = next(p for p in psutil.process_iter(["name"])
                 if p.info["name"] and "python" not in p.info["name"].lower() and p.pid > 4)
    (tmp_path / lifecycle.RUNNING).write_text(str(other.pid), encoding="utf-8")
    assert lifecycle.previous_instance_died(tmp_path)


def test_garbage_marker_is_ignored(tmp_path):
    (tmp_path / lifecycle.RUNNING).write_text("not a pid", encoding="utf-8")
    assert lifecycle.previous_instance_died(tmp_path) is None


def test_heartbeat_updates_modification_time_and_recreates_a_missing_marker(tmp_path):
    lifecycle.mark_running(tmp_path)
    marker = tmp_path / lifecycle.RUNNING
    old = time.time() - 600
    os.utime(marker, (old, old))
    lifecycle.heartbeat(tmp_path)
    assert marker.stat().st_mtime > old + 500
    marker.unlink()
    lifecycle.heartbeat(tmp_path)
    assert marker.exists()


def test_user_quit_marker(tmp_path):
    assert not lifecycle.user_quit(tmp_path)
    lifecycle.mark_user_quit(tmp_path)
    assert lifecycle.user_quit(tmp_path)
    lifecycle.clear_user_quit(tmp_path)
    assert not lifecycle.user_quit(tmp_path)
    lifecycle.clear_user_quit(tmp_path)  # idempotent
