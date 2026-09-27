import os
import sys

_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = int(exitstatus)


def pytest_unconfigure(config):
    """OpenVINO's native threads can block interpreter shutdown (the app has the same
    workaround in __main__). Once pytest is completely done, exit hard if it was loaded."""
    if "openvino_genai" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_status)
