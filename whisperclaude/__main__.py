"""Console version (step 2): python -m whisperclaude"""
import logging
import time

from whisperclaude.app import App, State
from whisperclaude.hotkey import start_hotkey_listener
from whisperclaude.inserter import paste_text
from whisperclaude.recorder import Recorder
from whisperclaude.transcriber import Transcriber


def print_state(state: State, msg: str) -> None:
    labels = {State.IDLE: "idle", State.RECORDING: "● recording", State.TRANSCRIBING: "transcribing…"}
    print(f"[{time.strftime('%H:%M:%S')}] {labels[state]}" + (f"  {msg}" if msg else ""), flush=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    print("Loading model...", flush=True)
    t0 = time.perf_counter()
    transcriber = Transcriber()
    print(f"Model loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    app = App(Recorder(), transcriber, paste_text, on_state=print_state)
    listener = start_hotkey_listener(app.events)
    print("Ready. Tap Left Ctrl to start/stop dictation. Ctrl+C here to quit.", flush=True)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print("Bye.")


if __name__ == "__main__":
    main()
