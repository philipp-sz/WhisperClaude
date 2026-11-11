"""Records what you say and shows what each engine heard, so you can compare them on your own speech.

.venv\\Scripts\\python.exe scripts\\compare_gui.py

1. Click Record, say what you usually say, click Stop. The recording is saved in bench_audio/real/ (git-ignored).
2. Every engine transcribes it. Write what you really said into the reference box (use "Copy" to start from an
   engine's text and fix it) and mark names as [[Name]]. Save.
   Play listens to the selected recording again (and Stop ends it), to check what you really said.
3. Wrong words (red) and invented words (orange) are highlighted per engine, with the error counts below.
Everything is kept: scripts/real_eval.py compares any engine, now or later, over all saved recordings.
Parakeet needs the PyTorch environment (set WC_TORCH_PYTHON, see scripts/real_engines.py).
"""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.real_data import Recording, evaluate, load_recordings, markup, read_wav, save_recording, save_reference # noqa: E402
from scripts.real_engines import Session # noqa: E402

TITLES = {"whisper": "Whisper", "parakeet": "Parakeet"}
HINT = "What you really said. Mark names as [[Name]]."
MIN_SECONDS = 0.5


class Player:
    """Plays 16 kHz mono audio through the default output device; play() returns at once."""

    def play(self, audio) -> None:
        import sounddevice as sd
        sd.play(audio, 16_000)

    def stop(self) -> None:
        import sounddevice as sd
        sd.stop()


class Column:
    """One engine's output: a title, the highlighted text and the error summary."""

    def __init__(self, parent: tk.Widget, name: str) -> None:
        self.name = name
        self.frame = ttk.LabelFrame(parent, text=TITLES[name])
        self.text = tk.Text(self.frame, width=34, height=14, wrap="word", state="disabled")
        self.text.tag_configure("sub", background="#f4a6a6")
        self.text.tag_configure("ins", background="#f7c98b")
        self.summary = ttk.Label(self.frame, text="", wraplength=250, justify="left")
        self.text.pack(fill="both", expand=True, padx=4, pady=4)
        self.summary.pack(fill="x", padx=4, pady=(0, 4))

    def show(self, reference: str, output: dict | None, pending: bool) -> None:
        """Draws the text (errors highlighted if there is a reference) and the summary."""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if output is None:
            self.text.insert("end", "…" if pending else "(not run)")
            self.summary.configure(text="")
        else:
            self._fill(reference, output)
        self.text.configure(state="disabled")

    def _fill(self, reference: str, output: dict) -> None:
        if not reference.strip():
            self.text.insert("end", output["text"])
            self.summary.configure(text=f"{output['s']:.2f} s")
            return
        pieces, missing = markup(reference, output["text"])
        for piece, kind in pieces:
            self.text.insert("end", piece, kind if kind != "ok" else ())
        r = evaluate(reference, output["text"])
        names = f" | names {r['names_wrong']}/{r['names']}" if r["names"] else ""
        lost = f"\nmissing: {' '.join(missing)}" if missing else ""
        self.summary.configure(text=f"{r['errors']} errors (wrong {r['sub']}, missing {r['del']}, invented {r['ins']})"
                                    f"{names} | {output['s']:.2f} s{lost}")


class App:
    """The window: recorder, recordings list, reference box and one column per engine."""

    def __init__(self, root: tk.Tk, session: Session, player: Player | None = None) -> None:
        self.root, self.session = root, session
        self.player = player or Player()
        self.playing = 0 # token of the running playback, 0 = none (the end of a playback checks it)
        self.recorder = None
        self.recordings: list[Recording] = load_recordings(session.directory)
        self.current: Recording | None = None
        self.jobs: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        self.ready: set[str] = set()
        self.failed: set[str] = set()
        self.pending: set[str] = set()
        self._build()
        threading.Thread(target=self._worker, daemon=True).start()
        for name in session.available():
            self.jobs.put(("load", name))
        self._pump()

    # ---------- layout ----------

    def _build(self) -> None:
        self.root.title("Compare engines on your own recordings")
        top = ttk.Frame(self.root, padding=6)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.record = ttk.Button(top, text="Record", command=self.toggle_recording)
        self.record.pack(side="left")
        self.level = ttk.Progressbar(top, length=160, maximum=0.3)
        self.level.pack(side="left", padx=8)
        self.play = ttk.Button(top, text="Play", command=self.toggle_playback)
        self.play.pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Import WAV…", command=self.import_wav).pack(side="left", padx=(4, 0))
        self.status = ttk.Label(top, text="Loading engines…")
        self.status.pack(side="left", padx=12)
        side = ttk.Frame(self.root, padding=6)
        side.grid(row=1, column=0, sticky="ns")
        ttk.Label(side, text="Recordings").pack(anchor="w")
        self.listbox = tk.Listbox(side, width=22, height=24, exportselection=False)
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.select(self.listbox.curselection()))
        main = ttk.Frame(self.root, padding=6)
        main.grid(row=1, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._build_reference(main)
        self.columns = {n: Column(main, n) for n in self.session.available()}
        for i, column in enumerate(self.columns.values()):
            column.frame.grid(row=2, column=i, sticky="nsew", padx=2)
            main.columnconfigure(i, weight=1)
        main.rowconfigure(2, weight=1)
        self._refresh_list()

    def _build_reference(self, main: ttk.Frame) -> None:
        box = ttk.LabelFrame(main, text=HINT)
        box.grid(row=0, column=0, columnspan=8, sticky="ew")
        self.reference = tk.Text(box, height=4, wrap="word")
        self.reference.pack(fill="x", padx=4, pady=4)
        bar = ttk.Frame(box)
        bar.pack(fill="x", padx=4, pady=(0, 4))
        for name in self.session.available():
            ttk.Button(bar, text=f"Copy {TITLES[name]}", command=lambda n=name: self.copy_from(n)).pack(side="left", padx=2)
        ttk.Button(bar, text="Save reference", command=self.save).pack(side="right")
        ttk.Button(bar, text="Run engines again", command=lambda: self.run_engines(force=True)).pack(side="right", padx=6)
        self.note = ttk.Entry(bar, width=28)
        self.note.pack(side="right")
        ttk.Label(bar, text="Note:").pack(side="right")

    # ---------- recordings ----------

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for rec in self.recordings:
            self.listbox.insert("end", ("✓ " if rec.reference.strip() else "· ") + rec.id)

    def select(self, index) -> None:
        if not index:
            return
        self.stop_playback()
        self.current = self.recordings[index[0]]
        self.reference.delete("1.0", "end")
        self.reference.insert("1.0", self.current.reference)
        self.note.delete(0, "end")
        self.note.insert(0, self.current.note)
        self.run_engines()
        self.redraw()

    def toggle_playback(self) -> None:
        """Plays the selected recording, or stops the playback that is running."""
        if self.playing:
            self.stop_playback()
            return
        if self.current is None or self.recorder is not None:
            return
        audio = read_wav(self.current.wav)
        self.player.play(audio)
        self.playing = token = self.playing + 1 if self.playing else int(self.root.tk.call("clock", "milliseconds"))
        self.play.configure(text="Stop playing")
        self.root.after(int(len(audio) / 16_000 * 1000) + 300, lambda: self._playback_ended(token))

    def stop_playback(self) -> None:
        if self.playing:
            self.player.stop()
            self._playback_ended(self.playing)

    def _playback_ended(self, token: int) -> None:
        """Resets the button when the playback with this token is the one still running."""
        if self.playing == token:
            self.playing = 0
            self.play.configure(text="Play")

    def toggle_recording(self) -> None:
        if self.recorder is None:
            self.stop_playback()
            from whisperclaude.recorder import Recorder
            self.recorder = Recorder()
            self.recorder.start()
            self.record.configure(text="Stop")
            self.status.configure(text="Recording… say what you usually say")
            self._meter()
            return
        audio, self.recorder = self.recorder.stop(), None
        self.record.configure(text="Record")
        self.level["value"] = 0
        if len(audio) < MIN_SECONDS * 16_000:
            self.status.configure(text="Too short, discarded")
            return
        self.add_recording(save_recording(audio, self.session.directory))

    def _meter(self) -> None:
        if self.recorder is not None:
            self.level["value"] = self.recorder.level
            self.root.after(60, self._meter)

    def import_wav(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("16 kHz mono WAV", "*.wav")])
        if path:
            self.add_recording(save_recording(read_wav(Path(path)), self.session.directory))

    def add_recording(self, rec: Recording) -> None:
        self.recordings.append(rec)
        self._refresh_list()
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set("end")
        self.select(self.listbox.curselection())

    def save(self) -> None:
        if self.current is None:
            return
        save_reference(self.current, self.reference.get("1.0", "end").strip(), self.note.get())
        self._refresh_list()
        self.listbox.selection_set(self.recordings.index(self.current))
        self.redraw()

    def copy_from(self, name: str) -> None:
        outputs = self.session.outputs(self.current) if self.current else {}
        if name in outputs:
            self.reference.delete("1.0", "end")
            self.reference.insert("1.0", outputs[name]["text"])

    # ---------- engines (one worker thread, so the models are never used twice at once) ----------

    def run_engines(self, force: bool = False) -> None:
        """Queues the current recording for every engine (cached outputs return at once)."""
        if self.current is None:
            return
        for name in self.session.available():
            self.pending.add(f"{self.current.id}/{name}")
            self.jobs.put(("run", self.current, name, force))

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                self._handle(job)
            except Exception as e: # a failing engine must not kill the window
                name = job[1] if job[0] == "load" else job[2]
                if job[0] == "load":
                    self.failed.add(name)
                self.events.put(("error", name, str(e)))

    def _handle(self, job: tuple) -> None:
        if job[0] == "load":
            self.session.load(job[1])
            self.events.put(("loaded", job[1], None))
            return
        _, rec, name, force = job
        if name in self.failed:
            self.events.put(("skipped", f"{rec.id}/{name}", None))
        elif name in self.session.engines or (not force and name in self.session.outputs(rec)):
            self.session.transcribe(rec, name, force)
            self.events.put(("done", f"{rec.id}/{name}", None))
        else: # the engine is still loading: try again shortly
            self.jobs.put(job)
            threading.Event().wait(0.5)

    def _pump(self) -> None:
        """Handles messages of the worker on the Tk thread."""
        while not self.events.empty():
            kind, what, error = self.events.get()
            if kind == "loaded":
                self.ready.add(what)
                self.status.configure(text="Ready: " + ", ".join(sorted(self.ready)))
            elif kind == "error":
                self.status.configure(text=f"{what} failed: {error[:90]}")
                self.pending = {p for p in self.pending if not p.endswith("/" + what)}
            else:
                self.pending.discard(what)
            self.redraw()
        self.root.after(150, self._pump)

    def redraw(self) -> None:
        if self.current is None:
            return
        reference = self.reference.get("1.0", "end") if self.reference.get("1.0", "end").strip() else self.current.reference
        outputs = self.session.outputs(self.current)
        for name, column in self.columns.items():
            column.show(reference, outputs.get(name), f"{self.current.id}/{name}" in self.pending)


def main() -> None:
    root = tk.Tk()
    App(root, Session())
    root.mainloop()


if __name__ == "__main__":
    main()
