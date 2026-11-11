"""Smoke test of the comparison window with fake engines (skipped without a display)."""
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import compare_gui as gui # noqa: E402
from scripts import real_data as rd # noqa: E402
from scripts import real_engines as engines # noqa: E402


class Fake:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return self.text


class FakeSession(engines.Session):
    def load(self, name):
        self.engines[name] = Fake({"whisper": "Das ist die Krankenkarte", "parakeet": "Das ist die Krankheit"}[name])


def pump(root, seconds=2.0, until=lambda: False):
    end = time.time() + seconds
    while time.time() < end and not until():
        root.update()
        time.sleep(0.02)


def test_window_shows_each_engine_with_highlighted_errors(tmp_path):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        rec = rd.save_recording(np.zeros(16_000, np.float32), tmp_path)
        rd.save_reference(rec, "Das ist die Krankheit")
        app = gui.App(root, FakeSession(tmp_path, python="C:/x/python.exe"))
        app.listbox.selection_set(0)
        app.select(app.listbox.curselection())
        column = app.columns["whisper"]
        other = app.columns["parakeet"]
        pump(root, until=lambda: column.summary.cget("text") and other.summary.cget("text"))
        assert "Krankenkarte" in column.text.get("1.0", "end")
        assert column.text.tag_ranges("sub") # the wrong word is highlighted
        assert "1 errors" in column.summary.cget("text") and "0 errors" in other.summary.cget("text")
    finally:
        root.destroy()


class FakePlayer:
    def __init__(self):
        self.played, self.stopped = [], 0

    def play(self, audio):
        self.played.append(len(audio))

    def stop(self):
        self.stopped += 1


def make_app(tmp_path, player):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    for text in ("eins", "zwei"):
        rec = rd.save_recording(np.zeros(16_000, np.float32), tmp_path)
        rd.save_reference(rec, text)
        time.sleep(1.05) # the id is the time in seconds
    app = gui.App(root, FakeSession(tmp_path), player)
    return root, app


def test_play_button_plays_the_selected_recording_and_stops_it(tmp_path):
    player = FakePlayer()
    root, app = make_app(tmp_path, player)
    try:
        app.toggle_playback() # nothing selected yet
        assert player.played == []
        app.listbox.selection_set(0)
        app.select(app.listbox.curselection())
        app.toggle_playback()
        assert player.played == [16_000] and app.play.cget("text") == "Stop playing"
        app.toggle_playback() # second click stops
        assert player.stopped == 1 and app.play.cget("text") == "Play" and not app.playing
    finally:
        root.destroy()


def test_selecting_another_recording_stops_the_playback(tmp_path):
    player = FakePlayer()
    root, app = make_app(tmp_path, player)
    try:
        app.listbox.selection_set(0)
        app.select(app.listbox.curselection())
        app.toggle_playback()
        app.listbox.selection_clear(0, "end")
        app.listbox.selection_set(1)
        app.select(app.listbox.curselection())
        assert player.stopped == 1 and app.play.cget("text") == "Play"
    finally:
        root.destroy()


def test_the_button_resets_itself_when_the_audio_has_ended(tmp_path):
    player = FakePlayer()
    root, app = make_app(tmp_path, player)
    try:
        app.listbox.selection_set(0)
        app.select(app.listbox.curselection())
        app.toggle_playback()
        pump(root, seconds=2.0, until=lambda: app.play.cget("text") == "Play") # 1 s of audio + 0.3 s
        assert app.play.cget("text") == "Play" and player.stopped == 0
    finally:
        root.destroy()
