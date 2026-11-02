"""Language picker for the translate key: "From → To" with a search box and a list.

Unlike the pill it takes the keyboard focus (for typing), and hands it back to the window it
was opened over when the languages are picked or Esc is pressed, so the paste lands there.
The list picks the target language; Tab (or a click on "From") switches it to the source
language, which is "Auto" (detected per sentence, its guess shown next to it) until the user
picks one. A wrong source makes NLLB write nonsense on short texts, see docs/translation.md.
Recently used languages come first; typing filters by English name, native name or code.
All methods run on the Tk thread.
"""
from __future__ import annotations

import ctypes
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

from whisperclaude.languages import BY_CODE, Language, search
from whisperclaude.overlay import _work_area

BG = "#202124"
FIELD = "#303134"
FG = "#ffffff"
GREY = "#9aa0a6"
EDGE = "#5f6368"
SELECT = "#1a73e8"
ROWS = 8
AUTO = "Auto-detect"

Choice = tuple[str, str | None] # (target, source or None = auto-detect)


def _force_foreground(hwnd: int) -> None:
    """Windows only lets the foreground app hand out the focus. Attaching to its input queue
    for a moment makes our window count as part of it."""
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    current = user32.GetForegroundWindow()
    other = user32.GetWindowThreadProcessId(current, None)
    me = kernel32.GetCurrentThreadId()
    attached = bool(other) and other != me and user32.AttachThreadInput(me, other, True)
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, other, False)


def _name(code: str) -> str:
    return BY_CODE[code].name if code in BY_CODE else code


class LanguagePicker:
    def __init__(self, root: tk.Tk, languages: list[Language],
                 recent: Callable[[], list[str]] = lambda: [],
                 recent_sources: Callable[[], list[str]] = lambda: [],
                 own: Callable[[], list[str]] = lambda: []) -> None:
        """recent: recently picked targets; recent_sources: recently picked sources; own: the
        user's own languages (NLLB codes), offered first as sources."""
        self.root = root
        self.languages = languages
        self.recent, self.recent_sources, self.own = recent, recent_sources, own
        self.scale = root.winfo_fpixels("1i") / 96
        self.font = tkfont.Font(family="Segoe UI", size=11)
        self.small = tkfont.Font(family="Segoe UI", size=9)
        self._on_done: Callable[[Choice | None, bool], None] | None = None
        self._mode = "to"
        self._target: str | None = None
        self._source: str | None = None
        self._guess: list[str] = []
        self._hint = ""
        self._shown: list[Language | None] = [] # None: the "Auto-detect" row
        self._previous = 0 # the window to give the focus back to
        self._hwnd = 0
        self._build()

    def _build(self) -> None:
        s = self.scale
        win = self.win = tk.Toplevel(self.root)
        win.withdraw()
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=BG, highlightthickness=1, highlightbackground=EDGE)
        pad = round(10 * s)
        self._build_header(pad)
        self.query = tk.StringVar()
        self.entry = tk.Entry(win, textvariable=self.query, bg=FIELD, fg=FG, insertbackground=FG,
                              relief="flat", font=self.font, highlightthickness=0)
        self.entry.pack(fill="x", padx=pad, pady=round(8 * s), ipady=round(4 * s))
        self.list = tk.Listbox(win, height=ROWS, bg=BG, fg=FG, font=self.font, relief="flat",
                               highlightthickness=0, selectbackground=SELECT,
                               selectforeground=FG, activestyle="none", takefocus=0,
                               exportselection=False)
        self.list.pack(fill="both", padx=pad)
        self.hint = tk.Label(win, bg=BG, fg=GREY, font=self.small, anchor="w")
        self.hint.pack(fill="x", padx=pad, pady=(round(4 * s), pad))
        self.query.trace_add("write", lambda *a: self._refresh())
        self._bind_keys()

    def _build_header(self, pad: int) -> None:
        """"From [chip] → To [chip]"; a click on a chip switches the list to it."""
        s = self.scale
        header = tk.Frame(self.win, bg=BG)
        header.pack(fill="x", padx=pad, pady=(pad, 0))
        self.chips: dict[str, tk.Label] = {}
        for mode, caption in (("from", "From"), ("to", "To")):
            box = tk.Frame(header, bg=BG)
            tk.Label(box, text=caption, bg=BG, fg=GREY, font=self.small, anchor="w").pack(fill="x")
            chip = tk.Label(box, bg=FIELD, fg=FG, font=self.font, anchor="w", width=1,
                            padx=round(8 * s), pady=round(3 * s), cursor="hand2",
                            highlightthickness=1, highlightbackground=FIELD)
            chip.pack(fill="x")
            chip.bind("<Button-1>", lambda e, m=mode: self._switch(m))
            box.pack(side="left", fill="x", expand=True)
            self.chips[mode] = chip
            if mode == "from":
                tk.Label(header, text="→", bg=BG, fg=GREY, font=self.font).pack(
                    side="left", anchor="s", padx=round(6 * s), pady=(0, round(4 * s)))

    def _bind_keys(self) -> None:
        # Enter and Esc act on the key-up: the whole key press stays in the picker, so no
        # auto-repeat reaches the app the focus goes back to.
        for key, fn in (("<KeyRelease-Return>", lambda e: self._enter()),
                        ("<KeyRelease-KP_Enter>", lambda e: self._enter()),
                        ("<KeyRelease-Escape>", lambda e: self._escape()),
                        ("<Tab>", lambda e: self._switch("from" if self._mode == "to" else "to")),
                        ("<Shift-Tab>", lambda e: self._switch("from" if self._mode == "to" else "to")),
                        ("<Down>", lambda e: self._move(1)), ("<Up>", lambda e: self._move(-1)),
                        ("<Next>", lambda e: self._move(ROWS)),
                        ("<Prior>", lambda e: self._move(-ROWS))):
            self.entry.bind(key, fn)
        self.list.bind("<ButtonRelease-1>", lambda e: self._click(e))
        self.list.bind("<MouseWheel>", lambda e: None) # Listbox scrolls by itself
        self.win.bind("<FocusOut>", lambda e: self.root.after(150, self._check_focus))

    @property
    def is_open(self) -> bool:
        return self._on_done is not None

    def open(self, on_done: Callable[[Choice | None, bool], None],
             hint: str = "Enter: translate · Tab: from · Esc: cancel",
             guess: list[str] | tuple[str, ...] = ()) -> None:
        """Show the picker. on_done(choice, focus_back) runs once: ((target, source), True)
        when picked (source None = auto-detect), (None, True) after Esc, (None, False) when
        the user clicked somewhere else (the focus is then not given back). guess: the
        languages auto-detection found, shown next to "Auto"."""
        if self.is_open:
            self._finish(None, True)
        self._on_done = on_done
        self._previous = ctypes.windll.user32.GetForegroundWindow()
        self._hint, self._guess = hint, list(guess)
        targets = search(self.languages, "", self.recent())
        self._target = targets[0].code if targets else None
        self._source = None
        self._mode = "to"
        self.query.set("") # refreshes the list (a write trace)
        self._place()
        self.win.deiconify()
        self.win.lift()
        self.win.update_idletasks()
        self._hwnd = ctypes.windll.user32.GetParent(self.win.winfo_id())
        _force_foreground(self._hwnd)
        self.entry.focus_force()
        self.root.after(200, self._ensure_focus)

    def set_guess(self, codes: list[str]) -> None:
        """Auto-detection's result arrived (a dictation is transcribed while the picker is
        open): show it next to "Auto"."""
        self._guess = list(codes)
        if self.is_open:
            self._refresh()

    def confirm(self) -> None:
        """Translate with what is chosen (Enter, or the translate key tapped again)."""
        if self.is_open and self._target is not None:
            self._finish((self._target, self._source), True)

    def close(self) -> None:
        """Esc: no translation, focus back to the app."""
        if self.is_open:
            self._finish(None, True)

    # ---------- internals ----------

    def _place(self) -> None:
        """Centered above the pill, i.e. near the bottom of the primary work area."""
        s = self.scale
        width = round(380 * s)
        self.win.geometry(f"{width}x1") # height follows the content
        self.win.update_idletasks()
        height = self.win.winfo_reqheight()
        left, _, right, bottom = _work_area()
        x = left + (right - left - width) // 2
        y = bottom - height - round(80 * s)
        self.win.geometry(f"{width}x{height}+{x}+{y}")

    def _auto_label(self) -> str:
        return " + ".join(_name(c) for c in self._guess[:2])

    def _rows(self, query: str) -> list[Language | None]:
        if self._mode == "to":
            return list(search(self.languages, query, self.recent()))
        first = list(dict.fromkeys(self.own() + self.recent_sources()))
        auto: list[Language | None] = [None] if AUTO.lower().startswith(query.lower().strip()) else []
        return auto + list(search(self.languages, query, first))

    def _refresh(self) -> None:
        query = self.query.get()
        self._shown = self._rows(query)
        self.list.delete(0, "end")
        for lang in self._shown:
            if lang is None:
                guess = self._auto_label()
                self.list.insert("end", f"{AUTO} ({guess})" if guess else AUTO)
            else:
                native = "" if lang.native in ("", lang.name) else f" ({lang.native})"
                self.list.insert("end", lang.name + native)
        current = self._target if self._mode == "to" else self._source
        codes = [lang.code if lang else None for lang in self._shown]
        index = codes.index(current) if not query.strip() and current in codes else 0
        if self._shown:
            self._select(index)
        self._show_state()

    def _select(self, i: int) -> None:
        self.list.selection_clear(0, "end")
        self.list.selection_set(i)
        self.list.see(i)
        lang = self._shown[i]
        if self._mode == "to":
            self._target = lang.code if lang else self._target
        else:
            self._source = lang.code if lang else None
        self._show_state()

    def _show_state(self) -> None:
        more = f" +{len(self._guess) - 1}" if len(self._guess) > 1 else ""
        guess = f"Auto · {_name(self._guess[0])}{more}" if self._guess else "Auto"
        source = _name(self._source) if self._source else guess
        self.chips["from"].configure(text=source)
        self.chips["to"].configure(text=_name(self._target) if self._target else "")
        for mode, chip in self.chips.items():
            chip.configure(highlightbackground=SELECT if mode == self._mode else FIELD)
        self.hint.configure(text=self._hint if self._mode == "to"
                            else "Enter: choose · Tab or Esc: back to the target")

    def _switch(self, mode: str) -> str:
        if mode != self._mode:
            self._mode = mode
            self.query.set("") # refreshes the list, with the current choice highlighted
        return "break" # Tab: keep the focus in the search box

    def _enter(self) -> None:
        if self._mode == "from":
            self._switch("to")
        else:
            self.confirm()

    def _escape(self) -> None:
        if self._mode == "from":
            self._switch("to")
        else:
            self.close()

    def _move(self, step: int) -> str:
        if self._shown:
            selection = self.list.curselection()
            self._select(min(len(self._shown) - 1,
                             max(0, (selection[0] if selection else 0) + step)))
        return "break" # keep the cursor in the search box

    def _click(self, event: tk.Event) -> None:
        i = self.list.nearest(event.y)
        if 0 <= i < len(self._shown):
            self._select(i)
            self._enter()

    def _ensure_focus(self) -> None:
        """Windows sometimes refuses the first focus change; one retry."""
        if self.is_open and ctypes.windll.user32.GetForegroundWindow() != self._hwnd:
            _force_foreground(self._hwnd)
            self.entry.focus_force()

    def _check_focus(self) -> None:
        """The user clicked into another window: give up without taking the focus back."""
        if self.is_open and ctypes.windll.user32.GetForegroundWindow() != self._hwnd:
            self._finish(None, False)

    def _finish(self, choice: Choice | None, focus_back: bool) -> None:
        on_done, self._on_done = self._on_done, None
        self.win.withdraw()
        if focus_back and self._previous:
            ctypes.windll.user32.SetForegroundWindow(self._previous)
        if on_done is not None:
            on_done(choice, focus_back)
