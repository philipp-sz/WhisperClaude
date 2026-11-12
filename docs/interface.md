# Status pill, tray and history

## Status pill

Tk's canvas draws without anti-aliasing, gradients or transparency. The "wave" and "gradient" pill styles are therefore drawn with numpy into one image per frame (`pillart.py`): bars as anti-aliased capsules with a color gradient, waves as three translucent layers mirrored around a thin line. One frame takes 0.6 ms (waves) and 0.2 ms (bars) at 100% display scaling and 2.0 ms and 0.8 ms at 200%, on a Core i5-10600K. At 30 frames per second while recording, the waves cost up to 6% of one core. Frames only run while recording, transcribing or translating. The wave picture is symmetric around its middle line, so only the top half is computed and mirrored (pixel-identical, 1.5 times faster); on the laptop at 200% scaling the waves then cost about 4% of one core while recording (6% before), the tray icon practically nothing (its frames are cached). While the microphone starts, the pill shows the "Starting…" spinner for a few tenths of a second, then turns red.

The waves and bars follow the microphone level in dB. A fixed scale (-55 dB flat, -20 dB full) left a quiet laptop microphone nearly flat. The scale now adapts (`LevelMeter`): its top follows the loudest recent speech 75% of the way down from -20 dB, and it starts 6 dB above the background noise with at least 15 dB of range. Simulated speech, highest wave height (0 to 1):

| Speech peak | Fixed scale | Adaptive |
|---|---|---|
| -52 dB | 0.09 | 0.54 |
| -45 dB | 0.29 | 0.73 |
| -35 dB | 0.57 | 0.88 |
| -22 dB | 0.94 | 0.99 |

Room noise alone stays below 0.15. The tray icon's sound bars use the same value.

The other pills of the "wave" and "gradient" styles use the same rendering: a spinner while the model loads or the app restarts, and a badge with a symbol for every message (check for "Ready", "Pasted" and "Copied", "!" for errors, "i" for notes, a stop square for "Cancelled" and "stopped"). The badge pops in and its symbol draws itself, in 0.45 s. The "classic" style keeps plain canvas dots.

## Tray icon

The tray icon shows the state: grey loading, green ready, red recording, amber transcribing or translating, and a short green flash after a paste. The styles "bars", "pulse" and "ring" are animated, "classic" is the original static icon. Frames are drawn with Pillow at 4 times the size and scaled down, then cached as icon handles. pystray's own icon setter writes a temporary .ico file for every change, so the app sets the icon handles directly. The animation thread only runs while something moves.

## Clipboard and history

The previous clipboard is restored in every case: after a paste, after a paste that fails (for example a keyboard error after the text went in) and after the app copied a selection for translation (also if reading the copy fails). The restore waits up to 3 s for a busy clipboard. It is skipped only if the user copied something new while the text was pasted. The app detects that with Windows' clipboard change counter, so a clipboard manager that writes the same text again does not count. If the old clipboard cannot be read (another app holds it for more than a second), nothing is pasted and no selection is copied: the dictation stays in "Recent dictations", and what cannot be put back is never overwritten. Formats beyond 1 GB are not saved, which is logged. The Copy button and the tray entry are the only places that write the clipboard on purpose. Not covered: the app being killed within the 0.3 s of a paste, and an app that copies the selection more than 0.6 s after Ctrl+Insert.

Pasting needs the clipboard: the app puts the text there, sends Ctrl+V and restores the previous content after 0.3 s, including images and files. Apart from that, the app never writes the clipboard. Earlier versions copied a dictation to the clipboard when it could not be pasted, which replaced whatever the user had copied. Such a dictation is now kept instead:

| Case | Pill | Note in "Recent dictations" |
|---|---|---|
| Esc while recording | "Cancelled" + Copy | cancelled |
| No speech for 60 s (auto-stop) | "Stopped after 60 s of silence" + Copy | not pasted |
| The language picker lost the focus | "Not pasted (the focus moved)" + Copy | not pasted |
| Another window is in front than at the stop tap (dictation) or before translating (translation) | "Not pasted (the focus moved)" + Copy | not pasted |
| The paste failed (clipboard locked by another app) | "Paste failed: …" + Copy | |
| A dictation's translation failed | "Translation failed: …" + Copy (the original) | |

Every dictation is saved to the history before it is pasted, so a paste into a window without a text field cannot lose it. A translated dictation or selection is one entry with both texts.

## Spacing at the insertion point

Whisper's text has no leading space, so a second dictation pasted behind the first ran into it ("works.Next"). With `[insert] smart_spacing` (default on) the paste is adapted to what stands before the caret (`spacing.py`, `uia.py`). History and the Copy button keep the transcript as it is; only the pasted text differs. No text is logged.

1. **UI Automation** (`uia.CaretReader`): the focused control's TextPattern2 caret range (TextPattern selection as fallback) is moved back by one character and that character is read: 3-11 ms. It runs on its own long-lived COM thread; the paste waits at most 50 ms, a stuck app costs one timeout and further reads are refused until it answers. The thread starts at app start, because loading UIA takes about 70 ms.
2. **Last paste** (`spacing.PasteTracker`), when UIA says nothing (Tk and other self-drawn UIs): if the next paste goes to the same foreground window, within 5 minutes, and the hook saw no key (modifiers excluded: Ctrl/Alt are our hotkeys) and no mouse click since, the caret still is behind the last text.
3. **Rule** (`needs_space`): a space is added unless the caret is at the start of the text, after whitespace or a line break, after an opening bracket or `/ \ @ # $ ~ „ ¿ ¡`, or the dictation starts with whitespace or closing punctuation (`.,;:!?…)]}%`). Unknown context: pasted as is.

`[insert] adapt_case` (default off) also lowercases the first word when the sentence goes on (last non-blank character before the caret on the same line is a letter, comma or semicolon). Only function words (`FUNCTION_WORDS`: the, und, dass, ...) are lowercased, never nouns (German), "I", acronyms or names. It stays off because the guess fails at sentence ends without punctuation (titles, list items typed without a period) and because the benefit is small next to the spacing.

Measured support (2026-11-12; own test windows, VS Code: the focused field of the running instance): Notepad (Windows 11), Edge textarea, Windows Terminal (PowerShell) and VS Code answer; Tk Entry and Text do not (fallback). Chromium turns its accessibility tree on when a UIA client asks; the cost of that in the browser is not measured.

Known limits (2026-11-12):
- Chromium lets the range walk out of an empty input into the page text in front of it; the read is now clamped to the focused element's own text range (`limit` in `_text_before_range`), checked on Edge test pages (empty and filled contenteditable and textarea).
- **VS Code's chat box gets a leading space when empty.** It draws its placeholder ("Queue another message…") with CSS, and Chromium reports that as the box's text with the caret after it. Structure (child roles, enclosing elements), text colour, caret and bounding rectangles did not separate it from typed text: every check that caught the placeholder also broke real text on the Edge pages. A normal `placeholder` attribute is not affected. Left as is; ideas: record the structure while typing in that box, or treat a box whose only text ends in "…" as empty.
