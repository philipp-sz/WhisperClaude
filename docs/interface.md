# Status pill, tray and history

## Status pill

Tk's canvas draws without anti-aliasing, gradients or transparency. The "wave" and "gradient" pill styles are therefore drawn with numpy into one image per frame (`pillart.py`): bars as anti-aliased capsules with a color gradient, waves as three translucent layers mirrored around a thin line. One frame takes 0.6 ms (waves) and 0.2 ms (bars) at 100% display scaling and 2.0 ms and 0.8 ms at 200%, on a Core i5-10600K. At 30 frames per second while recording, the waves cost up to 6% of one core. Frames only run while recording, transcribing or translating.

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

Pasting needs the clipboard: the app puts the text there, sends Ctrl+V and restores the previous content after 0.3 s, including images and files. Apart from that, the app never writes the clipboard. Earlier versions copied a dictation to the clipboard when it could not be pasted, which replaced whatever the user had copied. Such a dictation is now kept instead:

| Case | Pill | Note in "Recent dictations" |
|---|---|---|
| Esc while recording | "Cancelled" + Copy | cancelled |
| No speech for 60 s (auto-stop) | "Stopped after 60 s of silence" + Copy | not pasted |
| The language picker lost the focus | "Not pasted (the focus moved)" + Copy | not pasted |
| A dictation's translation failed | "Translation failed: …" + Copy (the original) | |

Every dictation is saved to the history before it is pasted, so a paste into a window without a text field cannot lose it. A translated dictation or selection is one entry with both texts.
