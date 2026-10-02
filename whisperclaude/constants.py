"""Light constants shared by many modules.

Kept apart from transcriber.py on purpose: importing that pulls in the heavy Whisper stack
(ctranslate2, onnxruntime, OpenVINO), and the UI, tray and hotkey should come up without
waiting for it.
"""

SAMPLE_RATE = 16_000

# Correctly punctuated bilingual "previous text": Whisper copies its style. Fixes missing
# punctuation, capitals after pauses, and German+English audio being translated to one language.
DEFAULT_PROMPT = "Hallo, das ist ein Test. Okay, so let's see how this works, and then we'll decide."
DEFAULT_VOCABULARY = ("Claude", "cloud", "Gemini", "VS Code", "iGPU")
