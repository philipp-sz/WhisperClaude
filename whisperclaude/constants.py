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

# Whisper's language codes (large-v3 and turbo: 100; "yue" is not in older models)
WHISPER_LANGUAGES = frozenset((
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs',
    'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu',
    'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka',
    'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml',
    'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt',
    'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo',
    'yue', 'zh',
))
