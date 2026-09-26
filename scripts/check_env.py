"""Quick sanity check: versions, CPU compute types, microphone."""
import sys

import ctranslate2
import faster_whisper
import psutil
import sounddevice as sd

print("Python", sys.version.split()[0])
print("faster-whisper", faster_whisper.__version__, "| ctranslate2", ctranslate2.__version__)
print("CT2 CPU compute types:", sorted(ctranslate2.get_supported_compute_types("cpu")))
print("CPU cores:", psutil.cpu_count(logical=False), "physical /", psutil.cpu_count(), "logical")
print("RAM total: %.1f GB" % (psutil.virtual_memory().total / 1e9))
print("Default input device:", sd.query_devices(kind="input")["name"])
