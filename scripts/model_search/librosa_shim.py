"""Stand-in for the one librosa function that Parakeet's feature extractor in transformers uses.

librosa's scipy DLL was blocked by Windows application control on the laptop (an unsigned .pyd), so the
Slaney mel filterbank (librosa.filters.mel) is rebuilt here. Verified: identical to the filters inside
faster-whisper (max difference 0.0). Call install() before importing transformers.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    return np.where(freq >= 1000, 15 + np.log(np.maximum(freq, 1e-10) / 1000) / (np.log(6.4) / 27), freq / (200 / 3))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel >= 15, 1000 * np.exp((np.log(6.4) / 27) * (mel - 15)), mel * (200 / 3))


def mel(*, sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None,
        htk: bool = False, norm: str = "slaney", dtype: type = np.float32) -> np.ndarray:
    """Slaney mel filterbank, shape (n_mels, 1 + n_fft // 2)."""
    assert not htk and norm == "slaney"
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(sr / 2 if fmax is None else fmax), n_mels + 2))
    ramps = np.subtract.outer(mel_f, fft_freqs)
    lower = -ramps[:-2] / np.diff(mel_f)[:-1, None]
    upper = ramps[2:] / np.diff(mel_f)[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    return (weights * (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]).astype(dtype)


def install() -> None:
    """Register the stand-in as the `librosa` module."""
    module = types.ModuleType("librosa")
    module.__spec__ = importlib.machinery.ModuleSpec("librosa", None)
    module.__version__ = "0.11.0" # transformers rejects a package without a version as a namespace package
    module.__file__ = __file__
    module.filters = types.SimpleNamespace(mel=mel)
    sys.modules["librosa"] = module
