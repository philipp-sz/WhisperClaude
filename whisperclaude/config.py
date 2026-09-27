"""Load and validate config.toml. Every key is optional; defaults = benchmark results."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from whisperclaude.hotkey import parse_key
from whisperclaude.transcriber import DEFAULT_PROMPT, DEFAULT_VOCABULARY

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
COMPUTE_TYPES = {"int8", "int8_float32", "int16", "float32"}
DEVICES = {"auto", "gpu", "cpu"}


class ConfigError(ValueError):
    """Invalid config file; the message says which key and why."""


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ConfigError(msg)


@dataclass(frozen=True)
class HotkeyConfig:
    key: str = "ctrl_l"       # tap this key alone to toggle
    max_hold_s: float = 0.5   # held longer = not a tap

    def __post_init__(self) -> None:
        try:
            parse_key(self.key)
        except ValueError as e:
            raise ConfigError(f"[hotkey] key: {e}") from None
        _check(0.05 <= self.max_hold_s <= 3, "[hotkey] max_hold_s must be 0.05–3")


@dataclass(frozen=True)
class ModelConfig:
    device: str = "auto"              # "auto" = iGPU if available, else CPU | "gpu" | "cpu"
    gpu_model: str = "large-v3-turbo"  # OpenVINO model on the iGPU
    cpu_model: str = "small"           # faster-whisper model on the CPU (fallback)
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 1
    batch_size: int = 4
    language: str = ""        # "" = auto-detect, else e.g. "de" / "en"
    initial_prompt: str = DEFAULT_PROMPT
    vocabulary: tuple[str, ...] = DEFAULT_VOCABULARY  # words Whisper should spell exactly

    def __post_init__(self) -> None:
        _check(self.device in DEVICES, f"[model] device must be one of {sorted(DEVICES)}")
        _check(all(isinstance(w, str) and w.strip() for w in self.vocabulary),
               "[model] vocabulary must be a list of non-empty strings")
        _check(self.compute_type in COMPUTE_TYPES,
               f"[model] compute_type must be one of {sorted(COMPUTE_TYPES)}")
        for key in ("cpu_threads", "beam_size", "batch_size"):
            _check(1 <= getattr(self, key) <= 32, f"[model] {key} must be 1–32")
        _check(self.language == "" or (len(self.language) == 2 and self.language.isalpha()),
               '[model] language must be "" (auto) or a 2-letter code like "de"')


@dataclass(frozen=True)
class PasteConfig:
    restore_delay_s: float = 0.3  # wait before restoring the old clipboard

    def __post_init__(self) -> None:
        _check(0 <= self.restore_delay_s <= 5, "[paste] restore_delay_s must be 0–5")


@dataclass(frozen=True)
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paste: PasteConfig = field(default_factory=PasteConfig)


SECTIONS = {"hotkey": HotkeyConfig, "model": ModelConfig, "paste": PasteConfig}


def _build(cls: type, raw: object, section: str):
    """Type-check one [section] against the dataclass defaults and construct it."""
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    defaults = cls()
    kwargs = {}
    for key, value in raw.items():
        _check(hasattr(defaults, key), f"[{section}] unknown key {key!r}")
        expected = type(getattr(defaults, key))
        if expected is float and type(value) is int:
            value = float(value)
        if expected is tuple and type(value) is list:  # TOML arrays -> immutable tuple
            value = tuple(value)
        _check(type(value) is expected,
               f"[{section}] {key}: expected {expected.__name__}, got {value!r}")
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Read the TOML file (missing file = all defaults). Raises ConfigError if invalid."""
    if not path.exists():
        return Config()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: {e}") from None
    unknown = set(raw) - set(SECTIONS)
    _check(not unknown, f"unknown section(s): {', '.join(sorted(unknown))}")
    return Config(**{name: _build(cls, raw[name], name)
                     for name, cls in SECTIONS.items() if name in raw})
