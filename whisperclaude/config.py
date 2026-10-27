"""Load and validate config.toml. Every key is optional; defaults = benchmark results."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from whisperclaude.hotkey import parse_key
from whisperclaude.constants import DEFAULT_PROMPT, DEFAULT_VOCABULARY

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
COMPUTE_TYPES = {"int8", "int8_float32", "int16", "float32"}
TRAY_STYLES = ("classic", "pulse", "bars", "ring")  # = tray_icons.STYLES (not imported: PIL)
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
    cancel_key: str = "esc"   # pressed while recording: discard the recording; "" = off

    def __post_init__(self) -> None:
        for name in ("key", "cancel_key"):
            value = getattr(self, name)
            if name == "cancel_key" and value == "":
                continue
            try:
                parse_key(value)
            except ValueError as e:
                raise ConfigError(f"[hotkey] {name}: {e}") from None
        _check(self.cancel_key != self.key, "[hotkey] cancel_key must differ from key")
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
class RecordingConfig:
    stop_after_silence_s: float = 60.0  # forgotten recording: stop, copy the text; 0 = never

    def __post_init__(self) -> None:
        _check(self.stop_after_silence_s == 0 or 10 <= self.stop_after_silence_s <= 3600,
               "[recording] stop_after_silence_s must be 0 (off) or 10–3600")


@dataclass(frozen=True)
class HistoryConfig:
    keep: int = 5  # last dictations saved locally (tray "Recent dictations"); 0 = off

    def __post_init__(self) -> None:
        _check(0 <= self.keep <= 50, "[history] keep must be 0 (off) to 50")


@dataclass(frozen=True)
class TrayConfig:
    style: str = "bars"  # tray icon look, animated except "classic" (the original static icon)

    def __post_init__(self) -> None:
        _check(self.style in TRAY_STYLES, f"[tray] style must be one of {list(TRAY_STYLES)}")


@dataclass(frozen=True)
class PasteConfig:
    restore_delay_s: float = 0.3  # wait before restoring the old clipboard

    def __post_init__(self) -> None:
        _check(0 <= self.restore_delay_s <= 5, "[paste] restore_delay_s must be 0–5")


@dataclass(frozen=True)
class CommandsConfig:
    enabled: bool = False  # opt-in: turn on in config.toml
    trigger: str = "command"  # say "<trigger> new line", "<trigger> bullet", ...
    bullet: str = "- "        # what "<trigger> bullet" writes at the start of the line

    def __post_init__(self) -> None:
        _check(re.fullmatch(r"[A-Za-zÄÖÜäöüß]+( [A-Za-zÄÖÜäöüß]+)*", self.trigger) is not None
               and len(self.trigger) <= 30,
               "[commands] trigger must be one or a few plain words (letters and spaces)")
        _check(1 <= len(self.bullet) <= 4 and not any(c in self.bullet for c in "\r\n"),
               "[commands] bullet must be 1-4 characters on one line")


@dataclass(frozen=True)
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)
    paste: PasteConfig = field(default_factory=PasteConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)


SECTIONS = {"hotkey": HotkeyConfig, "model": ModelConfig, "recording": RecordingConfig,
            "history": HistoryConfig, "tray": TrayConfig, "paste": PasteConfig,
            "commands": CommandsConfig}


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
