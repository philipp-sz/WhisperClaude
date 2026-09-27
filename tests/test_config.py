import pytest

from whisperclaude.config import CONFIG_PATH, Config, ConfigError, load_config


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.toml") == Config()


def test_repo_config_matches_defaults():
    """config.toml documents the defaults; keep both in sync."""
    assert load_config(CONFIG_PATH) == Config()


def test_valid_overrides(tmp_path):
    cfg = load_config(write(tmp_path, '[hotkey]\nkey = "f9"\nmax_hold_s = 1\n'
                                      '[model]\nlanguage = "de"\n'))
    assert cfg.hotkey.key == "f9"
    assert cfg.hotkey.max_hold_s == 1.0  # int accepted for float
    assert cfg.model.language == "de"
    assert cfg.model.name == "small"  # untouched default


@pytest.mark.parametrize("text, message", [
    ('[model]\ncpu_threads = 0', "cpu_threads must be 1"),
    ('[model]\ncpu_threads = "4"', "expected int"),
    ('[model]\nbeam_size = true', "expected int"),
    ('[model]\ncompute_type = "int4"', "compute_type"),
    ('[model]\nlanguage = "deu"', "language"),
    ('[hotkey]\nkey = "ctrl_x"', "unknown key"),
    ('[hotkey]\nmax_hold_s = 10', "max_hold_s"),
    ('[paste]\nrestore_delay = 1', "unknown key 'restore_delay'"),
    ('[modle]\nname = "small"', "unknown section"),
    ('[model\n', "config.toml"),
])
def test_invalid_values(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))
