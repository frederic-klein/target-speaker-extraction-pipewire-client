"""Tests for configuration management."""

import stat
from pathlib import Path

from tse_pipewire.config import Config


def test_default_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    assert config.config_dir == tmp_path / "tse-pipewire"


def test_default_config_dir_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = Config()
    assert config.config_dir == tmp_path / ".config" / "tse-pipewire"


def test_profiles_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    assert config.profiles_dir == tmp_path / "tse-pipewire" / "profiles"


def test_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    config = Config()
    assert config.models_dir == tmp_path / "tse-pipewire" / "models"


def test_models_dir_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = Config()
    assert config.models_dir == tmp_path / ".local" / "share" / "tse-pipewire" / "models"


def test_save_and_load_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    config.set("audio.sample_rate", 48000)
    config.set("audio.segment_ms", 160)
    config.save()

    loaded = Config()
    loaded.load()
    assert loaded.get("audio.sample_rate") == 48000
    assert loaded.get("audio.segment_ms") == 160


def test_get_default_value(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    assert config.get("nonexistent.key", default=42) == 42


def test_get_without_default_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    assert config.get("nonexistent.key") is None


def test_default_values(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    assert config.get("audio.sample_rate") == 48000
    assert config.get("audio.tse_sample_rate") == 16000
    assert config.get("audio.segment_ms") == 160
    assert config.get("model.embedding_dim") == 256


def test_ensure_dirs_creates_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    config = Config()
    config.ensure_dirs()
    assert config.config_dir.is_dir()
    assert config.profiles_dir.is_dir()
    assert config.models_dir.is_dir()


def test_profile_path(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    path = config.profile_path("frederic")
    assert path == tmp_path / "tse-pipewire" / "profiles" / "frederic.npy"


def test_set_nested_key(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    config.set("rnnoise.vad_threshold", 50.0)
    assert config.get("rnnoise.vad_threshold") == 50.0


def test_ensure_dirs_permissions_700(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    config = Config()
    config.ensure_dirs()

    for d in [config.config_dir, config.profiles_dir, config.models_dir]:
        assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_save_config_file_permissions_600(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = Config()
    config.save()

    config_file = config.config_dir / "config.toml"
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
