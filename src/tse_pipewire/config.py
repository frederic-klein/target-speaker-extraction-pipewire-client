"""Configuration management for tse-pipewire."""

import os
from pathlib import Path

import toml

DEFAULTS = {
    "audio": {
        "sample_rate": 48000,
        "tse_sample_rate": 16000,
        "segment_ms": 160,
    },
    "model": {
        "embedding_dim": 256,
    },
    "rnnoise": {
        "vad_threshold": 50.0,
        "vad_grace_period_ms": 200,
    },
}


class Config:
    """Manages tse-pipewire configuration using XDG base directories."""

    def __init__(self):
        xdg_config = os.environ.get(
            "XDG_CONFIG_HOME",
            str(Path.home() / ".config"),
        )
        xdg_data = os.environ.get(
            "XDG_DATA_HOME",
            str(Path.home() / ".local" / "share"),
        )
        self._config_dir = Path(xdg_config) / "tse-pipewire"
        self._data_dir = Path(xdg_data) / "tse-pipewire"
        self._data: dict = {}
        self._apply_defaults()

    def _apply_defaults(self):
        for section, values in DEFAULTS.items():
            if section not in self._data:
                self._data[section] = {}
            for key, value in values.items():
                if key not in self._data[section]:
                    self._data[section][key] = value

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def profiles_dir(self) -> Path:
        return self._config_dir / "profiles"

    @property
    def models_dir(self) -> Path:
        return self._data_dir / "models"

    def profile_path(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.npy"

    def ensure_dirs(self):
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get(self, dotted_key: str, default=None):
        parts = dotted_key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, dotted_key: str, value):
        parts = dotted_key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def save(self):
        self._config_dir.mkdir(parents=True, exist_ok=True)
        config_file = self._config_dir / "config.toml"
        with open(config_file, "w") as f:
            toml.dump(self._data, f)

    def load(self):
        config_file = self._config_dir / "config.toml"
        if config_file.exists():
            with open(config_file) as f:
                self._data = toml.load(f)
        self._apply_defaults()
