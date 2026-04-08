"""Centralized application settings with JSON persistence and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_PATH = Path("settings.json")


@dataclass
class AppSettings:
    beta_threshold: float = 2.0
    alpha_threshold: float = 2.0
    asym_factor: float = 1.3
    hysteresis_count: int = 3
    key_mode: str = "arrow"
    forwarding_enabled: bool = False
    debug_logging: bool = False
    debug_eeg_file: bool = False
    debug_logging_enabled: bool = False

    def validate(self) -> None:
        """Validate current settings values and raise ``ValueError`` on invalid values."""
        if not 0.0 <= float(self.beta_threshold) <= 100.0:
            raise ValueError("beta_threshold must be in range 0.0..100.0")
        if not 0.0 <= float(self.alpha_threshold) <= 100.0:
            raise ValueError("alpha_threshold must be in range 0.0..100.0")
        if not 1.0 <= float(self.asym_factor) <= 5.0:
            raise ValueError("asym_factor must be in range 1.0..5.0")
        if not 1 <= int(self.hysteresis_count) <= 20:
            raise ValueError("hysteresis_count must be in range 1..20")
        if self.key_mode not in {"arrow", "wasd"}:
            raise ValueError("key_mode must be either 'arrow' or 'wasd'")
        if not isinstance(self.forwarding_enabled, bool):
            raise ValueError("forwarding_enabled must be a boolean")
        if not isinstance(self.debug_logging, bool):
            raise ValueError("debug_logging must be a boolean")
        if not isinstance(self.debug_eeg_file, bool):
            raise ValueError("debug_eeg_file must be a boolean")
        if not isinstance(self.debug_logging_enabled, bool):
            raise ValueError("debug_logging_enabled must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        defaults = cls()
        merged = {
            "beta_threshold": data.get("beta_threshold", defaults.beta_threshold),
            "alpha_threshold": data.get("alpha_threshold", defaults.alpha_threshold),
            "asym_factor": data.get("asym_factor", defaults.asym_factor),
            "hysteresis_count": data.get("hysteresis_count", defaults.hysteresis_count),
            "key_mode": data.get("key_mode", defaults.key_mode),
            "forwarding_enabled": data.get("forwarding_enabled", defaults.forwarding_enabled),
            "debug_logging": data.get("debug_logging", defaults.debug_logging),
            "debug_eeg_file": data.get("debug_eeg_file", defaults.debug_eeg_file),
            "debug_logging_enabled": data.get(
                "debug_logging_enabled",
                data.get("debug_eeg_file", defaults.debug_logging_enabled),
            ),
        }
        settings = cls(**merged)
        settings.validate()
        return settings


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> AppSettings:
    file_path = Path(path)
    if not file_path.exists():
        return AppSettings()
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Settings JSON root must be an object")
    return AppSettings.from_dict(data)


def save_settings(settings: AppSettings, path: Path | str = DEFAULT_SETTINGS_PATH) -> None:
    settings.validate()
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
