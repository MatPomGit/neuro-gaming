"""Obsługa profili kalibracyjnych EEG zapisywanych per użytkownik i urządzenie."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROFILES_DIR = Path("profiles")


@dataclass
class CalibrationProfile:
    """Struktura pojedynczego profilu kalibracji EEG."""

    profile_id: str
    user_id: str
    device_address: str
    timestamp: str
    firmware_if_available: str | None
    baseline_mean: dict[str, dict[str, float]]
    baseline_std: dict[str, dict[str, float]]
    stage_metrics: dict[str, dict[str, dict[str, float]]]
    initial_thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Zwraca słownik gotowy do serializacji JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationProfile":
        """Buduje obiekt profilu z danych JSON."""
        return cls(
            profile_id=str(data["profile_id"]),
            user_id=str(data.get("user_id", "default-user")),
            device_address=str(data.get("device_address", "unknown-device")),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            firmware_if_available=(
                str(data["firmware_if_available"])
                if data.get("firmware_if_available") not in (None, "")
                else None
            ),
            baseline_mean=data.get("baseline_mean", {}),
            baseline_std=data.get("baseline_std", {}),
            stage_metrics=data.get("stage_metrics", {}),
            initial_thresholds=data.get("initial_thresholds", {}),
        )


class CalibrationProfileStore:
    """Repozytorium profili kalibracyjnych przechowywanych w katalogu ``profiles/``."""

    def __init__(self, root: Path | str = PROFILES_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, profile: CalibrationProfile) -> Path:
        """Zapisuje profil do pliku JSON i zwraca jego ścieżkę."""
        path = self.root / f"{profile.profile_id}.json"
        path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, profile_id: str) -> CalibrationProfile:
        """Wczytuje profil o zadanym identyfikatorze."""
        path = self.root / f"{profile_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Calibration profile JSON root must be an object")
        return CalibrationProfile.from_dict(data)

    def exists(self, profile_id: str) -> bool:
        """Sprawdza, czy plik profilu istnieje."""
        return (self.root / f"{profile_id}.json").exists()

    def list_ids(self) -> list[str]:
        """Zwraca listę dostępnych identyfikatorów profili."""
        return sorted(path.stem for path in self.root.glob("*.json"))
