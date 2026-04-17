"""Wspólne modele danych używane przez konektor Muse i warstwę UI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EEGFrame:
    """Pojedyncza ramka EEG z jednego kanału.

    Atrybut ``samples`` przechowuje pięć próbek napięcia (µV),
    zgodnie z formatem notyfikacji BLE Muse.
    """

    channel: str
    sequence: int
    samples: np.ndarray
    timestamp: float


@dataclass(slots=True)
class IMUFrame:
    """Ramka IMU (akcelerometr lub żyroskop).

    ``samples`` ma kształt ``(N, 3)`` i zawiera osie ``X/Y/Z``.
    Wartości są już przeskalowane do jednostek fizycznych.
    """

    sensor: str
    sequence: int
    samples: np.ndarray
    timestamp: float


@dataclass(slots=True)
class PPGFrame:
    """Ramka PPG dla kanału optycznego Muse (Ambient/IR/Red)."""

    channel: str
    sequence: int
    samples: np.ndarray
    timestamp: float


@dataclass(slots=True)
class DeviceTelemetry:
    """Metadane urządzenia przekazywane do UI i logiki aplikacji."""

    battery_level: int | None
    stream_activity: dict[str, bool]
    signal_quality: dict[str, float]
    motion_artifact: bool
    timestamp: float
