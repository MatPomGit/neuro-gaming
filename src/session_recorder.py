"""Rejestrator sesji EEG oraz narzędzia CSV/replay dla ekranu diagnostycznego."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SessionSample:
    """Pojedyncza próbka stanu sesji zapisana przez UI tick."""

    relative_time: float
    direction: str
    connected: bool
    motion_artifact: bool
    alpha_left: float
    alpha_right: float
    beta_left: float
    beta_right: float
    quality_avg: float


class SessionRecorder:
    """Lekki rejestrator sesji działający w pamięci procesu.

    Klasa zapisuje próbki w postaci listy, umożliwia eksport CSV,
    a także zwraca dane do prostego replayu na ekranie diagnostycznym.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._session_started_monotonic = 0.0
        self._session_name = ""
        self._samples: list[SessionSample] = []

    @property
    def active(self) -> bool:
        """Informacja, czy nagrywanie jest aktualnie aktywne."""
        with self._lock:
            return self._active

    def start(self, *, now_monotonic: float, session_name: str | None = None) -> str:
        """Rozpoczyna nową sesję i czyści poprzednie próbki."""
        with self._lock:
            self._active = True
            self._session_started_monotonic = now_monotonic
            self._session_name = session_name or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
            self._samples = []
            return self._session_name

    def stop(self) -> int:
        """Kończy nagrywanie i zwraca liczbę zapisanych próbek."""
        with self._lock:
            self._active = False
            return len(self._samples)

    def clear(self) -> None:
        """Usuwa wszystkie próbki bez zmiany stanu aktywności."""
        with self._lock:
            self._samples = []

    def record_sample(
        self,
        *,
        now_monotonic: float,
        direction: str,
        connected: bool,
        motion_artifact: bool,
        metrics: dict[str, dict[str, float]],
        signal_quality: dict[str, float],
    ) -> None:
        """Dopisuje próbkę do sesji, jeśli nagrywanie jest aktywne."""
        with self._lock:
            if not self._active:
                return
            relative_time = max(0.0, now_monotonic - self._session_started_monotonic)
            alpha_left = float(metrics.get("AF7", {}).get("alpha", 0.0))
            alpha_right = float(metrics.get("AF8", {}).get("alpha", 0.0))
            beta_left = float(metrics.get("AF7", {}).get("beta", 0.0))
            beta_right = float(metrics.get("AF8", {}).get("beta", 0.0))
            if signal_quality:
                quality_avg = float(sum(signal_quality.values()) / max(1, len(signal_quality)))
            else:
                quality_avg = 0.0
            self._samples.append(
                SessionSample(
                    relative_time=relative_time,
                    direction=direction,
                    connected=connected,
                    motion_artifact=motion_artifact,
                    alpha_left=alpha_left,
                    alpha_right=alpha_right,
                    beta_left=beta_left,
                    beta_right=beta_right,
                    quality_avg=quality_avg,
                )
            )

    def snapshot(self) -> dict[str, Any]:
        """Zwraca podsumowanie stanu sesji do wyświetlenia w UI."""
        with self._lock:
            duration = self._samples[-1].relative_time if self._samples else 0.0
            return {
                "session_name": self._session_name,
                "active": self._active,
                "samples": len(self._samples),
                "duration": duration,
            }

    def replay_data(self) -> list[dict[str, Any]]:
        """Zwraca próbki w formacie gotowym do replayu w UI."""
        with self._lock:
            return [
                {
                    "t": item.relative_time,
                    "direction": item.direction,
                    "quality": item.quality_avg,
                    "motion_artifact": item.motion_artifact,
                }
                for item in self._samples
            ]

    def export_csv(self, destination: Path | str) -> Path:
        """Eksportuje całą sesję do CSV i zwraca ścieżkę pliku."""
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self._samples)
            session_name = self._session_name

        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            # Komentarz JSON ułatwia późniejsze parsowanie metadanych.
            fh.write(f"# {json.dumps({'session_name': session_name}, ensure_ascii=False)}\n")
            writer.writerow(
                [
                    "relative_time",
                    "direction",
                    "connected",
                    "motion_artifact",
                    "alpha_left",
                    "alpha_right",
                    "beta_left",
                    "beta_right",
                    "quality_avg",
                ]
            )
            for sample in rows:
                writer.writerow(
                    [
                        f"{sample.relative_time:.4f}",
                        sample.direction,
                        int(sample.connected),
                        int(sample.motion_artifact),
                        f"{sample.alpha_left:.6f}",
                        f"{sample.alpha_right:.6f}",
                        f"{sample.beta_left:.6f}",
                        f"{sample.beta_right:.6f}",
                        f"{sample.quality_avg:.6f}",
                    ]
                )
        return output
