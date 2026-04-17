"""Rejestrator sesji sygnałów EEG/IMU/PPG oraz zdarzeń sterowania."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSION_FORMAT_VERSION = "1.0"


@dataclass(slots=True)
class SessionSample:
    """Uproszczona próbka stanu używana przez ekran diagnostyczny."""

    relative_time: float
    direction: str
    connected: bool
    motion_artifact: bool
    alpha_left: float
    alpha_right: float
    beta_left: float
    beta_right: float
    quality_avg: float
    confidence: float
    rejected_window: bool
    decision_latency_ms: float


class SessionRecorder:
    """Rejestrator sesji działający w pamięci procesu.

    Oprócz prostych próbek do UI zapisuje też surowe ramki EEG/IMU/PPG,
    zdarzenia sterowania oraz metadane potrzebne do pełnego replayu.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._session_started_monotonic = 0.0
        self._session_name = ""
        self._samples: list[SessionSample] = []
        self._event_log: list[dict[str, Any]] = []
        self._session_metadata: dict[str, Any] = {}
        self._last_eeg_relative_time: float | None = None

    @property
    def active(self) -> bool:
        """Informacja, czy nagrywanie jest aktualnie aktywne."""
        with self._lock:
            return self._active

    def start(
        self,
        *,
        now_monotonic: float,
        session_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Rozpoczyna nową sesję i czyści poprzednie próbki."""
        with self._lock:
            self._active = True
            self._session_started_monotonic = now_monotonic
            self._session_name = session_name or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
            self._samples = []
            self._event_log = []
            self._last_eeg_relative_time = None
            self._session_metadata = {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "session_name": self._session_name,
                "format_version": SESSION_FORMAT_VERSION,
            }
            if metadata:
                self._session_metadata.update(metadata)
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
            self._event_log = []

    def _relative_time(self, now_monotonic: float) -> float:
        return max(0.0, now_monotonic - self._session_started_monotonic)

    def _record_event(self, payload: dict[str, Any]) -> None:
        if not self._active:
            return
        self._event_log.append(payload)

    def record_eeg_frame(self, *, now_monotonic: float, channel: str, samples: list[float]) -> None:
        """Zapisuje surową ramkę EEG jako zdarzenie replay."""
        with self._lock:
            if not self._active:
                return
            rel = self._relative_time(now_monotonic)
            self._last_eeg_relative_time = rel
            self._record_event({"t": rel, "type": "eeg", "channel": channel, "samples": samples})

    def record_imu_frame(self, *, now_monotonic: float, sensor: str, samples: list[list[float]]) -> None:
        """Zapisuje surową ramkę IMU."""
        with self._lock:
            if not self._active:
                return
            self._record_event({"t": self._relative_time(now_monotonic), "type": "imu", "sensor": sensor, "samples": samples})

    def record_ppg_frame(self, *, now_monotonic: float, channel: str, samples: list[float]) -> None:
        """Zapisuje surową ramkę PPG."""
        with self._lock:
            if not self._active:
                return
            self._record_event({"t": self._relative_time(now_monotonic), "type": "ppg", "channel": channel, "samples": samples})

    def record_control_event(self, *, now_monotonic: float, event_name: str, payload: dict[str, Any] | None = None) -> None:
        """Zapisuje zdarzenie sterowania (klawisz/mysz/UI)."""
        with self._lock:
            if not self._active:
                return
            self._record_event({
                "t": self._relative_time(now_monotonic),
                "type": "control_event",
                "event_name": event_name,
                "payload": payload or {},
            })

    def record_sample(
        self,
        *,
        now_monotonic: float,
        direction: str,
        connected: bool,
        motion_artifact: bool,
        metrics: dict[str, dict[str, float]],
        signal_quality: dict[str, float],
        confidence: float = 0.0,
        rejected_window: bool = False,
    ) -> None:
        """Dopisuje próbkę decyzji kierunku oraz metryki okna."""
        with self._lock:
            if not self._active:
                return
            relative_time = self._relative_time(now_monotonic)
            alpha_left = float(metrics.get("AF7", {}).get("alpha", 0.0))
            alpha_right = float(metrics.get("AF8", {}).get("alpha", 0.0))
            beta_left = float(metrics.get("AF7", {}).get("beta", 0.0))
            beta_right = float(metrics.get("AF8", {}).get("beta", 0.0))
            if signal_quality:
                quality_avg = float(sum(signal_quality.values()) / max(1, len(signal_quality)))
            else:
                quality_avg = 0.0
            latency_ms = 0.0
            if self._last_eeg_relative_time is not None:
                latency_ms = max(0.0, (relative_time - self._last_eeg_relative_time) * 1000.0)

            sample = SessionSample(
                relative_time=relative_time,
                direction=direction,
                connected=connected,
                motion_artifact=motion_artifact,
                alpha_left=alpha_left,
                alpha_right=alpha_right,
                beta_left=beta_left,
                beta_right=beta_right,
                quality_avg=quality_avg,
                confidence=float(confidence),
                rejected_window=bool(rejected_window),
                decision_latency_ms=float(latency_ms),
            )
            self._samples.append(sample)
            self._record_event(
                {
                    "t": relative_time,
                    "type": "direction_decision",
                    "direction": sample.direction,
                    "connected": sample.connected,
                    "motion_artifact": sample.motion_artifact,
                    "metrics": {
                        "AF7": {"alpha": sample.alpha_left, "beta": sample.beta_left},
                        "AF8": {"alpha": sample.alpha_right, "beta": sample.beta_right},
                    },
                    "quality_avg": sample.quality_avg,
                    "confidence": sample.confidence,
                    "rejected_window": sample.rejected_window,
                    "decision_latency_ms": sample.decision_latency_ms,
                }
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

    def build_report(self) -> dict[str, Any]:
        """Liczy raport po sesji: opóźnienie, stabilność i odrzuty okien."""
        with self._lock:
            latencies = [s.decision_latency_ms for s in self._samples if s.connected]
            rejected = sum(1 for s in self._samples if s.rejected_window)
            decisions = [s.direction for s in self._samples if s.connected]

        transitions = 0
        for i in range(1, len(decisions)):
            if decisions[i] != decisions[i - 1]:
                transitions += 1
        stability = 1.0
        if len(decisions) > 1:
            stability = 1.0 - (transitions / (len(decisions) - 1))

        return {
            "session_name": self._session_name,
            "format_version": SESSION_FORMAT_VERSION,
            "samples": len(self._samples),
            "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "direction_stability": max(0.0, min(1.0, stability)),
            "rejected_windows": rejected,
        }

    def export_session(self, destination: Path | str) -> Path:
        """Eksportuje pełny zapis sesji do formatu JSON Lines."""
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            metadata = dict(self._session_metadata)
            events = list(self._event_log)

        with output.open("w", encoding="utf-8") as fh:
            header = {"type": "session_header", **metadata}
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return output

    def export_report(self, destination: Path | str) -> Path:
        """Eksportuje raport po sesji do pliku JSON."""
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.build_report(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def export_csv(self, destination: Path | str) -> Path:
        """Eksportuje uproszczoną serię decyzji do CSV."""
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self._samples)
            session_name = self._session_name

        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
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
                    "confidence",
                    "rejected_window",
                    "decision_latency_ms",
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
                        f"{sample.confidence:.6f}",
                        int(sample.rejected_window),
                        f"{sample.decision_latency_ms:.3f}",
                    ]
                )
        return output
