"""Narzędzia do odtwarzania sesji i porównań algorytmów na replayu."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


class SessionReplay:
    """Wczytuje plik sesji JSONL i udostępnia metadane oraz zdarzenia."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.header: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            payload = json.loads(line)
            if index == 0 and payload.get("type") == "session_header":
                self.header = payload
            else:
                self.events.append(payload)

    def iter_sensor_events(self) -> list[dict[str, Any]]:
        """Zwraca tylko zdarzenia z sensorów (EEG/IMU/PPG)."""
        return [e for e in self.events if e.get("type") in {"eeg", "imu", "ppg"}]


class ReplayConnector:
    """Adapter zgodny z interfejsem MuseConnector, ale czyta dane z pliku."""

    def __init__(
        self,
        replay: SessionReplay,
        *,
        on_eeg: Callable[[str, np.ndarray], None],
        on_imu: Callable[[str, np.ndarray], None],
        on_ppg: Callable[[str, np.ndarray], None],
    ) -> None:
        self._replay = replay
        self._on_eeg = on_eeg
        self._on_imu = on_imu
        self._on_ppg = on_ppg
        self._status_callback: Callable[[str], None] = lambda _msg: None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stream_config = {"eeg": True, "accelerometer": True, "gyroscope": True, "ppg": True, "battery": True}
        self._connected = False
        self.devices: list[Any] = []
        self._device_state = {
            "device_name": replay.header.get("device_model", "ReplaySession"),
            "address": "REPLAY_FILE",
            "rssi": None,
            "battery_level": None,
            "sample_rate_hz": replay.header.get("sample_rate_hz", 256),
            "available_sensors": replay.header.get("active_channels", []),
            "streaming": True,
            "connection_state": "STREAMING",
            "reconnect_attempts": 0,
            "stream_activity": {
                "eeg": True,
                "accelerometer": True,
                "gyroscope": True,
                "ppg": True,
                "battery": False,
            },
            "motion_artifact": False,
            "signal_quality": {},
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_state(self) -> dict[str, Any]:
        return self._device_state

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        self._status_callback = callback

    def set_stream_config(self, config: dict[str, bool]) -> None:
        self._stream_config.update({k: bool(v) for k, v in config.items()})

    def start(self) -> None:
        """Uruchamia replay w tle i udaje aktywne połączenie BLE."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ReplayConnector")
        self._thread.start()

    def stop(self) -> None:
        self.disconnect()

    def scan(self, timeout: float = 0.0) -> None:
        """W trybie replay skanowanie nie jest wymagane."""
        del timeout
        self._status_callback("Replay mode: scan skipped")

    def connect(self, _device: Any = None) -> None:
        """W trybie replay połączenie jest logicznie aktywne od startu."""
        self._connected = True

    def disconnect(self) -> None:
        self._stop_event.set()
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        self._status_callback(f"Replay started: {self._replay.path}")
        start = time.monotonic()
        for event in self._replay.iter_sensor_events():
            if self._stop_event.is_set():
                break
            target = float(event.get("t", 0.0))
            while not self._stop_event.is_set() and (time.monotonic() - start) < target:
                time.sleep(0.001)
            etype = event.get("type")
            if etype == "eeg" and self._stream_config.get("eeg", True):
                self._on_eeg(str(event.get("channel", "AF7")), np.asarray(event.get("samples", []), dtype=np.float32))
            elif etype == "imu":
                sensor = str(event.get("sensor", "accelerometer"))
                if self._stream_config.get(sensor, True):
                    self._on_imu(sensor, np.asarray(event.get("samples", []), dtype=np.float32))
            elif etype == "ppg" and self._stream_config.get("ppg", True):
                self._on_ppg(str(event.get("channel", "PPG_IR")), np.asarray(event.get("samples", []), dtype=np.float32))
        self._status_callback("Replay finished")
        self._connected = False


def compare_direction_series(
    replay: SessionReplay,
    *,
    algorithm_a: Callable[[dict[str, Any]], str],
    algorithm_b: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    """Porównuje dwie wersje algorytmu na tych samych oknach sesji replay."""
    compared = 0
    matched = 0
    diffs: list[dict[str, Any]] = []
    for event in replay.events:
        if event.get("type") != "direction_decision":
            continue
        compared += 1
        a_dir = algorithm_a(event)
        b_dir = algorithm_b(event)
        if a_dir == b_dir:
            matched += 1
            continue
        diffs.append({"t": event.get("t", 0.0), "algorithm_a": a_dir, "algorithm_b": b_dir})

    agreement = (matched / compared) if compared else 1.0
    return {"compared_windows": compared, "agreement": agreement, "differences": diffs}
