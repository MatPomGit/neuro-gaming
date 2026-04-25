"""
Muse S Athena BLE connector.

Scans for a Muse S (or Muse 2) headband via Bluetooth Low Energy,
connects to it, and streams raw EEG data from four electrode channels
(TP9, AF7, AF8, TP10).  Parsed samples are delivered to a caller-
supplied callback on the asyncio event loop that is running inside a
dedicated background thread.

Previously connected devices are persisted to a JSON file and
automatically searched for on the next scan / auto-connect call.
"""

import asyncio
import json
import logging
import os
import inspect
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections.abc import Callable
from concurrent.futures import Future
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from src.data_models import DeviceTelemetry, IMUFrame, PPGFrame

logger = logging.getLogger(__name__)
_EEG_DEBUG_LOGGER_NAME = "neuro_gaming.eeg_debug"

# Default path for the known-devices store
_DEFAULT_STORE_PATH = os.path.join(os.path.expanduser("~"), ".neuro_gaming_devices.json")


def get_eeg_debug_log_path(
    app_name: str = "neuro_gaming",
    *,
    os_name: str | None = None,
    platform: str | None = None,
) -> Path:
    """Return a platform-safe path for EEG debug logs."""
    os_name = os_name or os.name
    platform = platform or sys.platform
    home = Path.home()
    if os_name == "nt":
        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    elif platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", home / ".local" / "state"))
    return base / app_name / "logs" / "eeg_debug.log"


def configure_eeg_debug_logger(
    *,
    enabled: bool = False,
    log_path: Path | str | None = None,
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 5,
) -> tuple[logging.Logger, Optional[Path]]:
    """Configure a rotating file logger dedicated to EEG samples."""
    eeg_logger = logging.getLogger(_EEG_DEBUG_LOGGER_NAME)
    eeg_logger.setLevel(logging.INFO)
    eeg_logger.propagate = False

    for handler in list(eeg_logger.handlers):
        eeg_logger.removeHandler(handler)
        handler.close()

    if not enabled:
        eeg_logger.disabled = True
        return eeg_logger, None

    resolved_path = Path(log_path) if log_path is not None else get_eeg_debug_log_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        resolved_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    eeg_logger.addHandler(handler)
    eeg_logger.disabled = False
    return eeg_logger, resolved_path

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Muse BLE UUIDs
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MUSE_SERVICE_UUID = "0000fe8d-0000-1000-8000-00805f9b34fb"

# Control characteristic â€“ used to start / stop EEG streaming
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"

# One notification characteristic per EEG channel (5 samples per packet)
EEG_UUIDS: dict[str, str] = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}

# Command sent to start EEG streaming
CMD_START = b"\x02\x64\x0a"
# Command sent to stop EEG streaming
CMD_STOP = b"\x02\x68\x0a"

# Sampling rate of the Muse S (Hz)
SAMPLE_RATE = 256
BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
GYRO_UUID = "273e0009-4c4d-454d-96be-f03bac821358"
ACCEL_UUID = "273e000a-4c4d-454d-96be-f03bac821358"
PPG_UUIDS: dict[str, str] = {
    "PPG_AMBIENT": "273e000f-4c4d-454d-96be-f03bac821358",
    "PPG_IR": "273e0010-4c4d-454d-96be-f03bac821358",
    "PPG_RED": "273e0011-4c4d-454d-96be-f03bac821358",
}
# Maksymalny czas (sekundy) oczekiwania na pojawienie siÄ™ charakterystyk EEG.
SERVICE_DISCOVERY_TIMEOUT_SECONDS = 30
STREAM_WATCHDOG_TIMEOUT_SECONDS = 3.0
RECONNECT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
# Maksymalna liczba prób połączenia w ramach pojedynczego wywołania connect.
CONNECT_RETRY_ATTEMPTS = 3
# Krótki backoff pomiędzy kolejnymi próbami zestawienia sesji BLE.
CONNECT_RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)

# Known Muse data characteristics (used for capability reporting in UI)
SENSOR_CHARACTERISTICS: dict[str, set[str]] = {
    "EEG": {
        "273e0003-4c4d-454d-96be-f03bac821358",
        "273e0004-4c4d-454d-96be-f03bac821358",
        "273e0005-4c4d-454d-96be-f03bac821358",
        "273e0006-4c4d-454d-96be-f03bac821358",
    },
    "Gyroscope": {"273e0009-4c4d-454d-96be-f03bac821358"},
    "Accelerometer": {"273e000a-4c4d-454d-96be-f03bac821358"},
    "PPG": {
        "273e000f-4c4d-454d-96be-f03bac821358",
        "273e0010-4c4d-454d-96be-f03bac821358",
        "273e0011-4c4d-454d-96be-f03bac821358",
    },
}
_EEG_UUID_SET = {uuid.lower() for uuid in EEG_UUIDS.values()}


class ConnectionState(str, Enum):
    """Stany poĹ‚Ä…czenia z Muse wykorzystywane przez centralnÄ… maszynÄ™ stanĂłw."""

    IDLE = "IDLE"
    SCANNING = "SCANNING"
    CONNECTING = "CONNECTING"
    STREAMING = "STREAMING"
    RECOVERING = "RECOVERING"
    ERROR = "ERROR"


@dataclass(slots=True)
class SessionMetrics:
    """Metryki jakoĹ›ci sesji EEG logowane po zakoĹ„czeniu poĹ‚Ä…czenia."""

    connected_since: float | None = None
    reconnect_count: int = 0
    sample_interval_sum: float = 0.0
    sample_interval_count: int = 0
    total_samples: int = 0
    dropout_samples: int = 0
    last_sample_monotonic: float | None = None
    sequence_by_channel: dict[str, int] = field(default_factory=dict)
    logged: bool = False

    def reset(self) -> None:
        """CzyĹ›ci metryki przy starcie nowej sesji."""
        self.connected_since = None
        self.reconnect_count = 0
        self.sample_interval_sum = 0.0
        self.sample_interval_count = 0
        self.total_samples = 0
        self.dropout_samples = 0
        self.last_sample_monotonic = None
        self.sequence_by_channel = {}
        self.logged = False

    @property
    def average_sample_interval(self) -> float:
        """Zwraca Ĺ›redni interwaĹ‚ pomiÄ™dzy kolejnymi callbackami EEG."""
        if self.sample_interval_count == 0:
            return 0.0
        return self.sample_interval_sum / self.sample_interval_count

    @property
    def dropout_percent(self) -> float:
        """Szacuje procent brakujÄ…cych prĂłbek wzglÄ™dem caĹ‚ej transmisji."""
        expected = self.total_samples + self.dropout_samples
        if expected == 0:
            return 0.0
        return (self.dropout_samples / expected) * 100.0


@dataclass(slots=True)
class ConnectorDevice:
    """Lekki opis urzÄ…dzenia, uĹĽywany m.in. przy reconnect."""

    address: str
    name: str = "Muse"
    rssi: int | None = None

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Known-devices persistent store
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class KnownDevicesStore:
    """Persists the addresses and names of previously connected Muse devices.

    Entries are stored as a JSON array in a plain file, one object per device::

        [{"address": "AA:BB:CC:DD:EE:FF", "name": "Muse-1234"}, ...]

    Parameters
    ----------
    path:
        Absolute path to the JSON file used for storage.  The file is
        created on first save if it does not already exist.
    """

    def __init__(self, path: str = _DEFAULT_STORE_PATH) -> None:
        self._path = path
        self._entries: list[dict[str, str]] = self._load()

    # â”€â”€ public interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def path(self) -> str:
        """Path to the backing JSON file."""
        return self._path

    def all(self) -> list[dict[str, str]]:
        """Return a copy of all stored device entries."""
        return list(self._entries)

    def addresses(self) -> set[str]:
        """Return the set of all stored device addresses (upper-cased)."""
        return {e["address"].upper() for e in self._entries}

    def save(self, address: str, name: str) -> None:
        """Persist *address* / *name*.  Updates an existing entry if the
        address is already known; appends a new entry otherwise."""
        address = address.upper()
        for entry in self._entries:
            if entry["address"].upper() == address:
                entry["name"] = name
                break
        else:
            self._entries.append({"address": address, "name": name})
        self._write()

    def remove(self, address: str) -> bool:
        """Remove the entry for *address*.  Returns ``True`` if it existed."""
        address = address.upper()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["address"].upper() != address]
        if len(self._entries) < before:
            self._write()
            return True
        return False

    def clear(self) -> None:
        """Remove all stored entries."""
        self._entries = []
        self._write()

    # â”€â”€ private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load(self) -> list[dict[str, str]]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [
                    {"address": str(e.get("address", "")), "name": str(e.get("name", ""))}
                    for e in data
                    if isinstance(e, dict) and e.get("address")
                ]
        except FileNotFoundError:
            pass
        except Exception as exc:  # malformed JSON or permission error
            logger.warning("Could not load known devices from %s: %s", self._path, exc)
        return []

    def _write(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._entries, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not save known devices to %s: %s", self._path, exc)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Packet parser
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_eeg_packet(data: bytes) -> tuple[int, np.ndarray]:
    """Decode one EEG BLE notification.

    The Muse S encodes five 12-bit samples per notification packet:
      bytes 0-1  â€“ big-endian 16-bit sequence number
      bytes 2-9  â€“ five 12-bit sample values, packed big-endian

    Raw sample values are converted to ÂµV using:
        voltage = (raw - 2048) Ă— 0.48828125

    Parameters
    ----------
    data:
        Raw bytes received from the BLE notification.

    Returns
    -------
    sequence:
        Packet sequence number.
    samples:
        NumPy array of 5 voltage samples in ÂµV.
    """
    sequence = struct.unpack(">H", data[:2])[0]
    payload = data[2:]
    samples: list[float] = []
    for i in range(5):
        bit_pos = i * 12
        byte_pos = bit_pos // 8
        bit_offset = bit_pos % 8
        if bit_offset == 0:
            raw = ((payload[byte_pos] << 4) | (payload[byte_pos + 1] >> 4)) & 0xFFF
        else:
            raw = ((payload[byte_pos] & 0x0F) << 8) | payload[byte_pos + 1]
        samples.append((raw - 2048) * 0.48828125)
    return sequence, np.array(samples, dtype=np.float32)


def _parse_imu_packet(data: bytes, *, scale: float) -> tuple[int, np.ndarray]:
    """Dekoduje pakiet IMU do tablicy ``NĂ—3`` (XYZ)."""
    if len(data) < 8:
        raise ValueError("IMU packet too short")
    values = struct.unpack(f">{len(data)//2}h", data[: (len(data) // 2) * 2])
    sequence = values[0] & 0xFFFF
    payload = values[1:]
    usable = (len(payload) // 3) * 3
    if usable == 0:
        return sequence, np.zeros((0, 3), dtype=np.float32)
    arr = np.array(payload[:usable], dtype=np.float32).reshape(-1, 3)
    return sequence, arr * scale


def _parse_ppg_packet(data: bytes) -> tuple[int, np.ndarray]:
    """Dekoduje pakiet PPG (2B sequence + prĂłbki 24-bit)."""
    if len(data) < 5:
        raise ValueError("PPG packet too short")
    sequence = struct.unpack(">H", data[:2])[0]
    payload = data[2:]
    sample_count = len(payload) // 3
    samples: list[float] = []
    for i in range(sample_count):
        chunk = payload[i * 3: (i + 1) * 3]
        samples.append(float(int.from_bytes(chunk, "big", signed=False)))
    return sequence, np.array(samples, dtype=np.float32)


def _parse_battery_payload(data: bytes) -> int:
    """Parsuje poziom baterii (0-100%) z charakterystyki BLE."""
    if not data:
        raise ValueError("Battery payload is empty")
    return int(max(0, min(100, data[0])))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Connector class
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MuseConnector:
    """Manages Bluetooth connection to a Muse S Athena device.

    Previously connected devices are automatically remembered and can be
    reconnected via :meth:`auto_connect`. Status messages are exposed via
    a callback that can be set or replaced at runtime using
    :meth:`set_status_callback`.

    Usage
    -----
    >>> connector = MuseConnector(on_eeg=my_callback)
    >>> connector.start()          # spawns background thread + asyncio loop
    >>> connector.scan()           # populate .devices
    >>> connector.connect(device)  # start streaming (device is remembered)
    >>> connector.auto_connect()   # scan + connect to any known device
    >>> connector.disconnect()
    >>> connector.stop()           # clean shutdown
    """

    def __init__(
        self,
        on_eeg: Callable[[str, np.ndarray], None],
        on_imu: Optional[Callable[[IMUFrame], None]] = None,
        on_ppg: Optional[Callable[[PPGFrame], None]] = None,
        on_telemetry: Optional[Callable[[DeviceTelemetry], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        known_devices_path: Optional[str] = None,
        debug_logging_enabled: bool = False,
        debug_log_path: Path | str | None = None,
        stream_config: Optional[dict[str, bool]] = None,
    ) -> None:
        """
        Parameters
        ----------
        on_eeg:
            Called with ``(channel_name, samples)`` whenever new EEG data
            arrives.  *channel_name* is one of ``"TP9"``, ``"AF7"``,
            ``"AF8"``, ``"TP10"``.  *samples* is a float32 array of 5 ÂµV
            values.  This callback is invoked from the background thread.
        on_status:
            Optional callback for human-readable status messages. Can also
            be set later via :meth:`set_status_callback`.
        known_devices_path:
            Optional path to the JSON file used to persist previously
            connected devices.  Defaults to ``~/.neuro_gaming_devices.json``.
        debug_logging_enabled:
            Enables/disables diagnostic EEG logging to a rotating file.
        debug_log_path:
            Optional override for the EEG debug log file location.
        """
        self._on_eeg = on_eeg
        self._on_imu = on_imu or (lambda _frame: None)
        self._on_ppg = on_ppg or (lambda _frame: None)
        self._on_telemetry = on_telemetry or (lambda _telemetry: None)
        self._status_callback = on_status or (lambda _: None)
        self._store = KnownDevicesStore(known_devices_path or _DEFAULT_STORE_PATH)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()
        self._client: Optional[BleakClient] = None
        self._connected = False
        self._battery_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._recovering_task: Optional[asyncio.Task | Future] = None
        self._manual_disconnect_requested = False
        self._state = ConnectionState.IDLE
        self._state_lock = threading.Lock()
        self._active_device: ConnectorDevice | None = None
        self._session_metrics = SessionMetrics()
        self._watchdog_timeout_seconds = STREAM_WATCHDOG_TIMEOUT_SECONDS
        self._reconnect_backoff_seconds = RECONNECT_BACKOFF_SECONDS
        self._device_state = {
            "device_name": "Unknown",
            "address": "",
            "rssi": None,
            "battery_level": None,
            "sample_rate_hz": SAMPLE_RATE,
            "available_sensors": [],
            "streaming": False,
            "connection_state": self._state.value,
            "reconnect_attempts": 0,
            "stream_activity": {
                "eeg": False,
                "accelerometer": False,
                "gyroscope": False,
                "ppg": False,
                "battery": False,
            },
            "motion_artifact": False,
            "signal_quality": {},
        }
        self._stream_config = {
            "eeg": True,
            "accelerometer": True,
            "gyroscope": True,
            "ppg": True,
            "battery": True,
        }
        if stream_config:
            self._stream_config.update({k: bool(v) for k, v in stream_config.items()})
        self._active_notifications: set[str] = set()

        self.devices: list[BLEDevice] = []
        self._debug_logging_enabled = debug_logging_enabled
        self._eeg_debug_logger, self._debug_log_path = configure_eeg_debug_logger(
            enabled=debug_logging_enabled,
            log_path=debug_log_path,
        )
        if self._debug_log_path:
            logger.info("EEG debug log path: %s", self._debug_log_path)

    # â”€â”€ lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def start(self) -> None:
        """Start the background asyncio thread."""
        with self._start_lock:
            if self._loop is not None and self._loop.is_running():
                return
            self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="MuseBLELoop",
        )
        self._thread.start()

    def stop(self) -> None:
        """Disconnect and shut down the background thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    # â”€â”€ public API (thread-safe) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def scan(self, timeout: float = 5.0) -> None:
        """Scan for nearby Muse devices.  Populates ``self.devices``."""
        if self._loop is None:
            raise RuntimeError("Call start() before scan()")
        future = asyncio.run_coroutine_threadsafe(
            self._async_scan(timeout), self._loop
        )
        future.result(timeout=timeout + 2)

    def connect(self, device: BLEDevice) -> None:
        """Connect to *device* and start EEG streaming."""
        if self._loop is None:
            raise RuntimeError("Call start() before connect()")
        future = asyncio.run_coroutine_threadsafe(
            self._async_connect(device), self._loop
        )
        # Utrzymujemy spĂłjny timeout caĹ‚ego poĹ‚Ä…czenia z timeoutem discovery.
        future.result(timeout=SERVICE_DISCOVERY_TIMEOUT_SECONDS + 10)

    def disconnect(self) -> None:
        """Stop EEG streaming and disconnect from the device."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_disconnect(), self._loop
            ).result(timeout=5)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_state(self) -> dict:
        sensors = list(self._device_state.get("available_sensors", []))
        return {**self._device_state, "available_sensors": sensors}

    @property
    def known_devices(self) -> list[dict[str, str]]:
        """Return a copy of all previously connected devices.

        Each entry is a dict with keys ``"address"`` and ``"name"``.
        """
        return self._store.all()

    def auto_connect(self, timeout: float = 10.0) -> bool:
        """Scan for nearby devices and connect to the first known one found.

        A device is considered *known* if its address was previously saved
        by a successful :meth:`connect` call.

        Parameters
        ----------
        timeout:
            BLE scan duration in seconds.

        Returns
        -------
        bool
            ``True`` if a known device was found and connected successfully,
            ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If :meth:`start` has not been called.
        """
        if self._loop is None:
            raise RuntimeError("Call start() before auto_connect()")
        future = asyncio.run_coroutine_threadsafe(
            self._async_auto_connect(timeout), self._loop
        )
        return future.result(timeout=timeout + 15)

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        """Register or replace the status callback used by the connector.

        Parameters
        ----------
        callback:
            Callable invoked with a single human-readable status message.
        """
        self._status_callback = callback

    def set_stream_config(self, config: dict[str, bool]) -> None:
        """Aktualizuje konfiguracjÄ™ aktywnych strumieni bez restartu aplikacji."""
        self._stream_config.update({k: bool(v) for k, v in config.items()})
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._apply_stream_config(), self._loop)

    # â”€â”€ async implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _emit_status(self, message: str) -> None:
        """Emit a human-readable status message via the configured callback."""
        self._status_callback(message)

    def _transition_state(self, new_state: ConnectionState, reason: str = "") -> None:
        """Centralny handler przejĹ›Ä‡ stanĂłw i raportowania do UI."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._device_state["connection_state"] = new_state.value
            if new_state != ConnectionState.STREAMING:
                self._device_state["streaming"] = False

        if old_state != new_state:
            logger.info(
                "Connection state transition: %s -> %s%s",
                old_state.value,
                new_state.value,
                f" ({reason})" if reason else "",
            )
        status_message = (
            f"[{new_state.value}] {reason}" if reason else f"[{new_state.value}]"
        )
        self._emit_status(status_message)

    async def _async_scan(self, timeout: float) -> None:
        self._transition_state(ConnectionState.SCANNING, "Scanning for Muse devices…")
        found: list[BLEDevice] = []
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            name = d.name or ""
            if "Muse" in name or "muse" in name:
                found.append(d)
                logger.info("Found Muse device: %s (%s)", d.name, d.address)
        self.devices = found
        self._transition_state(ConnectionState.IDLE, f"Found {len(found)} Muse device(s)")

    async def _async_connect(self, device: BLEDevice) -> None:
        device_name = getattr(device, "name", "Muse") or "Muse"
        device_address = getattr(device, "address", str(device))
        self._manual_disconnect_requested = False
        self._transition_state(ConnectionState.CONNECTING, f"Connecting to {device_name}…")
        # Na Windows prosimy backend WinRT o nieużywanie cache usług GATT.
        # To ogranicza przypadki, w których pierwszy odczyt zwraca tylko część
        # charakterystyk (bez kanałów EEG), mimo poprawnego połączenia.
        client_kwargs: dict[str, object] = {}
        if os.name == "nt":
            client_kwargs["winrt"] = {"use_cached_services": False}
        try:
            # Używamy adresu zamiast obiektu BLEDevice dla większej stabilności.
            self._client = BleakClient(device_address, **client_kwargs)
        except TypeError:
            # Fallback dla starszych wersji bleak bez argumentu `winrt`.
            self._client = BleakClient(device_address)
        set_callback = getattr(self._client, "set_disconnected_callback", None)
        if callable(set_callback):
            callback_result = set_callback(self._on_client_disconnected)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        self._active_device = ConnectorDevice(
            address=device_address,
            name=device_name,
            rssi=getattr(device, "rssi", None),
        )
        
        try:
            await self._connect_with_retries(device_name)

            self._connected = True
            
            self._device_state.update({
                "device_name": device_name,
                "address": device_address,
                "rssi": getattr(device, "rssi", None),
                "streaming": False,
            })
            self._emit_status(f"Connected to {device_name}")
            self._store.save(device_address, device_name)

            # Windows-specific: Try to pair if not already paired
            if os.name == 'nt':
                try:
                    logger.info("Verifying pairing status for %s...", device_address)
                    await self._client.pair()
                except Exception as e:
                    logger.debug("Pairing info: %s", e)

            # --- Aggressive Service Discovery ---
            # Na Windows usĹ‚ugi BLE potrafiÄ… pojawiÄ‡ siÄ™ z opĂłĹşnieniem po poĹ‚Ä…czeniu.
            # Dlatego odĹ›wieĹĽamy listÄ™ usĹ‚ug przez maksymalnie 30 sekund.
            services_stabilized = False
            discovery_attempts = SERVICE_DISCOVERY_TIMEOUT_SECONDS
            all_chars: list[str] = []
            for attempt in range(1, discovery_attempts + 1):
                logger.info("Service discovery attempt %d/%d...", attempt, discovery_attempts)

                # Wymuszamy odĹ›wieĹĽenie usĹ‚ug, bo sama wĹ‚aĹ›ciwoĹ›Ä‡ .services
                # moĹĽe zwracaÄ‡ niepeĹ‚ny cache zaraz po zestawieniu poĹ‚Ä…czenia.
                all_chars = await self._collect_characteristic_uuids()

                # Sprawdzamy dowolny kanał EEG (nie tylko TP9), bo część backendów
                # BLE raportuje charakterystyki w różnej kolejności.
                if any(uuid in _EEG_UUID_SET for uuid in all_chars):
                    logger.info("EEG characteristics discovered after %d attempts!", attempt)
                    services_stabilized = True
                    break

                logger.warning(
                    "EEG characteristics not found yet. Discovered so far: %d. Waiting...",
                    len(all_chars),
                )
                await asyncio.sleep(1.0)

            if not services_stabilized:
                logger.error("Final check: Discovered UUIDs were: %s", all_chars)
                raise RuntimeError(
                    "EEG SENSORS NOT DETECTED. PLEASE PUT THE DEVICE ON YOUR HEAD, "
                    "ENSURE IT IS POWERED ON, AND PAIRED IN WINDOWS SETTINGS."
                )

            await self._refresh_device_state()

            subscribed_count = await self._apply_stream_config()
            if self._stream_config.get("eeg", True) and subscribed_count == 0:
                raise RuntimeError("Failed to subscribe to any EEG channels.")

            # Send "start streaming" command
            await self._client.write_gatt_char(CONTROL_UUID, CMD_START)
            self._device_state["streaming"] = True
            if self._state != ConnectionState.RECOVERING:
                self._session_metrics.reset()
            if self._session_metrics.connected_since is None:
                self._session_metrics.connected_since = time.monotonic()
            if self._stream_config.get("battery", True):
                self._battery_task = asyncio.create_task(self._battery_poll_loop())
            self._start_watchdog()
            self._transition_state(
                ConnectionState.STREAMING,
                f"Streaming: {subscribed_count} channels",
            )
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            self._connected = False
            self._transition_state(ConnectionState.ERROR, f"Error: {exc}")
            if self._client:
                await self._client.disconnect()
            raise

    async def _connect_with_retries(self, device_name: str) -> None:
        """Podejmuje kilka prób połączenia BLE z kontrolowanym backoffem."""
        if self._client is None:
            raise RuntimeError("BLE client is not initialized.")

        attempts = max(CONNECT_RETRY_ATTEMPTS, 1)
        backoffs = list(CONNECT_RETRY_BACKOFF_SECONDS)
        if len(backoffs) < attempts:
            backoffs.extend([backoffs[-1] if backoffs else 1.0] * (attempts - len(backoffs)))

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                await self._client.connect()
                if self._client.is_connected:
                    if attempt > 1:
                        logger.info(
                            "Connected to %s on retry attempt %d/%d.",
                            device_name,
                            attempt,
                            attempts,
                        )
                    return
                last_error = RuntimeError("Connection established but immediately lost.")
                logger.warning(
                    "Connection attempt %d/%d to %s succeeded but link dropped immediately.",
                    attempt,
                    attempts,
                    device_name,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Connection attempt %d/%d to %s failed: %s",
                    attempt,
                    attempts,
                    device_name,
                    exc,
                )

            if attempt < attempts:
                wait_seconds = backoffs[attempt - 1]
                self._emit_status(
                    f"[{ConnectionState.CONNECTING.value}] "
                    f"Retrying {device_name} connection ({attempt + 1}/{attempts}) in {wait_seconds:.1f}s…"
                )
                await asyncio.sleep(wait_seconds)

        raise RuntimeError(
            f"Unable to establish a stable BLE connection after {attempts} attempts: {last_error}"
        )

    async def _collect_characteristic_uuids(self) -> list[str]:
        """Zwraca listÄ™ UUID charakterystyk po wymuszonym odĹ›wieĹĽeniu usĹ‚ug GATT."""
        if self._client is None:
            return []

        services = None
        try:
            # Najpierw próbujemy oficjalnej metody klienta, bo ona zwykle
            # odświeża cache usług najbardziej niezawodnie.
            refresh_services = getattr(self._client, "get_services", None)
            if callable(refresh_services):
                services = await self._call_services_refresh(refresh_services)
        except Exception as exc:
            logger.debug("Client get_services failed: %s", exc)

        if not services:
            try:
                # Fallback dla backendów bleak, które nie wystawiają API
                # `client.get_services`, ale mają je na obiekcie backend.
                backend = getattr(self._client, "_backend", None)
                backend_refresh = getattr(backend, "get_services", None)
                if callable(backend_refresh):
                    services = await self._call_services_refresh(backend_refresh)
            except Exception as exc:
                logger.debug("Backend get_services failed: %s", exc)

        if not services:
            services = getattr(self._client, "services", None)

        if not services:
            return []

        uuids: list[str] = []
        for service in services:
            characteristics = getattr(service, "characteristics", [])
            for characteristic in characteristics:
                uuid = getattr(characteristic, "uuid", None)
                if uuid:
                    uuids.append(str(uuid).lower())
        return uuids

    async def _call_services_refresh(self, refresh_callable):
        """Wywołuje odświeżanie usług z preferencją dla trybu uncached na Windows."""
        if os.name != "nt":
            return await refresh_callable()

        # Część implementacji WinRT akceptuje `service_cache_mode` i/lub
        # `cache_mode`. Sprawdzamy sygnaturę dynamicznie, by pozostać zgodnym
        # z różnymi wersjami bleak/WinRT.
        kwargs: dict[str, object] = {}
        try:
            signature = inspect.signature(refresh_callable)
            if "service_cache_mode" in signature.parameters:
                kwargs["service_cache_mode"] = "uncached"
            if "cache_mode" in signature.parameters:
                kwargs["cache_mode"] = "uncached"
        except (TypeError, ValueError):
            kwargs = {}

        if kwargs:
            try:
                return await refresh_callable(**kwargs)
            except TypeError:
                logger.debug("Service refresh kwargs not supported; falling back to default call.")
        return await refresh_callable()

    async def _async_auto_connect(self, timeout: float) -> bool:
        """Scan and connect to the first known device found."""
        known = self._store.addresses()
        if not known:
            self._transition_state(ConnectionState.IDLE, "No previously connected devices found.")
            return False

        known_names = ", ".join(
            e["name"] or e["address"] for e in self._store.all()
        )
        self._transition_state(ConnectionState.SCANNING, f"Searching for known device(s): {known_names}…")
        logger.info("Auto-connect: searching for known addresses %s", known)

        all_devices = await BleakScanner.discover(timeout=timeout)
        known_candidates = [d for d in all_devices if d.address.upper() in known]
        known_candidates.sort(key=lambda item: getattr(item, "rssi", -9999), reverse=True)

        for d in known_candidates:
            logger.info(
                "Auto-connect: trying known device %s (%s), RSSI=%s",
                d.name,
                d.address,
                getattr(d, "rssi", "n/a"),
            )
            try:
                await self._async_connect(d)
                return True
            except Exception as exc:
                logger.warning(
                    "Auto-connect: failed for %s (%s): %s",
                    d.name,
                    d.address,
                    exc,
                )
                self._emit_status(
                    f"[{ConnectionState.RECOVERING.value}] Auto-connect fallback: next known device…"
                )

        if known_candidates:
            self._transition_state(ConnectionState.ERROR, "Known device(s) found, but connection failed.")
        else:
            self._transition_state(ConnectionState.IDLE, "Known device(s) not found nearby.")
        return False

    async def _async_disconnect(
        self,
        *,
        manual: bool = True,
        report_state: bool = True,
        log_session: bool = True,
    ) -> None:
        self._manual_disconnect_requested = manual
        if self._battery_task:
            self._battery_task.cancel()
            self._battery_task = None
        self._cancel_watchdog()
        if self._client and self._client.is_connected:
            try:
                await self._client.write_gatt_char(CONTROL_UUID, CMD_STOP)
            except Exception:
                pass
            await self._client.disconnect()
        self._active_notifications.clear()
        self._connected = False
        self._device_state["streaming"] = False
        for key in self._device_state.get("stream_activity", {}):
            self._device_state["stream_activity"][key] = False
        if log_session:
            self._log_session_metrics("disconnect")
        if report_state:
            self._transition_state(ConnectionState.IDLE, "Disconnected")

    def _make_eeg_handler(self, channel: str) -> Callable:
        """Return a BLE notification callback bound to *channel*."""

        def handler(sender, data: bytearray) -> None:  # noqa: ANN001
            # Tylko aktywny strumieĹ„ EEG moĹĽe aktualizowaÄ‡ prĂłbki.
            if not self._device_state.get("streaming") or not self._stream_config.get("eeg", True):
                return

            try:
                sequence, samples = _parse_eeg_packet(bytes(data))
                self._update_session_metrics(channel, sequence, len(samples))
                self._on_eeg(channel, samples)
                self._emit_telemetry()
                
                if self._debug_logging_enabled:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self._eeg_debug_logger.info("[%s] %s: %s", timestamp, channel, samples)
            except Exception as exc:
                logger.warning("EEG parse error on %s: %s", channel, exc)

        return handler

    def _make_imu_handler(self, sensor: str, scale: float) -> Callable:
        """Buduje callback BLE dla pakietĂłw IMU."""

        def handler(sender, data: bytearray) -> None:  # noqa: ANN001
            if not self._device_state.get("streaming") or not self._stream_config.get(sensor, True):
                return
            try:
                sequence, samples = _parse_imu_packet(bytes(data), scale=scale)
                self._on_imu(
                    IMUFrame(
                        sensor=sensor,
                        sequence=sequence,
                        samples=samples,
                        timestamp=time.time(),
                    )
                )
                self._device_state["stream_activity"][sensor] = True
                self._emit_telemetry()
            except Exception as exc:
                logger.warning("IMU parse error on %s: %s", sensor, exc)

        return handler

    def _make_ppg_handler(self, channel: str) -> Callable:
        """Buduje callback BLE dla pakietĂłw PPG."""

        def handler(sender, data: bytearray) -> None:  # noqa: ANN001
            if not self._device_state.get("streaming") or not self._stream_config.get("ppg", True):
                return
            try:
                sequence, samples = _parse_ppg_packet(bytes(data))
                self._on_ppg(
                    PPGFrame(
                        channel=channel,
                        sequence=sequence,
                        samples=samples,
                        timestamp=time.time(),
                    )
                )
                self._device_state["stream_activity"]["ppg"] = True
                self._emit_telemetry()
            except Exception as exc:
                logger.warning("PPG parse error on %s: %s", channel, exc)

        return handler

    async def _apply_stream_config(self) -> int:
        """WĹ‚Ä…cza/wyĹ‚Ä…cza notyfikacje BLE zgodnie z aktualnÄ… konfiguracjÄ…."""
        if self._client is None or not self._client.is_connected:
            return 0
        eeg_subscribed = 0
        for channel, uuid in EEG_UUIDS.items():
            active = self._stream_config.get("eeg", True)
            eeg_subscribed += await self._toggle_notify(uuid, self._make_eeg_handler(channel), active)
        await self._toggle_notify(
            ACCEL_UUID,
            self._make_imu_handler("accelerometer", scale=0.000061),
            self._stream_config.get("accelerometer", True),
        )
        await self._toggle_notify(
            GYRO_UUID,
            self._make_imu_handler("gyroscope", scale=0.0074768),
            self._stream_config.get("gyroscope", True),
        )
        for channel, uuid in PPG_UUIDS.items():
            await self._toggle_notify(
                uuid,
                self._make_ppg_handler(channel),
                self._stream_config.get("ppg", True),
            )
        if not self._stream_config.get("battery", True):
            if self._battery_task:
                self._battery_task.cancel()
                self._battery_task = None
            self._device_state["battery_level"] = None
        elif self._battery_task is None and self._connected:
            self._battery_task = asyncio.create_task(self._battery_poll_loop())
        self._device_state["stream_activity"]["eeg"] = eeg_subscribed > 0
        self._device_state["stream_activity"]["accelerometer"] = self._stream_config.get("accelerometer", True)
        self._device_state["stream_activity"]["gyroscope"] = self._stream_config.get("gyroscope", True)
        self._device_state["stream_activity"]["ppg"] = self._stream_config.get("ppg", True)
        self._device_state["stream_activity"]["battery"] = self._stream_config.get("battery", True)
        self._emit_telemetry()
        return eeg_subscribed

    async def _toggle_notify(self, uuid: str, handler: Callable, should_enable: bool) -> int:
        """Pomocniczo start/stop notyfikacji dla pojedynczej charakterystyki."""
        if self._client is None:
            return 0
        char = self._client.services.get_characteristic(uuid)
        if not char:
            return 0
        try:
            if should_enable and uuid not in self._active_notifications:
                await self._client.start_notify(uuid, handler)
                self._active_notifications.add(uuid)
                return 1
            if not should_enable and uuid in self._active_notifications:
                await self._client.stop_notify(uuid)
                self._active_notifications.discard(uuid)
        except Exception as exc:
            logger.warning("Failed to toggle notify %s: %s", uuid, exc)
        return 0

    def _emit_telemetry(self) -> None:
        """Publikuje ujednoliconÄ… telemetriÄ™ urzÄ…dzenia do warstw wyĹĽej."""
        telemetry = DeviceTelemetry(
            battery_level=self._device_state.get("battery_level"),
            stream_activity=dict(self._device_state.get("stream_activity", {})),
            signal_quality=dict(self._device_state.get("signal_quality", {})),
            motion_artifact=bool(self._device_state.get("motion_artifact", False)),
            timestamp=time.time(),
        )
        self._on_telemetry(telemetry)

    async def _refresh_device_state(self) -> None:
        """Collect additional metadata (battery + available sensors)."""
        if self._client is None:
            return
            
        services = self._client.services
        if not services:
            logger.warning("No GATT services discovered for device.")
            return

        characteristic_uuids = {
            ch.uuid.lower()
            for service in services
            for ch in service.characteristics
        }
        
        # Log discovered characteristics for debugging if needed
        logger.debug("Discovered %d characteristics", len(characteristic_uuids))

        sensors = [
            name
            for name, uuids in SENSOR_CHARACTERISTICS.items()
            if characteristic_uuids.intersection(uuids)
        ]
        self._device_state["available_sensors"] = sensors
        self._device_state["battery_level"] = await self._read_battery_level()
        self._device_state["stream_activity"].update({
            "eeg": self._stream_config.get("eeg", True) and "EEG" in sensors,
            "accelerometer": self._stream_config.get("accelerometer", True) and "Accelerometer" in sensors,
            "gyroscope": self._stream_config.get("gyroscope", True) and "Gyroscope" in sensors,
            "ppg": self._stream_config.get("ppg", True) and "PPG" in sensors,
            "battery": self._stream_config.get("battery", True),
        })
        self._emit_telemetry()

    async def _read_battery_level(self) -> Optional[int]:
        if self._client is None or not self._client.is_connected:
            return None
        try:
            char = self._client.services.get_characteristic(BATTERY_LEVEL_UUID)
            if not char:
                logger.debug("Battery characteristic not found")
                return None
            data = await self._client.read_gatt_char(char)
            if not data:
                return None
            level = _parse_battery_payload(data)
            self._device_state["stream_activity"]["battery"] = True
            self._emit_telemetry()
            return level
        except Exception as exc:
            logger.debug("Battery level unavailable: %s", exc)
            return None

    async def _battery_poll_loop(self) -> None:
        while self._client and self._client.is_connected:
            if self._stream_config.get("battery", True):
                self._device_state["battery_level"] = await self._read_battery_level()
            await asyncio.sleep(20)

    def _start_watchdog(self) -> None:
        """Uruchamia watchdog, ktĂłry pilnuje czy napĹ‚ywajÄ… prĂłbki EEG."""
        self._cancel_watchdog()
        self._watchdog_task = asyncio.create_task(self._stream_watchdog_loop())

    def _cancel_watchdog(self) -> None:
        """Zatrzymuje watchdog streamu EEG, jeĹĽeli byĹ‚ aktywny."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            self._watchdog_task = None

    async def _stream_watchdog_loop(self) -> None:
        """Wykrywa timeout prĂłbek i uruchamia automatyczne odzyskiwanie poĹ‚Ä…czenia."""
        while self._connected and not self._manual_disconnect_requested:
            await asyncio.sleep(0.5)
            if self._state != ConnectionState.STREAMING:
                continue
            last_sample = self._session_metrics.last_sample_monotonic
            if last_sample is None:
                continue
            if time.monotonic() - last_sample <= self._watchdog_timeout_seconds:
                continue
            logger.warning(
                "EEG watchdog timeout: no samples for %.2fs.",
                self._watchdog_timeout_seconds,
            )
            if not self._recovering_task or self._recovering_task.done():
                self._recovering_task = asyncio.create_task(
                    self._recover_stream("No EEG samples received within timeout"),
                )
            return

    def _on_client_disconnected(self, _client: BleakClient) -> None:
        """Callback bleak-a uruchamiany przy niespodziewanej utracie poĹ‚Ä…czenia."""
        if self._manual_disconnect_requested or self._loop is None:
            return
        if self._state not in {ConnectionState.STREAMING, ConnectionState.CONNECTING}:
            return
        logger.warning("Unexpected BLE disconnect detected.")
        if not self._recovering_task or self._recovering_task.done():
            self._recovering_task = asyncio.run_coroutine_threadsafe(
                self._recover_stream("Bluetooth link lost"),
                self._loop,
            )

    async def _recover_stream(self, reason: str) -> None:
        """PrĂłbuje odzyskaÄ‡ streaming uĹĽywajÄ…c polityki backoff."""
        if self._active_device is None:
            self._transition_state(ConnectionState.ERROR, "Recovery failed: no active device.")
            return

        self._transition_state(ConnectionState.RECOVERING, reason)

        for attempt, backoff in enumerate(self._reconnect_backoff_seconds, start=1):
            self._session_metrics.reconnect_count += 1
            self._device_state["reconnect_attempts"] = attempt
            self._emit_status(
                f"[{ConnectionState.RECOVERING.value}] Reconnect attempt {attempt}/{len(self._reconnect_backoff_seconds)} in {backoff:.0f}s",
            )
            await asyncio.sleep(backoff)
            try:
                await self._async_disconnect(manual=False, report_state=False, log_session=False)
                await self._async_connect(self._active_device)
                self._device_state["reconnect_attempts"] = attempt
                self._emit_status(
                    f"[{ConnectionState.STREAMING.value}] Reconnect succeeded on attempt {attempt}",
                )
                return
            except Exception as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt, exc)

        self._transition_state(
            ConnectionState.ERROR,
            f"Recovery failed after {len(self._reconnect_backoff_seconds)} attempts.",
        )

    def _update_session_metrics(self, channel: str, sequence: int, sample_count: int) -> None:
        """Aktualizuje metryki sesji po odebraniu pakietu EEG."""
        now = time.monotonic()
        if self._session_metrics.last_sample_monotonic is not None:
            interval = now - self._session_metrics.last_sample_monotonic
            self._session_metrics.sample_interval_sum += interval
            self._session_metrics.sample_interval_count += 1
        self._session_metrics.last_sample_monotonic = now
        self._session_metrics.total_samples += sample_count

        prev_sequence = self._session_metrics.sequence_by_channel.get(channel)
        if prev_sequence is not None:
            sequence_gap = (sequence - prev_sequence) % 65536
            if sequence_gap > 1:
                missing_packets = sequence_gap - 1
                self._session_metrics.dropout_samples += missing_packets * sample_count
        self._session_metrics.sequence_by_channel[channel] = sequence

    def _log_session_metrics(self, reason: str) -> None:
        """Loguje podsumowanie sesji (czas, reconnect, Ĺ›redni interwaĹ‚ i dropout)."""
        if self._session_metrics.logged:
            return
        if self._session_metrics.connected_since is None:
            return
        duration = max(0.0, time.monotonic() - self._session_metrics.connected_since)
        logger.info(
            (
                "EEG session metrics (%s): duration=%.2fs, reconnects=%d, "
                "avg_sample_interval=%.4fs, dropouts=%.2f%%"
            ),
            reason,
            duration,
            self._session_metrics.reconnect_count,
            self._session_metrics.average_sample_interval,
            self._session_metrics.dropout_percent,
        )
        self._session_metrics.logged = True
