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
import struct
import threading
from collections.abc import Callable
from typing import Optional

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

logger = logging.getLogger(__name__)

# Default path for the known-devices store
_DEFAULT_STORE_PATH = os.path.join(os.path.expanduser("~"), ".neuro_gaming_devices.json")

# ──────────────────────────────────────────────────────────────────────────────
# Muse BLE UUIDs
# ──────────────────────────────────────────────────────────────────────────────

MUSE_SERVICE_UUID = "0000fe8d-0000-1000-8000-00805f9b34fb"

# Control characteristic – used to start / stop EEG streaming
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

# ──────────────────────────────────────────────────────────────────────────────
# Known-devices persistent store
# ──────────────────────────────────────────────────────────────────────────────

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

    # ── public interface ───────────────────────────────────────────────────

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

    # ── private helpers ────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Packet parser
# ──────────────────────────────────────────────────────────────────────────────

def _parse_eeg_packet(data: bytes) -> tuple[int, np.ndarray]:
    """Decode one EEG BLE notification.

    The Muse S encodes five 12-bit samples per notification packet:
      bytes 0-1  – big-endian 16-bit sequence number
      bytes 2-9  – five 12-bit sample values, packed big-endian

    Raw sample values are converted to µV using:
        voltage = (raw - 2048) × 0.48828125

    Parameters
    ----------
    data:
        Raw bytes received from the BLE notification.

    Returns
    -------
    sequence:
        Packet sequence number.
    samples:
        NumPy array of 5 voltage samples in µV.
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


# ──────────────────────────────────────────────────────────────────────────────
# Connector class
# ──────────────────────────────────────────────────────────────────────────────

class MuseConnector:
    """Manages Bluetooth connection to a Muse S Athena device.

    Previously connected devices are automatically remembered and can be
    reconnected via :meth:`auto_connect`.

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
        on_status: Optional[Callable[[str], None]] = None,
        known_devices_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        on_eeg:
            Called with ``(channel_name, samples)`` whenever new EEG data
            arrives.  *channel_name* is one of ``"TP9"``, ``"AF7"``,
            ``"AF8"``, ``"TP10"``.  *samples* is a float32 array of 5 µV
            values.  This callback is invoked from the background thread.
        on_status:
            Optional callback for human-readable status messages.
        known_devices_path:
            Optional path to the JSON file used to persist previously
            connected devices.  Defaults to ``~/.neuro_gaming_devices.json``.
        """
        self._on_eeg = on_eeg
        self._on_status = on_status or (lambda _: None)
        self._store = KnownDevicesStore(known_devices_path or _DEFAULT_STORE_PATH)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[BleakClient] = None
        self._connected = False
        self._battery_task: Optional[asyncio.Task] = None
        self._device_state = {
            "device_name": "Unknown",
            "address": "",
            "rssi": None,
            "battery_level": None,
            "sample_rate_hz": SAMPLE_RATE,
            "available_sensors": [],
            "streaming": False,
        }

        self.devices: list[BLEDevice] = []

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background asyncio thread."""
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

    # ── public API (thread-safe) ───────────────────────────────────────────

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
        future.result(timeout=15)

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

    # ── async implementation ───────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _async_scan(self, timeout: float) -> None:
        self._on_status("Scanning for Muse devices…")
        found: list[BLEDevice] = []
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            name = d.name or ""
            if "Muse" in name or "muse" in name:
                found.append(d)
                logger.info("Found Muse device: %s (%s)", d.name, d.address)
        self.devices = found
        self._on_status(f"Found {len(found)} Muse device(s)")

    async def _async_connect(self, device: BLEDevice) -> None:
        self._on_status(f"Connecting to {device.name}…")
        # Use address instead of BLEDevice object for better stability on Windows
        self._client = BleakClient(device.address)
        
        try:
            await self._client.connect()
            self._connected = True
            device_name = device.name or "Muse device"
            self._device_state.update({
                "device_name": device_name,
                "address": device.address,
                "rssi": getattr(device, "rssi", None),
                "streaming": False,
            })
            self._on_status(f"Connected to {device.name}")

            # Remember this device for future auto-connect
            self._store.save(device.address, device_name)
            logger.info("Saved known device: %s (%s)", device_name, device.address)

            await self._refresh_device_state()

            # Subscribe to EEG notification characteristics
            for channel, uuid in EEG_UUIDS.items():
                await self._client.start_notify(
                    uuid,
                    self._make_eeg_handler(channel),
                )

            # Send "start streaming" command
            await self._client.write_gatt_char(CONTROL_UUID, CMD_START)
            self._device_state["streaming"] = True
            self._battery_task = asyncio.create_task(self._battery_poll_loop())
            self._on_status("EEG streaming started")
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            self._connected = False
            self._on_status(f"Error: {exc}")
            if self._client:
                await self._client.disconnect()
            raise

    async def _async_auto_connect(self, timeout: float) -> bool:
        """Scan and connect to the first known device found."""
        known = self._store.addresses()
        if not known:
            self._on_status("No previously connected devices found.")
            return False

        known_names = ", ".join(
            e["name"] or e["address"] for e in self._store.all()
        )
        self._on_status(f"Searching for known device(s): {known_names}…")
        logger.info("Auto-connect: searching for known addresses %s", known)

        all_devices = await BleakScanner.discover(timeout=timeout)
        for d in all_devices:
            if d.address.upper() in known:
                logger.info("Auto-connect: found known device %s (%s)", d.name, d.address)
                await self._async_connect(d)
                return True

        self._on_status("Known device(s) not found nearby.")
        return False

    async def _async_disconnect(self) -> None:
        if self._battery_task:
            self._battery_task.cancel()
            self._battery_task = None
        if self._client and self._client.is_connected:
            try:
                await self._client.write_gatt_char(CONTROL_UUID, CMD_STOP)
            except Exception:
                pass
            await self._client.disconnect()
        self._connected = False
        self._device_state["streaming"] = False
        self._on_status("Disconnected")

    def _make_eeg_handler(self, channel: str) -> Callable:
        """Return a BLE notification callback bound to *channel*."""

        def handler(sender, data: bytearray) -> None:  # noqa: ANN001
            try:
                _, samples = _parse_eeg_packet(bytes(data))
                self._on_eeg(channel, samples)
            except Exception as exc:
                logger.warning("EEG parse error on %s: %s", channel, exc)

        return handler

    async def _refresh_device_state(self) -> None:
        """Collect additional metadata (battery + available sensors)."""
        if self._client is None:
            return
        try:
            services = await self._client.get_services()
        except Exception as exc:
            logger.debug("Failed to query GATT services: %s", exc)
            services = self._client.services

        characteristic_uuids = {
            ch.uuid.lower()
            for service in services
            for ch in service.characteristics
        }
        sensors = [
            name
            for name, uuids in SENSOR_CHARACTERISTICS.items()
            if characteristic_uuids.intersection(uuids)
        ]
        self._device_state["available_sensors"] = sensors
        self._device_state["battery_level"] = await self._read_battery_level()

    async def _read_battery_level(self) -> Optional[int]:
        if self._client is None or not self._client.is_connected:
            return None
        try:
            data = await self._client.read_gatt_char(BATTERY_LEVEL_UUID)
            if not data:
                return None
            return int(data[0])
        except Exception as exc:
            logger.debug("Battery level unavailable: %s", exc)
            return None

    async def _battery_poll_loop(self) -> None:
        while self._client and self._client.is_connected:
            self._device_state["battery_level"] = await self._read_battery_level()
            await asyncio.sleep(20)
