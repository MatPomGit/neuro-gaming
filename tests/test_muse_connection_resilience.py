"""Testy odporności połączenia MuseConnector."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.muse_connector import ConnectionState, MuseConnector


class _FakeChar:
    """Minimalna reprezentacja charakterystyki BLE."""

    def __init__(self, uuid: str) -> None:
        self.uuid = uuid


class _FakeService:
    """Minimalna reprezentacja usługi BLE."""

    def __init__(self, characteristics: list[_FakeChar]) -> None:
        self.characteristics = characteristics


class _FakeServices(list):
    """Kontener usług z metodą zgodną z bleak."""

    def get_characteristic(self, target_uuid: str):
        target = target_uuid.lower()
        for service in self:
            for char in service.characteristics:
                if char.uuid.lower() == target:
                    return char
        return None


class _FakeBleakClient:
    """Mock klienta BLE do testów bez fizycznego urządzenia."""

    def __init__(self, _address: str) -> None:
        self.is_connected = False
        self.services = _FakeServices([_FakeService([_FakeChar("00002a19-0000-1000-8000-00805f9b34fb")])])
        self._disconnect_callback = None

    async def connect(self) -> None:
        self.is_connected = True

    async def disconnect(self) -> None:
        self.is_connected = False

    async def get_services(self):
        return self.services

    async def start_notify(self, _uuid: str, _handler):
        return None

    async def write_gatt_char(self, _uuid: str, _payload: bytes):
        return None

    async def read_gatt_char(self, _char) -> bytes:
        return bytes([95])

    async def pair(self):
        return None

    def set_disconnected_callback(self, callback):
        self._disconnect_callback = callback


def test_watchdog_timeout_triggers_recovering_state(monkeypatch):
    """Brak próbek EEG powinien uruchomić ścieżkę RECOVERING."""
    connector = MuseConnector(on_eeg=lambda _ch, _samples: None)
    connector._connected = True
    connector._watchdog_timeout_seconds = 0.1
    connector._transition_state(ConnectionState.STREAMING, "Test streaming")
    connector._session_metrics.last_sample_monotonic = time.monotonic() - 2.0

    recover_mock = AsyncMock()
    monkeypatch.setattr(connector, "_recover_stream", recover_mock)

    async def _run_watchdog() -> None:
        watchdog_task = asyncio.create_task(connector._stream_watchdog_loop())
        await asyncio.sleep(0.7)
        await asyncio.wait_for(watchdog_task, timeout=1.0)

    asyncio.run(_run_watchdog())

    recover_mock.assert_awaited_once()
    assert connector._state == ConnectionState.STREAMING


def test_connect_enters_error_state_when_eeg_characteristics_are_missing(monkeypatch):
    """Brak kanałów EEG powinien zakończyć próbę połączenia stanem ERROR."""
    connector = MuseConnector(on_eeg=lambda _ch, _samples: None)
    status_messages: list[str] = []
    connector.set_status_callback(status_messages.append)

    monkeypatch.setattr("src.muse_connector.BleakClient", _FakeBleakClient)
    monkeypatch.setattr("src.muse_connector.SERVICE_DISCOVERY_TIMEOUT_SECONDS", 1)

    fake_device = SimpleNamespace(name="Muse-Test", address="AA:BB:CC:DD:EE:FF", rssi=-55)

    async def _run_connect() -> None:
        try:
            await connector._async_connect(fake_device)
        except RuntimeError:
            return
        raise AssertionError("Expected RuntimeError for missing EEG characteristics.")

    asyncio.run(_run_connect())

    assert connector._state == ConnectionState.ERROR
    assert any("[ERROR]" in msg for msg in status_messages)


def test_recover_stream_reconnects_with_backoff(monkeypatch):
    """Reconnect powinien ponowić próbę i wrócić do STREAMING po sukcesie."""
    connector = MuseConnector(on_eeg=lambda _ch, _samples: None)
    status_messages: list[str] = []
    connector.set_status_callback(status_messages.append)
    connector._active_device = SimpleNamespace(name="Muse-Test", address="AA", rssi=-40)
    connector._reconnect_backoff_seconds = (0.0, 0.0, 0.0)

    disconnect_mock = AsyncMock()
    monkeypatch.setattr(connector, "_async_disconnect", disconnect_mock)

    attempts = {"count": 0}

    async def _connect_with_one_failure(_device):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        connector._transition_state(ConnectionState.STREAMING, "Streaming restored")

    monkeypatch.setattr(connector, "_async_connect", _connect_with_one_failure)

    asyncio.run(connector._recover_stream("Watchdog timeout"))

    assert attempts["count"] == 2
    assert connector._state == ConnectionState.STREAMING
    assert connector._session_metrics.reconnect_count == 2
    assert any("Reconnect succeeded" in msg for msg in status_messages)
