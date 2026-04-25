"""
Tests for KnownDevicesStore and MuseConnector auto-connect / device memory.
"""

import asyncio
import json
import os
import threading
import unittest.mock as mock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.muse_connector import KnownDevicesStore, MuseConnector


# ──────────────────────────────────────────────────────────────────────────────
# KnownDevicesStore
# ──────────────────────────────────────────────────────────────────────────────

class TestKnownDevicesStore:
    def test_empty_store_when_file_missing(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        assert store.all() == []
        assert store.addresses() == set()

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse-1234")
        assert os.path.exists(path)

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse-1234")

        store2 = KnownDevicesStore(path)
        entries = store2.all()
        assert len(entries) == 1
        assert entries[0]["address"] == "AA:BB:CC:DD:EE:FF"
        assert entries[0]["name"] == "Muse-1234"

    def test_addresses_returns_upper_case(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        store.save("aa:bb:cc:dd:ee:ff", "Muse")
        assert "AA:BB:CC:DD:EE:FF" in store.addresses()

    def test_save_updates_existing_name(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        store.save("AA:BB:CC:DD:EE:FF", "OldName")
        store.save("AA:BB:CC:DD:EE:FF", "NewName")
        entries = store.all()
        assert len(entries) == 1
        assert entries[0]["name"] == "NewName"

    def test_save_multiple_devices(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        store.save("AA:BB:CC:DD:EE:01", "Muse-A")
        store.save("AA:BB:CC:DD:EE:02", "Muse-B")
        assert len(store.all()) == 2
        assert len(store.addresses()) == 2

    def test_remove_existing_device(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        store.save("AA:BB:CC:DD:EE:FF", "Muse")
        result = store.remove("AA:BB:CC:DD:EE:FF")
        assert result is True
        assert store.all() == []

    def test_remove_nonexistent_device_returns_false(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        result = store.remove("00:00:00:00:00:00")
        assert result is False

    def test_remove_persists_to_file(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse")
        store.remove("AA:BB:CC:DD:EE:FF")
        store2 = KnownDevicesStore(path)
        assert store2.all() == []

    def test_clear_empties_store(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        store.save("AA:BB:CC:DD:EE:FF", "Muse")
        store.clear()
        assert store.all() == []

    def test_clear_persists_to_file(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse")
        store.clear()
        store2 = KnownDevicesStore(path)
        assert store2.all() == []

    def test_malformed_json_returns_empty(self, tmp_path):
        path = str(tmp_path / "devices.json")
        with open(path, "w") as fh:
            fh.write("this is not json")
        store = KnownDevicesStore(path)
        assert store.all() == []

    def test_json_without_address_field_ignored(self, tmp_path):
        path = str(tmp_path / "devices.json")
        with open(path, "w") as fh:
            json.dump([{"name": "Muse"}], fh)
        store = KnownDevicesStore(path)
        assert store.all() == []

    def test_path_property(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        assert store.path == path

    def test_remove_is_case_insensitive(self, tmp_path):
        store = KnownDevicesStore(str(tmp_path / "devices.json"))
        store.save("AA:BB:CC:DD:EE:FF", "Muse")
        result = store.remove("aa:bb:cc:dd:ee:ff")
        assert result is True
        assert store.all() == []


# ──────────────────────────────────────────────────────────────────────────────
# MuseConnector – known_devices property
# ──────────────────────────────────────────────────────────────────────────────

class TestMuseConnectorKnownDevices:
    def _make_connector(self, tmp_path):
        return MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )

    def test_known_devices_empty_initially(self, tmp_path):
        connector = self._make_connector(tmp_path)
        assert connector.known_devices == []

    def test_known_devices_reflects_store(self, tmp_path):
        path = str(tmp_path / "devices.json")
        # Pre-populate the store
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse-Test")

        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=path,
        )
        assert len(connector.known_devices) == 1
        assert connector.known_devices[0]["address"] == "AA:BB:CC:DD:EE:FF"

    def test_known_devices_returns_copy(self, tmp_path):
        connector = self._make_connector(tmp_path)
        copy1 = connector.known_devices
        copy1.append({"address": "fake", "name": "fake"})
        assert connector.known_devices == []


# ──────────────────────────────────────────────────────────────────────────────
# MuseConnector – auto_connect raises before start()
# ──────────────────────────────────────────────────────────────────────────────

class TestAutoConnectBeforeStart:
    def test_auto_connect_raises_if_not_started(self, tmp_path):
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )
        with pytest.raises(RuntimeError, match="start()"):
            connector.auto_connect()


# ──────────────────────────────────────────────────────────────────────────────
# MuseConnector – _async_auto_connect (tested in isolation)
# ──────────────────────────────────────────────────────────────────────────────

class TestAsyncAutoConnect:
    """Exercise the async auto-connect logic without real BLE hardware."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_false_when_no_known_devices(self, tmp_path):
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )
        statuses = []
        connector.set_status_callback(statuses.append)

        with patch("src.muse_connector.BleakScanner.discover", new=AsyncMock(return_value=[])):
            result = self._run(connector._async_auto_connect(timeout=1.0))

        assert result is False
        assert any("No previously" in s for s in statuses)

    def test_returns_false_when_known_device_not_nearby(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse-Test")

        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=path,
        )
        statuses = []
        connector.set_status_callback(statuses.append)

        with patch("src.muse_connector.BleakScanner.discover", new=AsyncMock(return_value=[])):
            result = self._run(connector._async_auto_connect(timeout=1.0))

        assert result is False
        assert any("not found" in s for s in statuses)

    def test_set_status_callback_after_start_still_receives_statuses(self, tmp_path):
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )
        statuses = []

        connector.start()
        try:
            connector.set_status_callback(statuses.append)
            result = connector.auto_connect(timeout=1.0)
        finally:
            connector.stop()

        assert result is False
        assert any("No previously connected devices found." in s for s in statuses)

    def test_connects_to_known_device_when_found(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:FF", "Muse-Test")

        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=path,
        )

        # Create a fake BLEDevice with the known address
        fake_device = MagicMock()
        fake_device.address = "AA:BB:CC:DD:EE:FF"
        fake_device.name = "Muse-Test"

        async def mock_successful_connect(device):
            pass  # succeed immediately

        with patch("src.muse_connector.BleakScanner.discover", new=AsyncMock(return_value=[fake_device])):
            with patch.object(connector, "_async_connect", side_effect=mock_successful_connect) as mock_conn:
                result = self._run(connector._async_auto_connect(timeout=1.0))

        assert result is True
        mock_conn.assert_called_once_with(fake_device)


class TestScanMuseDetection:
    """Testy detekcji Muse podczas skanowania BLE."""

    @staticmethod
    def _run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_is_muse_candidate_matches_advertisement_local_name(self, tmp_path):
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )
        fake_device = SimpleNamespace(name="", address="AA:BB:CC:DD:EE:FF")
        fake_adv = SimpleNamespace(local_name="Muse-S-9ABC", service_uuids=[])

        is_muse, reason = connector._is_muse_candidate(fake_device, fake_adv, known_addresses=set())

        assert is_muse is True
        assert reason == "advertisement local_name"

    def test_is_muse_candidate_matches_muse_service_uuid(self, tmp_path):
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )
        fake_device = SimpleNamespace(name="", address="AA:BB:CC:DD:EE:10")
        fake_adv = SimpleNamespace(
            local_name="",
            service_uuids=["0000fe8d-0000-1000-8000-00805f9b34fb"],
        )

        is_muse, reason = connector._is_muse_candidate(fake_device, fake_adv, known_addresses=set())

        assert is_muse is True
        assert reason == "Muse service UUID"

    def test_async_scan_uses_return_adv_dictionary_results(self, tmp_path):
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=str(tmp_path / "devices.json"),
        )
        fake_muse = SimpleNamespace(name="", address="AA:BB:CC:DD:EE:01")
        fake_tuya = SimpleNamespace(name="TUYA_Device", address="AA:BB:CC:DD:EE:02")
        fake_muse_adv = SimpleNamespace(local_name="Muse-2A1B", service_uuids=[])
        fake_tuya_adv = SimpleNamespace(local_name="TUYA_X", service_uuids=[])

        discover_result = {
            fake_muse.address: (fake_muse, fake_muse_adv),
            fake_tuya.address: (fake_tuya, fake_tuya_adv),
        }

        with patch("src.muse_connector.BleakScanner.discover", new=AsyncMock(return_value=discover_result)):
            self._run(connector._async_scan(timeout=1.0))

        assert len(connector.devices) == 1
        assert connector.devices[0].address == fake_muse.address

    def test_address_matching_is_case_insensitive(self, tmp_path):
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("aa:bb:cc:dd:ee:ff", "Muse-Test")

        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=path,
        )

        fake_device = MagicMock()
        fake_device.address = "AA:BB:CC:DD:EE:FF"
        fake_device.name = "Muse-Test"

        async def mock_successful_connect(device):
            pass

        with patch("src.muse_connector.BleakScanner.discover", new=AsyncMock(return_value=[fake_device])):
            with patch.object(connector, "_async_connect", side_effect=mock_successful_connect):
                result = self._run(connector._async_auto_connect(timeout=1.0))

        assert result is True

    def test_auto_connect_falls_back_to_next_known_device_on_failure(self, tmp_path):
        """Auto-connect powinien próbować kolejnych znanych urządzeń po błędzie."""
        path = str(tmp_path / "devices.json")
        store = KnownDevicesStore(path)
        store.save("AA:BB:CC:DD:EE:01", "Muse-1")
        store.save("AA:BB:CC:DD:EE:02", "Muse-2")

        connector = MuseConnector(on_eeg=lambda ch, s: None, known_devices_path=path)

        first = MagicMock()
        first.address = "AA:BB:CC:DD:EE:01"
        first.name = "Muse-1"
        first.rssi = -30
        second = MagicMock()
        second.address = "AA:BB:CC:DD:EE:02"
        second.name = "Muse-2"
        second.rssi = -65

        calls: list[str] = []

        async def _connect_with_single_failure(device):
            calls.append(device.address)
            if device.address.endswith("01"):
                raise RuntimeError("temporary failure")

        with patch("src.muse_connector.BleakScanner.discover", new=AsyncMock(return_value=[second, first])):
            with patch.object(connector, "_async_connect", side_effect=_connect_with_single_failure):
                result = self._run(connector._async_auto_connect(timeout=1.0))

        assert result is True
        assert calls == ["AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02"]


# ──────────────────────────────────────────────────────────────────────────────
# MuseConnector – device is saved after successful _async_connect
# ──────────────────────────────────────────────────────────────────────────────

class TestDeviceSavedOnConnect:
    def test_device_saved_after_async_connect(self, tmp_path):
        path = str(tmp_path / "devices.json")
        connector = MuseConnector(
            on_eeg=lambda ch, s: None,
            known_devices_path=path,
        )

        fake_device = MagicMock()
        fake_device.address = "AA:BB:CC:DD:EE:FF"
        fake_device.name = "Muse-Test"

        # Mock the BleakClient so no real BLE connection is attempted
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.connect = AsyncMock()
        mock_client.read_gatt_char = AsyncMock(return_value=b"\x50")
        mock_client.start_notify = AsyncMock()
        mock_client.write_gatt_char = AsyncMock()
        mock_char = MagicMock()
        mock_char.uuid = "273e0003-4c4d-454d-96be-f03bac821358"
        mock_service = MagicMock()
        mock_service.characteristics = [mock_char]
        mock_services = MagicMock()
        mock_services.__iter__.return_value = iter([mock_service])
        mock_services.get_characteristic.return_value = object()
        mock_client.get_services = AsyncMock(return_value=mock_services)
        mock_client.services = mock_services

        with patch("src.muse_connector.BleakClient", return_value=mock_client):
            asyncio.get_event_loop().run_until_complete(
                connector._async_connect(fake_device)
            )

        store = KnownDevicesStore(path)
        assert "AA:BB:CC:DD:EE:FF" in store.addresses()
        assert store.all()[0]["name"] == "Muse-Test"
