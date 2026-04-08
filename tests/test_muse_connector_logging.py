from pathlib import Path
from unittest.mock import MagicMock, patch

from src.muse_connector import (
    MuseConnector,
    configure_eeg_debug_logger,
    get_eeg_debug_log_path,
)


def test_get_eeg_debug_log_path_linux_xdg(monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/state-home")

    path = get_eeg_debug_log_path(os_name="posix", platform="linux")

    assert path == Path("/tmp/state-home/neuro_gaming/logs/eeg_debug.log")


def test_get_eeg_debug_log_path_windows(monkeypatch):
    monkeypatch.setenv("APPDATA", r"C:\\Users\\demo\\AppData\\Roaming")

    path = get_eeg_debug_log_path(os_name="nt", platform="win32")

    assert str(path).endswith("neuro_gaming/logs/eeg_debug.log")


def test_configure_eeg_debug_logger_disabled_does_not_create_handler():
    logger, path = configure_eeg_debug_logger(enabled=False)

    assert path is None
    assert logger.disabled is True
    assert logger.handlers == []


def test_configure_eeg_debug_logger_enabled_uses_rotating_handler():
    fake_handler = MagicMock()
    fake_path = Path("/tmp/app/logs/eeg_debug.log")

    with patch("src.muse_connector.Path.mkdir") as mkdir_mock:
        with patch("src.muse_connector.RotatingFileHandler", return_value=fake_handler) as handler_cls:
            logger, path = configure_eeg_debug_logger(enabled=True, log_path=fake_path, max_bytes=1024, backup_count=3)

    assert path == fake_path
    mkdir_mock.assert_called_once_with(parents=True, exist_ok=True)
    handler_cls.assert_called_once_with(
        fake_path,
        maxBytes=1024,
        backupCount=3,
        encoding="utf-8",
    )
    assert fake_handler in logger.handlers
    assert logger.disabled is False


def test_connector_does_not_emit_eeg_log_when_debug_disabled(tmp_path):
    connector = MuseConnector(
        on_eeg=lambda _ch, _samples: None,
        known_devices_path=str(tmp_path / "devices.json"),
        debug_logging_enabled=False,
    )
    connector._device_state["streaming"] = True

    with patch.object(connector._eeg_debug_logger, "info") as eeg_info:
        handler = connector._make_eeg_handler("AF7")
        handler(None, bytearray(b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"))

    eeg_info.assert_not_called()
