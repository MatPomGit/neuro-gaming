import pytest

from src.game_controller import GameController
from src.settings import AppSettings, load_settings, save_settings
from src.signal_processor import SignalProcessor


def test_settings_roundtrip_json(tmp_path):
    settings = AppSettings(
        beta_threshold=3.4,
        alpha_threshold=2.2,
        asym_factor=1.5,
        hysteresis_count=5,
        key_mode="wasd",
        forwarding_enabled=True,
        debug_logging=True,
        debug_eeg_file=True,
        debug_logging_enabled=True,
    )
    settings_path = tmp_path / "settings.json"

    save_settings(settings, settings_path)
    loaded = load_settings(settings_path)

    assert loaded == settings


def test_settings_from_dict_uses_legacy_debug_eeg_file_as_fallback():
    payload = {
        "beta_threshold": 2.0,
        "alpha_threshold": 2.0,
        "asym_factor": 1.3,
        "hysteresis_count": 3,
        "key_mode": "arrow",
        "forwarding_enabled": False,
        "debug_logging": False,
        "debug_eeg_file": True,
    }
    loaded = AppSettings.from_dict(payload)
    assert loaded.debug_logging_enabled is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("beta_threshold", -0.1),
        ("alpha_threshold", 200.0),
        ("asym_factor", 0.5),
        ("hysteresis_count", 0),
        ("key_mode", "invalid"),
    ],
)
def test_settings_validation_rejects_invalid_values(field, value):
    payload = AppSettings().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        AppSettings.from_dict(payload)


def test_controller_and_processor_use_shared_settings_object():
    settings = AppSettings(
        beta_threshold=4.2,
        alpha_threshold=1.7,
        asym_factor=1.8,
        hysteresis_count=7,
        key_mode="wasd",
        forwarding_enabled=True,
    )

    processor = SignalProcessor(settings=settings)
    controller = GameController(settings=settings)

    assert processor.beta_threshold == pytest.approx(4.2)
    assert processor.alpha_threshold == pytest.approx(1.7)
    assert processor.asym_factor == pytest.approx(1.8)
    assert controller.hysteresis_count == 7
    assert controller.key_mode == "wasd"
    assert controller.forwarding_enabled is True
