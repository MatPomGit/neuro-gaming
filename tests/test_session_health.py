"""Testy logiki monitoringu SessionHealth i fallbacku."""

from src.game_controller import GameController
from src.session_health import SessionHealth, SessionHealthMonitor, SessionHealthThresholds
from src.signal_processor import DIRECTION_FORWARD


def test_critical_dropout_forces_keyboard_fallback() -> None:
    """Krytyczny dropout powinien wymusić przejście na keyboard mode."""
    monitor = SessionHealthMonitor()
    assessment = monitor.evaluate(
        SessionHealth(
            battery_level=55,
            signal_quality=0.62,
            reconnect_count=1,
            dropout_rate=13.5,
        )
    )

    assert assessment.switch_to_keyboard_mode is True
    assert assessment.safe_pause is True
    assert assessment.level == "CRITICAL"


def test_low_battery_is_warning_without_forced_pause() -> None:
    """Niska bateria powinna dać ostrzeżenie, ale bez blokady manualnego sterowania."""
    monitor = SessionHealthMonitor(SessionHealthThresholds(low_battery_warning=30))
    assessment = monitor.evaluate(
        SessionHealth(
            battery_level=20,
            signal_quality=0.8,
            reconnect_count=0,
            dropout_rate=1.2,
        )
    )

    assert assessment.level == "WARNING"
    assert assessment.safe_pause is False
    assert assessment.switch_to_keyboard_mode is False


def test_safe_pause_does_not_block_manual_keyboard_control() -> None:
    """Alarmy jakości nie mogą blokować ręcznego sterowania klawiaturą."""
    monitor = SessionHealthMonitor(SessionHealthThresholds(low_signal_pause=0.5))
    assessment = monitor.evaluate(
        SessionHealth(
            battery_level=80,
            signal_quality=0.2,
            reconnect_count=0,
            dropout_rate=0.0,
        )
    )
    assert assessment.safe_pause is True

    controller = GameController()
    handled = controller.handle_key_down("up")

    assert handled is True
    assert controller.current_direction == DIRECTION_FORWARD
