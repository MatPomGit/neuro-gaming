"""
Unit tests for NeuroGaming core modules.

Run with:
    pip install pytest numpy
    pytest tests/ -v
"""

import numpy as np
import pytest

from src.signal_processor import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_LEFT,
    DIRECTION_NONE,
    DIRECTION_RIGHT,
    SAMPLE_RATE,
    SignalProcessor,
    _band_power,
)
from src.game_controller import (
    ACTION_LEFT_CLICK,
    ACTION_RIGHT_CLICK,
    HYSTERESIS_COUNT,
    MOUSE_LEFT,
    MOUSE_RIGHT,
    GameController,
    _key_to_direction,
)


# ──────────────────────────────────────────────────────────────────────────────
# _band_power helper
# ──────────────────────────────────────────────────────────────────────────────

class TestBandPower:
    def test_pure_alpha_sine(self):
        """A 10 Hz sine should have high alpha power and near-zero beta."""
        t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        alpha = _band_power(signal, 8, 13)
        beta  = _band_power(signal, 13, 30)
        assert alpha > beta * 5

    def test_pure_beta_sine(self):
        """A 20 Hz sine should have high beta power and near-zero alpha."""
        t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
        signal = np.sin(2 * np.pi * 20 * t).astype(np.float32)
        alpha = _band_power(signal, 8, 13)
        beta  = _band_power(signal, 13, 30)
        assert beta > alpha * 5

    def test_zero_signal(self):
        signal = np.zeros(SAMPLE_RATE, dtype=np.float32)
        assert _band_power(signal, 8, 13) == pytest.approx(0.0, abs=1e-6)

    def test_empty_signal(self):
        assert _band_power(np.array([]), 8, 13) == 0.0

    def test_short_signal_returns_zero(self):
        assert _band_power(np.array([1.0]), 8, 13) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SignalProcessor
# ──────────────────────────────────────────────────────────────────────────────

def _fill_processor_with_sine(processor: SignalProcessor, freq: float, amplitude: float = 10.0) -> None:
    """Fill all channels with a sine wave at *freq* Hz."""
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    for ch in ("TP9", "AF7", "AF8", "TP10"):
        # Feed in chunks of 5 (as the Muse connector delivers)
        for start in range(0, SAMPLE_RATE, 5):
            processor.add_samples(ch, signal[start: start + 5])


class TestSignalProcessor:
    def test_initial_state_returns_none_direction(self):
        proc = SignalProcessor()
        assert proc.get_direction() == DIRECTION_NONE

    def test_beta_dominant_gives_forward(self):
        proc = SignalProcessor()
        proc.beta_threshold = 0.1   # lower threshold so 20 Hz sine triggers it
        _fill_processor_with_sine(proc, freq=20.0, amplitude=20.0)
        direction = proc.get_direction()
        assert direction == DIRECTION_FORWARD

    def test_metrics_returns_all_channels(self):
        proc = SignalProcessor()
        _fill_processor_with_sine(proc, freq=10.0)
        m = proc.get_metrics()
        assert set(m.keys()) == {"TP9", "AF7", "AF8", "TP10"}
        for ch_data in m.values():
            assert "alpha" in ch_data
            assert "beta"  in ch_data

    def test_reset_clears_buffers(self):
        proc = SignalProcessor()
        _fill_processor_with_sine(proc, freq=10.0)
        proc.reset()
        m = proc.get_metrics()
        for ch_data in m.values():
            assert ch_data["alpha"] == pytest.approx(0.0, abs=1e-6)

    def test_signal_quality_zero_when_empty(self):
        proc = SignalProcessor()
        q = proc.get_signal_quality()
        for score in q.values():
            assert score == 0.0

    def test_signal_quality_nonzero_after_fill(self):
        proc = SignalProcessor()
        _fill_processor_with_sine(proc, freq=10.0)
        q = proc.get_signal_quality()
        # At least the forehead channels should show non-zero quality
        assert q["AF7"] > 0
        assert q["AF8"] > 0

    def test_left_right_asymmetry_triggers_left(self):
        proc = SignalProcessor()
        proc.asym_factor = 1.2
        t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
        alpha_signal = (15.0 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
        flat_signal  = np.zeros(SAMPLE_RATE, dtype=np.float32)

        # AF7 (left) has strong alpha, AF8 (right) is silent → LEFT
        for start in range(0, SAMPLE_RATE, 5):
            proc.add_samples("AF7",  alpha_signal[start: start + 5])
            proc.add_samples("AF8",  flat_signal[start: start + 5])
            proc.add_samples("TP9",  flat_signal[start: start + 5])
            proc.add_samples("TP10", flat_signal[start: start + 5])

        direction = proc.get_direction()
        assert direction == DIRECTION_LEFT


# ──────────────────────────────────────────────────────────────────────────────
# GameController
# ──────────────────────────────────────────────────────────────────────────────

class TestGameController:
    def test_initial_direction_is_none(self):
        ctrl = GameController()
        assert ctrl.current_direction == DIRECTION_NONE

    def test_hysteresis_requires_consecutive_updates(self):
        ctrl = GameController()
        for _ in range(HYSTERESIS_COUNT - 1):
            ctrl.update(DIRECTION_FORWARD)
        assert ctrl.current_direction == DIRECTION_NONE  # not yet committed

        ctrl.update(DIRECTION_FORWARD)
        assert ctrl.current_direction == DIRECTION_FORWARD

    def test_direction_change_resets_counter(self):
        ctrl = GameController()
        ctrl.update(DIRECTION_FORWARD)
        ctrl.update(DIRECTION_BACKWARD)   # different direction resets count (1 of HYSTERESIS_COUNT)
        for _ in range(HYSTERESIS_COUNT - 2):   # one more (total HYSTERESIS_COUNT - 1)
            ctrl.update(DIRECTION_BACKWARD)
        assert ctrl.current_direction == DIRECTION_NONE  # still not committed

    def test_on_direction_change_callback(self):
        received = []
        ctrl = GameController(on_direction_change=received.append)
        for _ in range(HYSTERESIS_COUNT):
            ctrl.update(DIRECTION_RIGHT)
        assert received == [DIRECTION_RIGHT]

    def test_set_direction_immediate(self):
        ctrl = GameController()
        ctrl.set_direction(DIRECTION_LEFT)
        assert ctrl.current_direction == DIRECTION_LEFT

    def test_reset(self):
        ctrl = GameController()
        ctrl.set_direction(DIRECTION_FORWARD)
        ctrl.reset()
        assert ctrl.current_direction == DIRECTION_NONE

    def test_arrow_key_mode(self):
        ctrl = GameController(key_mode="arrow")
        ctrl.set_direction(DIRECTION_FORWARD)
        assert ctrl.get_active_key() == "up"

    def test_wasd_key_mode(self):
        ctrl = GameController(key_mode="wasd")
        ctrl.set_direction(DIRECTION_FORWARD)
        assert ctrl.get_active_key() == "w"

    def test_handle_key_down_arrow(self):
        ctrl = GameController()
        assert ctrl.handle_key_down("up")
        assert ctrl.current_direction == DIRECTION_FORWARD

    def test_handle_key_down_wasd(self):
        ctrl = GameController()
        assert ctrl.handle_key_down("a")
        assert ctrl.current_direction == DIRECTION_LEFT

    def test_handle_key_up_clears_matching_direction(self):
        ctrl = GameController()
        ctrl.handle_key_down("down")
        ctrl.handle_key_up("down")
        assert ctrl.current_direction == DIRECTION_NONE

    def test_handle_key_up_ignores_non_matching_direction(self):
        ctrl = GameController()
        ctrl.handle_key_down("up")
        ctrl.handle_key_up("down")   # wrong key, should NOT clear
        assert ctrl.current_direction == DIRECTION_FORWARD

    def test_unknown_key_returns_false(self):
        ctrl = GameController()
        assert not ctrl.handle_key_down("space")
        assert not ctrl.handle_key_up("space")


class TestKeyMapping:
    @pytest.mark.parametrize("key,expected", [
        ("up",    DIRECTION_FORWARD),
        ("down",  DIRECTION_BACKWARD),
        ("left",  DIRECTION_LEFT),
        ("right", DIRECTION_RIGHT),
        ("w",     DIRECTION_FORWARD),
        ("s",     DIRECTION_BACKWARD),
        ("a",     DIRECTION_LEFT),
        ("d",     DIRECTION_RIGHT),
    ])
    def test_key_maps_to_direction(self, key, expected):
        assert _key_to_direction(key) == expected

    def test_unknown_key_returns_none(self):
        assert _key_to_direction("space") is None
        assert _key_to_direction("enter") is None


# ──────────────────────────────────────────────────────────────────────────────
# GameController – mouse button handling
# ──────────────────────────────────────────────────────────────────────────────

class TestMouseButtons:
    def test_initial_mouse_state_is_false(self):
        ctrl = GameController()
        assert ctrl.left_button_pressed is False
        assert ctrl.right_button_pressed is False

    def test_handle_mouse_down_left(self):
        ctrl = GameController()
        result = ctrl.handle_mouse_down(MOUSE_LEFT)
        assert result is True
        assert ctrl.left_button_pressed is True
        assert ctrl.right_button_pressed is False

    def test_handle_mouse_down_right(self):
        ctrl = GameController()
        result = ctrl.handle_mouse_down(MOUSE_RIGHT)
        assert result is True
        assert ctrl.right_button_pressed is True
        assert ctrl.left_button_pressed is False

    def test_handle_mouse_down_unknown_returns_false(self):
        ctrl = GameController()
        result = ctrl.handle_mouse_down("middle")
        assert result is False
        assert ctrl.left_button_pressed is False
        assert ctrl.right_button_pressed is False

    def test_handle_mouse_up_left(self):
        ctrl = GameController()
        ctrl.handle_mouse_down(MOUSE_LEFT)
        result = ctrl.handle_mouse_up(MOUSE_LEFT)
        assert result is True
        assert ctrl.left_button_pressed is False

    def test_handle_mouse_up_right(self):
        ctrl = GameController()
        ctrl.handle_mouse_down(MOUSE_RIGHT)
        result = ctrl.handle_mouse_up(MOUSE_RIGHT)
        assert result is True
        assert ctrl.right_button_pressed is False

    def test_handle_mouse_up_unknown_returns_false(self):
        ctrl = GameController()
        result = ctrl.handle_mouse_up("middle")
        assert result is False

    def test_on_mouse_action_callback_left(self):
        actions = []
        ctrl = GameController(on_mouse_action=actions.append)
        ctrl.handle_mouse_down(MOUSE_LEFT)
        assert actions == [ACTION_LEFT_CLICK]

    def test_on_mouse_action_callback_right(self):
        actions = []
        ctrl = GameController(on_mouse_action=actions.append)
        ctrl.handle_mouse_down(MOUSE_RIGHT)
        assert actions == [ACTION_RIGHT_CLICK]

    def test_mouse_up_does_not_fire_callback(self):
        actions = []
        ctrl = GameController(on_mouse_action=actions.append)
        ctrl.handle_mouse_down(MOUSE_LEFT)
        ctrl.handle_mouse_up(MOUSE_LEFT)
        # Callback only fires on press, not on release
        assert actions == [ACTION_LEFT_CLICK]

    def test_reset_clears_mouse_state(self):
        ctrl = GameController()
        ctrl.handle_mouse_down(MOUSE_LEFT)
        ctrl.handle_mouse_down(MOUSE_RIGHT)
        ctrl.reset()
        assert ctrl.left_button_pressed is False
        assert ctrl.right_button_pressed is False

    @pytest.mark.parametrize("button,action", [
        (MOUSE_LEFT,  ACTION_LEFT_CLICK),
        (MOUSE_RIGHT, ACTION_RIGHT_CLICK),
    ])
    def test_action_constants(self, button, action):
        actions = []
        ctrl = GameController(on_mouse_action=actions.append)
        ctrl.handle_mouse_down(button)
        assert actions[-1] == action


# ──────────────────────────────────────────────────────────────────────────────
# SignalProcessor – calibration
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalProcessorCalibration:
    def test_calibration_changes_metrics_to_zscores(self):
        """After calibration, band powers should be expressed as Z-scores."""
        proc = SignalProcessor()
        # Feed a 10 Hz sine to all channels for the calibration window
        _fill_processor_with_sine(proc, freq=10.0, amplitude=10.0)
        proc.start_calibration()
        _fill_processor_with_sine(proc, freq=10.0, amplitude=10.0)
        proc.stop_calibration()

        # With baseline == current signal the Z-score should be near 0
        m = proc.get_metrics()
        for ch in ("AF7", "AF8"):
            assert abs(m[ch]["alpha"]) < 2.0, (
                f"Expected near-zero alpha Z-score for {ch}, got {m[ch]['alpha']:.3f}"
            )

    def test_start_calibration_resets_samples(self):
        """Calling start_calibration twice should discard earlier samples."""
        proc = SignalProcessor()
        _fill_processor_with_sine(proc, freq=10.0)
        proc.start_calibration()
        _fill_processor_with_sine(proc, freq=10.0)
        proc.start_calibration()   # second call should clear the buffer
        # Internal calib sample lists should be empty after restart
        for ch in ("TP9", "AF7", "AF8", "TP10"):
            assert proc._calib_samples[ch] == []  # noqa: SLF001

    def test_stop_calibration_without_enough_data_is_safe(self):
        """stop_calibration with fewer than BUFFER_SIZE samples should not crash."""
        proc = SignalProcessor()
        proc.start_calibration()
        # Feed only 10 samples – far fewer than BUFFER_SIZE
        tiny = np.ones(10, dtype=np.float32)
        for ch in ("TP9", "AF7", "AF8", "TP10"):
            proc.add_samples(ch, tiny)
        proc.stop_calibration()   # must not raise
        # Metrics should still return zeros (no baseline established)
        m = proc.get_metrics()
        for ch in ("TP9", "AF7", "AF8", "TP10"):
            assert m[ch]["alpha"] == pytest.approx(0.0, abs=1e-3)

    def test_calibration_flag_is_cleared_after_stop(self):
        proc = SignalProcessor()
        proc.start_calibration()
        assert proc._calibrating is True  # noqa: SLF001
        proc.stop_calibration()
        assert proc._calibrating is False  # noqa: SLF001

    def test_samples_accumulated_during_calibration(self):
        """Samples added while calibrating should appear in _calib_samples."""
        proc = SignalProcessor()
        proc.start_calibration()
        signal = np.ones(50, dtype=np.float32)
        proc.add_samples("AF7", signal)
        assert len(proc._calib_samples["AF7"]) == 50  # noqa: SLF001

    def test_samples_not_accumulated_outside_calibration(self):
        """Samples added before calibration starts should not pollute calib buffer."""
        proc = SignalProcessor()
        signal = np.ones(50, dtype=np.float32)
        proc.add_samples("AF7", signal)
        proc.start_calibration()
        assert proc._calib_samples["AF7"] == []  # noqa: SLF001
