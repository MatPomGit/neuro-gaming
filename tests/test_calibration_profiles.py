"""Testy profili kalibracyjnych i ich spójności."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.calibration_profiles import CalibrationProfile, CalibrationProfileStore
from src.signal_processor import SAMPLE_RATE, SignalProcessor


def _feed_stage(processor: SignalProcessor, stage: str, freq: float, amplitude: float = 12.0) -> None:
    """Wypełnia etap kalibracji syntetycznym sygnałem EEG."""
    processor.start_calibration(stage)
    t = np.linspace(0, 8, SAMPLE_RATE * 8, endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    for channel in ("TP9", "AF7", "AF8", "TP10"):
        for start in range(0, len(signal), 32):
            processor.add_samples(channel, signal[start : start + 32])
    processor.stop_calibration()


def test_profile_serialization_roundtrip(tmp_path):
    """Sprawdza serializację/deserializację pliku profilu."""
    store = CalibrationProfileStore(tmp_path / "profiles")
    profile = CalibrationProfile(
        profile_id="userA_device1_v1",
        user_id="userA",
        device_address="AA:BB:CC:DD",
        timestamp=datetime.now(timezone.utc).isoformat(),
        firmware_if_available="1.2.3",
        baseline_mean={"AF7": {"alpha": 1.0, "beta": 2.0}},
        baseline_std={"AF7": {"alpha": 0.5, "beta": 0.4}},
        stage_metrics={"baseline_rest": {"AF7": {"alpha_mean": 1.0}}},
        initial_thresholds={"alpha_threshold": 1.1, "beta_threshold": 1.2, "asym_factor": 1.3},
    )

    store.save(profile)
    loaded = store.load("userA_device1_v1")

    assert loaded.to_dict() == profile.to_dict()


def test_same_profile_produces_same_initial_thresholds(tmp_path):
    """Weryfikuje, że ten sam profil daje identyczne progi startowe."""
    processor = SignalProcessor()
    processor.profile_store = CalibrationProfileStore(tmp_path / "profiles")

    _feed_stage(processor, "baseline_rest", freq=10.0, amplitude=11.0)
    _feed_stage(processor, "focus_task", freq=20.0, amplitude=16.0)
    _feed_stage(processor, "relax_task", freq=10.0, amplitude=15.0)

    profile = processor.finalize_calibration_profile(
        profile_id="stable_profile",
        user_id="default-user",
        device_address="11:22:33:44",
        firmware_if_available=None,
    )

    loaded_once = processor.profile_store.load("stable_profile")
    loaded_twice = processor.profile_store.load("stable_profile")

    assert loaded_once.initial_thresholds == loaded_twice.initial_thresholds
    assert loaded_once.initial_thresholds == profile.initial_thresholds
