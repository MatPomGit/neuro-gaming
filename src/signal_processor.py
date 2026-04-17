"""
EEG signal processor for the NeuroGaming app.

Maintains a one-second rolling buffer for each of the four Muse S
channels (TP9, AF7, AF8, TP10), computes alpha (8-13 Hz) and beta
(13-30 Hz) band powers via FFT, and maps the resulting metrics to
one of five directional commands: FORWARD, BACKWARD, LEFT, RIGHT,
or NONE.

Control mapping
---------------
+----------+---------------------------------------+
| Command  | Condition                             |
+==========+=======================================+
| FORWARD  | mean beta > BETA_THRESHOLD            |
|          | (concentration / active focus)        |
+----------+---------------------------------------+
| BACKWARD | mean alpha > ALPHA_THRESHOLD          |
|          | AND mean alpha > mean beta            |
|          | (relaxation)                          |
+----------+---------------------------------------+
| LEFT     | AF7 alpha > AF8 alpha × ASYM_FACTOR   |
|          | (right hemisphere dominant)           |
+----------+---------------------------------------+
| RIGHT    | AF8 alpha > AF7 alpha × ASYM_FACTOR   |
|          | (left hemisphere dominant)            |
+----------+---------------------------------------+
| NONE     | none of the above thresholds met      |
+----------+---------------------------------------+
"""

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.calibration_profiles import CalibrationProfile, CalibrationProfileStore
from src.settings import AppSettings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 256          # Hz
BUFFER_SIZE = SAMPLE_RATE  # one-second window
FREQ_RESOLUTION = SAMPLE_RATE / BUFFER_SIZE  # = 1 Hz

# EEG frequency band limits (Hz)
ALPHA_LOW, ALPHA_HIGH = 8.0, 13.0
BETA_LOW,  BETA_HIGH  = 13.0, 30.0

# Decision thresholds (µV² – tuned empirically, adjustable at runtime)
BETA_THRESHOLD  = 2.0   # µV² – mean beta power for FORWARD
ALPHA_THRESHOLD = 2.0   # µV² – mean alpha power for BACKWARD
ASYM_FACTOR     = 1.3   # ratio of left vs. right alpha for LEFT/RIGHT
MIN_CONFIDENCE = 0.35

# Artefact rejection
MAX_SAMPLE_AMPLITUDE = 300.0   # µV, gross movement/blink artefact
POWER_SPIKE_FACTOR = 4.0       # sudden power jump/drop between windows
MOTION_ARTEFACT_LATCH_SECONDS = 0.8
MOTION_ARTEFACT_ACCEL_THRESHOLD = 2.4
MOTION_ARTEFACT_GYRO_THRESHOLD = 130.0
LOW_QUALITY_HOLD_SECONDS = 1.2
WARMUP_AFTER_RECONNECT_SECONDS = 1.5

# Adaptive baselines
EMA_ALPHA = 0.2

# Direction constants
DIRECTION_NONE     = "NONE"
DIRECTION_FORWARD  = "FORWARD"
DIRECTION_BACKWARD = "BACKWARD"
DIRECTION_LEFT     = "LEFT"
DIRECTION_RIGHT    = "RIGHT"
LOW_QUALITY_HOLD = "LOW_QUALITY_HOLD"

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
CALIBRATION_STAGES = ("baseline_rest", "focus_task", "relax_task")
MIN_CALIBRATION_SECONDS = 4.0
MIN_CALIBRATION_QUALITY = 0.35


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _band_power(signal: np.ndarray, low: float, high: float, fs: int = SAMPLE_RATE) -> float:
    """Return average power of *signal* in the frequency band [low, high] Hz.

    Parameters
    ----------
    signal:
        1-D NumPy array of EEG samples (µV).
    low:
        Lower frequency boundary (Hz).
    high:
        Upper frequency boundary (Hz).
    fs:
        Sampling frequency (Hz).
    """
    n = len(signal)
    if n < 2:
        return 0.0
    # Apply Hann window to reduce spectral leakage
    windowed = signal * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(fft_vals) ** 2) / n
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(power[mask]))


# ──────────────────────────────────────────────────────────────────────────────
# SignalProcessor
# ──────────────────────────────────────────────────────────────────────────────

class SignalProcessor:
    """Processes raw EEG samples and produces directional game commands.

    Thread safety
    -------------
    ``add_samples`` may be called from the BLE background thread.
    ``get_direction`` and ``get_metrics`` are typically called from the
    Kivy main thread via ``Clock.schedule_interval``.  Access to the
    internal deque objects is protected by a simple lock.

    Calibration
    -----------
    Call ``start_calibration()`` / ``stop_calibration()`` to record a
    baseline while the user sits still with eyes open.  Subsequent band
    powers are expressed relative to the baseline using a simple Z-score.
    Before calibration the raw µV² values are used directly.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        import threading
        self._lock = threading.Lock()
        self._buffers: dict[str, deque] = {
            ch: deque(maxlen=BUFFER_SIZE) for ch in CHANNELS
        }
        # Calibration state
        self._calibrating = False
        self._calib_samples: dict[str, list] = {ch: [] for ch in CHANNELS}
        self._stage_samples: dict[str, dict[str, list[float]]] = {}
        self._current_stage: str | None = None
        self._baseline_mean: dict[str, dict[str, float]] = {}
        self._baseline_std:  dict[str, dict[str, float]] = {}
        self.profile_store = CalibrationProfileStore()

        # Adjustable thresholds (can be changed by the UI)
        self.beta_threshold = BETA_THRESHOLD
        self.alpha_threshold = ALPHA_THRESHOLD
        self.asym_factor = ASYM_FACTOR
        self.min_confidence = MIN_CONFIDENCE
        # Offsety startują od zera, aby próg dynamiczny na starcie był
        # równy progowi skonfigurowanemu przez użytkownika.
        self.alpha_offset = 0.0
        self.beta_offset = 0.0
        self._ema_band: dict[str, dict[str, float | None]] = {
            "AF7": {"alpha": None, "beta": None},
            "AF8": {"alpha": None, "beta": None},
        }
        self._last_total_power: dict[str, float | None] = {ch: None for ch in CHANNELS}
        self._artifact_channels: dict[str, bool] = {ch: False for ch in CHANNELS}
        self._power_spike_channels: dict[str, bool] = {ch: False for ch in CHANNELS}
        self._saturated_channels: dict[str, bool] = {ch: False for ch in CHANNELS}
        self._last_confidence = 0.0
        # Znacznik artefaktu ruchowego (IMU) blokujący decyzje EEG przez krótki czas.
        self._motion_artifact_until = 0.0
        # Ostatni stabilny kierunek, który możemy utrzymać podczas spadku jakości.
        self._last_stable_direction = DIRECTION_NONE
        self._last_stable_direction_ts = 0.0
        self._processing_state = DIRECTION_NONE
        self._channel_quality_cache: dict[str, float] = {ch: 0.0 for ch in CHANNELS}
        self._global_quality_score = 0.0
        self._session_quality_score = 0.0
        self._asymmetry_sign_history: deque[int] = deque(maxlen=6)
        self._temporal_asymmetry_unstable = False
        self._warmup_until = 0.0
        if settings is not None:
            self.apply_settings(settings)

    def apply_settings(self, settings: AppSettings) -> None:
        """Apply runtime thresholds from the shared application settings."""
        self.beta_threshold = settings.beta_threshold
        self.alpha_threshold = settings.alpha_threshold
        self.asym_factor = settings.asym_factor
        # Po wczytaniu ustawień zachowujemy offset adaptacyjny jako 0,
        # żeby uniknąć podwajania efektywnego progu decyzji.
        self.beta_offset = 0.0
        self.alpha_offset = 0.0

    # ── data ingestion ─────────────────────────────────────────────────────

    def add_samples(self, channel: str, samples: np.ndarray) -> None:
        """Append *samples* to the rolling buffer for *channel*."""
        if channel not in self._buffers:
            return
        if len(samples) and float(np.max(np.abs(samples))) > MAX_SAMPLE_AMPLITUDE:
            with self._lock:
                self._artifact_channels[channel] = True
                self._saturated_channels[channel] = True
            return
        with self._lock:
            self._buffers[channel].extend(samples.tolist())
            if len(samples):
                # Saturacja kanału: wartości blisko maksymalnej amplitudy.
                self._saturated_channels[channel] = (
                    float(np.max(np.abs(samples))) > MAX_SAMPLE_AMPLITUDE * 0.9
                )
            if self._calibrating:
                self._calib_samples[channel].extend(samples.tolist())
                if self._current_stage:
                    self._stage_samples[self._current_stage][channel].extend(samples.tolist())

    def reset(self) -> None:
        """Clear all sample buffers."""
        with self._lock:
            for buf in self._buffers.values():
                buf.clear()
            self._last_total_power = {ch: None for ch in CHANNELS}
            self._artifact_channels = {ch: False for ch in CHANNELS}
            self._power_spike_channels = {ch: False for ch in CHANNELS}
            self._saturated_channels = {ch: False for ch in CHANNELS}
            self._last_confidence = 0.0
            self._motion_artifact_until = 0.0
            self._last_stable_direction = DIRECTION_NONE
            self._last_stable_direction_ts = 0.0
            self._processing_state = DIRECTION_NONE
            self._channel_quality_cache = {ch: 0.0 for ch in CHANNELS}
            self._global_quality_score = 0.0
            self._session_quality_score = 0.0
            self._asymmetry_sign_history.clear()
            self._temporal_asymmetry_unstable = False
            self._warmup_until = 0.0

    def notify_stream_reconnected(self, warmup_seconds: float = WARMUP_AFTER_RECONNECT_SECONDS) -> None:
        """Aktywuje warm-up po reconnect, aby chwilowo zablokować sterowanie."""
        with self._lock:
            self._warmup_until = time.monotonic() + max(0.0, warmup_seconds)
            self._processing_state = "WARMUP"

    def add_imu_frame(self, sensor: str, samples: np.ndarray) -> None:
        """Analizuje ramkę IMU i oznacza potencjalny artefakt ruchowy.

        Parametry
        ---------
        sensor:
            Nazwa sensora: ``\"accelerometer\"`` lub ``\"gyroscope\"``.
        samples:
            Tablica ``N×3`` z osiami XYZ.
        """
        if samples.size == 0:
            return
        norms = np.linalg.norm(samples, axis=1)
        threshold = (
            MOTION_ARTEFACT_ACCEL_THRESHOLD
            if sensor == "accelerometer"
            else MOTION_ARTEFACT_GYRO_THRESHOLD
        )
        if float(np.max(norms)) < threshold:
            return
        with self._lock:
            self._motion_artifact_until = max(
                self._motion_artifact_until,
                time.monotonic() + MOTION_ARTEFACT_LATCH_SECONDS,
            )

    # ── calibration ────────────────────────────────────────────────────────

    def start_calibration(self, stage: str = "baseline_rest") -> None:
        """Rozpoczyna nagrywanie etapu kalibracji."""
        if stage not in CALIBRATION_STAGES:
            raise ValueError(f"Unsupported calibration stage: {stage}")
        with self._lock:
            self._calibrating = True
            self._current_stage = stage
            if stage == "baseline_rest":
                self._calib_samples = {ch: [] for ch in CHANNELS}
                self._stage_samples = {}
            self._stage_samples.setdefault(stage, {ch: [] for ch in CHANNELS})
        logger.info("Calibration stage started: %s", stage)

    def stop_calibration(self) -> None:
        """Kończy bieżący etap kalibracji bez finalizacji profilu."""
        with self._lock:
            # Zgodność wsteczna: pojedynczy etap baseline ma od razu
            # aktualizować statystyki normalizacji Z-score.
            if self._current_stage == "baseline_rest":
                baseline_metrics = self._compute_stage_metrics(self._calib_samples)
                self._baseline_mean = {
                    ch: {
                        "alpha": baseline_metrics[ch]["alpha_mean"],
                        "beta": baseline_metrics[ch]["beta_mean"],
                    }
                    for ch in CHANNELS
                }
                self._baseline_std = {
                    ch: {
                        "alpha": baseline_metrics[ch]["alpha_std"],
                        "beta": baseline_metrics[ch]["beta_std"],
                    }
                    for ch in CHANNELS
                }
            self._calibrating = False
            self._current_stage = None

    def _compute_stage_metrics(
        self, stage_data: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Liczy metryki pasm dla pojedynczego etapu kalibracji."""
        metrics: dict[str, dict[str, float]] = {}
        for ch in CHANNELS:
            data = np.array(stage_data[ch], dtype=np.float32)
            alpha_powers: list[float] = []
            beta_powers: list[float] = []
            for start in range(0, len(data) - BUFFER_SIZE + 1, BUFFER_SIZE // 2):
                window = data[start: start + BUFFER_SIZE]
                alpha_powers.append(_band_power(window, ALPHA_LOW, ALPHA_HIGH))
                beta_powers.append(_band_power(window, BETA_LOW, BETA_HIGH))
            metrics[ch] = {
                "alpha_mean": float(np.mean(alpha_powers)) if alpha_powers else 0.0,
                "alpha_std": max(float(np.std(alpha_powers)), 1e-6) if alpha_powers else 1e-6,
                "beta_mean": float(np.mean(beta_powers)) if beta_powers else 0.0,
                "beta_std": max(float(np.std(beta_powers)), 1e-6) if beta_powers else 1e-6,
            }
        return metrics

    def _estimate_stage_quality(self, stage_data: dict[str, list[float]]) -> float:
        """Szacuje jakość etapu kalibracji na podstawie wariancji sygnału."""
        scores: list[float] = []
        for ch in CHANNELS:
            signal = np.array(stage_data[ch], dtype=np.float32)
            if len(signal) < BUFFER_SIZE // 2:
                scores.append(0.0)
                continue
            var = float(np.var(signal))
            if var < 0.5 or var > 10_000:
                scores.append(0.1)
            else:
                scores.append(min(1.0, var / 100.0))
        return float(np.mean(scores)) if scores else 0.0

    def finalize_calibration_profile(
        self,
        profile_id: str,
        user_id: str,
        device_address: str,
        firmware_if_available: str | None = None,
    ) -> CalibrationProfile:
        """Finalizuje kalibrację, waliduje sesję i zapisuje profil."""
        with self._lock:
            missing = [stage for stage in CALIBRATION_STAGES if stage not in self._stage_samples]
            if missing:
                raise ValueError(f"Brak etapów kalibracji: {', '.join(missing)}")
            for stage in CALIBRATION_STAGES:
                lengths = [len(self._stage_samples[stage][ch]) for ch in CHANNELS]
                min_required = int(MIN_CALIBRATION_SECONDS * SAMPLE_RATE)
                if min(lengths) < min_required:
                    raise ValueError(
                        f"Etap '{stage}' ma zbyt mało danych ({min(lengths)} < {min_required})."
                    )
                quality = self._estimate_stage_quality(self._stage_samples[stage])
                if quality < MIN_CALIBRATION_QUALITY:
                    raise ValueError(
                        f"Etap '{stage}' ma zbyt niską jakość sygnału ({quality:.2f})."
                    )

            stage_metrics = {
                stage: self._compute_stage_metrics(self._stage_samples[stage])
                for stage in CALIBRATION_STAGES
            }
            self._baseline_mean = {
                ch: {
                    "alpha": stage_metrics["baseline_rest"][ch]["alpha_mean"],
                    "beta": stage_metrics["baseline_rest"][ch]["beta_mean"],
                }
                for ch in CHANNELS
            }
            self._baseline_std = {
                ch: {
                    "alpha": stage_metrics["baseline_rest"][ch]["alpha_std"],
                    "beta": stage_metrics["baseline_rest"][ch]["beta_std"],
                }
                for ch in CHANNELS
            }

        focus_beta = (
            stage_metrics["focus_task"]["AF7"]["beta_mean"] +
            stage_metrics["focus_task"]["AF8"]["beta_mean"]
        ) / 2.0
        relax_alpha = (
            stage_metrics["relax_task"]["AF7"]["alpha_mean"] +
            stage_metrics["relax_task"]["AF8"]["alpha_mean"]
        ) / 2.0
        initial_thresholds = {
            "beta_threshold": max(0.1, round(focus_beta * 0.85, 6)),
            "alpha_threshold": max(0.1, round(relax_alpha * 0.85, 6)),
            "asym_factor": self.asym_factor,
        }

        profile = CalibrationProfile(
            profile_id=profile_id,
            user_id=user_id,
            device_address=device_address,
            timestamp=datetime.now(timezone.utc).isoformat(),
            firmware_if_available=firmware_if_available,
            baseline_mean=self._baseline_mean,
            baseline_std=self._baseline_std,
            stage_metrics=stage_metrics,
            initial_thresholds=initial_thresholds,
        )
        self.profile_store.save(profile)
        self.apply_calibration_profile(profile)
        logger.info("Calibration complete and profile saved: %s", profile.profile_id)
        return profile

    def apply_calibration_profile(self, profile: CalibrationProfile) -> None:
        """Aplikuje profil kalibracyjny jako aktywny punkt startowy."""
        with self._lock:
            self._baseline_mean = profile.baseline_mean
            self._baseline_std = profile.baseline_std
        self.beta_threshold = float(profile.initial_thresholds.get("beta_threshold", self.beta_threshold))
        self.alpha_threshold = float(profile.initial_thresholds.get("alpha_threshold", self.alpha_threshold))
        self.asym_factor = float(profile.initial_thresholds.get("asym_factor", self.asym_factor))

    # ── metrics & direction ────────────────────────────────────────────────

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Return current alpha and beta band powers for all channels.

        Returns
        -------
        dict mapping channel name → {"alpha": float, "beta": float}.
        Values are in µV² (raw) or Z-scores (after calibration).
        """
        metrics: dict[str, dict[str, float]] = {}
        with self._lock:
            for ch, buf in self._buffers.items():
                signal = np.array(buf, dtype=np.float32)
                if len(signal) < BUFFER_SIZE // 2:
                    metrics[ch] = {"alpha": 0.0, "beta": 0.0}
                    continue
                alpha = _band_power(signal, ALPHA_LOW, ALPHA_HIGH)
                beta  = _band_power(signal, BETA_LOW,  BETA_HIGH)
                total_power = alpha + beta
                prev_power = self._last_total_power[ch]
                if self._artifact_channels[ch]:
                    alpha, beta = 0.0, 0.0
                    self._artifact_channels[ch] = False
                    self._power_spike_channels[ch] = False
                elif prev_power is not None and prev_power > 1e-6:
                    ratio = total_power / prev_power
                    if ratio > POWER_SPIKE_FACTOR or ratio < (1.0 / POWER_SPIKE_FACTOR):
                        alpha, beta = 0.0, 0.0
                        self._power_spike_channels[ch] = True
                    else:
                        self._power_spike_channels[ch] = False
                else:
                    self._power_spike_channels[ch] = False
                self._last_total_power[ch] = max(total_power, 1e-6)
                if ch in self._baseline_mean:
                    bm = self._baseline_mean[ch]
                    bs = self._baseline_std[ch]
                    alpha = (alpha - bm["alpha"]) / bs["alpha"]
                    beta  = (beta  - bm["beta"])  / bs["beta"]
                metrics[ch] = {"alpha": alpha, "beta": beta}
        return metrics

    def _get_dynamic_thresholds(self) -> tuple[float, float]:
        """Return dynamic (beta, alpha) thresholds using EMA baselines + offset."""
        alpha_vals = [self._ema_band[ch]["alpha"] for ch in ("AF7", "AF8")]
        beta_vals = [self._ema_band[ch]["beta"] for ch in ("AF7", "AF8")]
        alpha_valid = [v for v in alpha_vals if v is not None]
        beta_valid = [v for v in beta_vals if v is not None]
        dyn_alpha = (float(np.mean(alpha_valid)) if alpha_valid else self.alpha_threshold) + self.alpha_offset
        dyn_beta = (float(np.mean(beta_valid)) if beta_valid else self.beta_threshold) + self.beta_offset
        return dyn_beta, dyn_alpha

    def _update_ema_baselines(self, metrics: dict[str, dict[str, float]]) -> None:
        for ch in ("AF7", "AF8"):
            for band in ("alpha", "beta"):
                current = metrics[ch][band]
                prev = self._ema_band[ch][band]
                self._ema_band[ch][band] = current if prev is None else (EMA_ALPHA * current + (1.0 - EMA_ALPHA) * prev)

    def get_direction_confidence(self) -> float:
        """Return confidence (0-1) for the latest direction decision."""
        return self._last_confidence

    def get_processing_state(self) -> str:
        """Zwraca aktualny stan pipeline'u decyzyjnego."""
        return self._processing_state

    def get_quality_snapshot(self) -> dict[str, float | str | dict[str, float]]:
        """Zwraca per-kanał i globalne metryki jakości do publikacji w UI."""
        channel_quality = self.get_signal_quality()
        return {
            "channels": channel_quality,
            "global_score": self._global_quality_score,
            "session_score": self._session_quality_score,
            "state": self._processing_state,
            "warmup_remaining": max(0.0, self._warmup_until - time.monotonic()),
        }

    def _update_temporal_asymmetry(self, metrics: dict[str, dict[str, float]]) -> None:
        """Aktualizuje heurystykę niestabilnej asymetrii międzypółkulowej."""
        alpha_diff = metrics["AF7"]["alpha"] - metrics["AF8"]["alpha"]
        if abs(alpha_diff) < 0.2:
            self._temporal_asymmetry_unstable = False
            return
        sign = 1 if alpha_diff > 0 else -1
        self._asymmetry_sign_history.append(sign)
        if len(self._asymmetry_sign_history) < 4:
            return
        flips = sum(
            1
            for idx in range(1, len(self._asymmetry_sign_history))
            if self._asymmetry_sign_history[idx] != self._asymmetry_sign_history[idx - 1]
        )
        # 2+ zmian znaku w krótkim oknie wskazuje artefakt (mrugnięcia/ruch).
        self._temporal_asymmetry_unstable = flips >= 2

    def _quality_gate(self, metrics: dict[str, dict[str, float]]) -> tuple[bool, str]:
        """Wykonuje quality gate przed mapowaniem kierunku."""
        quality_scores = self.get_signal_quality()
        frontal_quality = (quality_scores["AF7"] + quality_scores["AF8"]) / 2.0
        self._update_temporal_asymmetry(metrics)
        now = time.monotonic()

        if now < self._warmup_until:
            return False, "WARMUP"
        if self.is_motion_artifact_active():
            return False, "IMU_MOTION"
        if frontal_quality < 0.35:
            return False, "LOW_CHANNEL_QUALITY"
        if self._temporal_asymmetry_unstable:
            return False, "TEMPORAL_ASYMMETRY"
        return True, "OK"

    def get_direction(self) -> str:
        """Determine the current directional command from EEG metrics.

        Returns one of the ``DIRECTION_*`` constants defined in this module.
        """
        m = self.get_metrics()
        gate_passed, gate_reason = self._quality_gate(m)
        now = time.monotonic()
        if not gate_passed:
            hold_is_valid = (
                self._last_stable_direction != DIRECTION_NONE
                and (now - self._last_stable_direction_ts) <= LOW_QUALITY_HOLD_SECONDS
            )
            if hold_is_valid and gate_reason not in {"WARMUP", "IMU_MOTION"}:
                self._processing_state = LOW_QUALITY_HOLD
                self._last_confidence = max(0.0, min(1.0, self._global_quality_score))
                return self._last_stable_direction
            self._processing_state = gate_reason
            self._last_confidence = 0.0
            return DIRECTION_NONE

        # Average beta and alpha across forehead channels (most informative)
        beta_avg  = (m["AF7"]["beta"]  + m["AF8"]["beta"])  / 2
        alpha_avg = (m["AF7"]["alpha"] + m["AF8"]["alpha"]) / 2

        alpha_left  = m["AF7"]["alpha"]   # left forehead
        alpha_right = m["AF8"]["alpha"]   # right forehead
        dyn_beta_threshold, dyn_alpha_threshold = self._get_dynamic_thresholds()

        # Margin-based confidence terms
        beta_margin = (beta_avg - dyn_beta_threshold) / (abs(dyn_beta_threshold) + 1e-6)
        left_ratio = alpha_left / (abs(alpha_right) + 1e-6)
        right_ratio = alpha_right / (abs(alpha_left) + 1e-6)
        left_margin = (left_ratio - self.asym_factor) / self.asym_factor
        right_margin = (right_ratio - self.asym_factor) / self.asym_factor
        backward_margin = min(
            (alpha_avg - dyn_alpha_threshold) / (abs(dyn_alpha_threshold) + 1e-6),
            (alpha_avg - beta_avg) / (abs(alpha_avg) + abs(beta_avg) + 1e-6),
        )
        quality = self._global_quality_score

        # Priority: FORWARD > lateral (LEFT/RIGHT) > BACKWARD > NONE
        direction = DIRECTION_NONE
        margin = 0.0
        if beta_avg > dyn_beta_threshold:
            direction, margin = DIRECTION_FORWARD, beta_margin
        elif alpha_left > alpha_right * self.asym_factor:
            direction, margin = DIRECTION_LEFT, left_margin
        elif alpha_right > alpha_left * self.asym_factor:
            direction, margin = DIRECTION_RIGHT, right_margin
        elif alpha_avg > dyn_alpha_threshold and alpha_avg > beta_avg:
            direction, margin = DIRECTION_BACKWARD, backward_margin

        margin_norm = max(0.0, min(1.0, margin))
        self._last_confidence = float(max(0.0, min(1.0, 0.65 * quality + 0.35 * margin_norm)))
        self._update_ema_baselines(m)
        if direction == DIRECTION_NONE or self._last_confidence < self.min_confidence:
            self._processing_state = DIRECTION_NONE
            return DIRECTION_NONE
        self._last_stable_direction = direction
        self._last_stable_direction_ts = now
        self._processing_state = "ACTIVE"
        return direction

    def is_motion_artifact_active(self) -> bool:
        """Zwraca ``True``, gdy ostatni ruch głowy może zafałszować EEG."""
        with self._lock:
            return time.monotonic() < self._motion_artifact_until

    def get_signal_quality(self) -> dict[str, float]:
        """Return a 0–1 signal-quality score for each channel.

        A score of 1 means the buffer is full and the signal variance is
        within a reasonable range.  Used to indicate electrode contact
        quality in the UI.
        """
        scores: dict[str, float] = {}
        with self._lock:
            motion_active = time.monotonic() < self._motion_artifact_until
            for ch, buf in self._buffers.items():
                if len(buf) < BUFFER_SIZE // 4:
                    scores[ch] = 0.0
                    continue
                signal = np.array(buf, dtype=np.float32)
                fill  = len(buf) / BUFFER_SIZE
                # Variance-based quality: very low or very high variance → bad contact
                var = float(np.var(signal))
                # Typical range: 1–500 µV²; penalise outside this range
                if var < 0.5:
                    quality = 0.1   # flat – likely disconnected
                elif var > 10_000:
                    quality = 0.1   # saturated / movement artefact
                else:
                    quality = min(1.0, var / 100.0)
                # Heurystyki artefaktów obniżające score jakości.
                if self._power_spike_channels.get(ch, False):
                    quality *= 0.25
                if self._saturated_channels.get(ch, False):
                    quality *= 0.2
                if motion_active:
                    quality *= 0.35
                if self._temporal_asymmetry_unstable and ch in {"AF7", "AF8"}:
                    quality *= 0.5
                scores[ch] = fill * quality
            self._channel_quality_cache = dict(scores)
            self._global_quality_score = float(sum(scores.values()) / max(1, len(scores)))
            # Globalny score sesji jako wygładzana EMA bieżącej jakości.
            if self._session_quality_score == 0.0:
                self._session_quality_score = self._global_quality_score
            else:
                self._session_quality_score = float(
                    0.1 * self._global_quality_score + 0.9 * self._session_quality_score
                )
        return scores
