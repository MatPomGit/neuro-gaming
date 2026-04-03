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
from collections import deque
from typing import Optional

import numpy as np

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

# Direction constants
DIRECTION_NONE     = "NONE"
DIRECTION_FORWARD  = "FORWARD"
DIRECTION_BACKWARD = "BACKWARD"
DIRECTION_LEFT     = "LEFT"
DIRECTION_RIGHT    = "RIGHT"

CHANNELS = ("TP9", "AF7", "AF8", "TP10")


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

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._buffers: dict[str, deque] = {
            ch: deque(maxlen=BUFFER_SIZE) for ch in CHANNELS
        }
        # Calibration state
        self._calibrating = False
        self._calib_samples: dict[str, list] = {ch: [] for ch in CHANNELS}
        self._baseline_mean: dict[str, dict[str, float]] = {}
        self._baseline_std:  dict[str, dict[str, float]] = {}

        # Adjustable thresholds (can be changed by the UI)
        self.beta_threshold  = BETA_THRESHOLD
        self.alpha_threshold = ALPHA_THRESHOLD
        self.asym_factor     = ASYM_FACTOR

    # ── data ingestion ─────────────────────────────────────────────────────

    def add_samples(self, channel: str, samples: np.ndarray) -> None:
        """Append *samples* to the rolling buffer for *channel*."""
        if channel not in self._buffers:
            return
        with self._lock:
            self._buffers[channel].extend(samples.tolist())
            if self._calibrating:
                self._calib_samples[channel].extend(samples.tolist())

    def reset(self) -> None:
        """Clear all sample buffers."""
        with self._lock:
            for buf in self._buffers.values():
                buf.clear()

    # ── calibration ────────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Begin recording calibration data (5-10 s recommended)."""
        with self._lock:
            self._calibrating = True
            self._calib_samples = {ch: [] for ch in CHANNELS}
        logger.info("Calibration started")

    def stop_calibration(self) -> None:
        """Finish calibration and compute baselines."""
        with self._lock:
            self._calibrating = False
            for ch in CHANNELS:
                data = np.array(self._calib_samples[ch], dtype=np.float32)
                if len(data) < BUFFER_SIZE:
                    continue
                # Compute band powers over consecutive 1-second windows
                alpha_powers, beta_powers = [], []
                for start in range(0, len(data) - BUFFER_SIZE + 1, BUFFER_SIZE // 2):
                    window = data[start: start + BUFFER_SIZE]
                    alpha_powers.append(_band_power(window, ALPHA_LOW, ALPHA_HIGH))
                    beta_powers.append(_band_power(window, BETA_LOW, BETA_HIGH))
                self._baseline_mean[ch] = {
                    "alpha": float(np.mean(alpha_powers)),
                    "beta":  float(np.mean(beta_powers)),
                }
                self._baseline_std[ch] = {
                    "alpha": max(float(np.std(alpha_powers)), 1e-6),
                    "beta":  max(float(np.std(beta_powers)),  1e-6),
                }
        logger.info("Calibration complete: %s", self._baseline_mean)

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
                if ch in self._baseline_mean:
                    bm = self._baseline_mean[ch]
                    bs = self._baseline_std[ch]
                    alpha = (alpha - bm["alpha"]) / bs["alpha"]
                    beta  = (beta  - bm["beta"])  / bs["beta"]
                metrics[ch] = {"alpha": alpha, "beta": beta}
        return metrics

    def get_direction(self) -> str:
        """Determine the current directional command from EEG metrics.

        Returns one of the ``DIRECTION_*`` constants defined in this module.
        """
        m = self.get_metrics()

        # Average beta and alpha across forehead channels (most informative)
        beta_avg  = (m["AF7"]["beta"]  + m["AF8"]["beta"])  / 2
        alpha_avg = (m["AF7"]["alpha"] + m["AF8"]["alpha"]) / 2

        alpha_left  = m["AF7"]["alpha"]   # left forehead
        alpha_right = m["AF8"]["alpha"]   # right forehead

        # Priority: FORWARD > lateral (LEFT/RIGHT) > BACKWARD > NONE
        if beta_avg > self.beta_threshold:
            return DIRECTION_FORWARD

        if alpha_left > alpha_right * self.asym_factor:
            return DIRECTION_LEFT

        if alpha_right > alpha_left * self.asym_factor:
            return DIRECTION_RIGHT

        if alpha_avg > self.alpha_threshold and alpha_avg > beta_avg:
            return DIRECTION_BACKWARD

        return DIRECTION_NONE

    def get_signal_quality(self) -> dict[str, float]:
        """Return a 0–1 signal-quality score for each channel.

        A score of 1 means the buffer is full and the signal variance is
        within a reasonable range.  Used to indicate electrode contact
        quality in the UI.
        """
        scores: dict[str, float] = {}
        with self._lock:
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
                scores[ch] = fill * quality
        return scores
