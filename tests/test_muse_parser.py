"""
Tests for the Muse S EEG packet parser.
"""

import struct

import numpy as np
import pytest

from src.muse_connector import _parse_eeg_packet


def _pack_eeg_packet(sequence: int, samples: list[int]) -> bytes:
    """Build a synthetic 12-byte EEG notification packet.

    Parameters
    ----------
    sequence:
        16-bit packet sequence number.
    samples:
        List of 5 raw 12-bit integer sample values (0–4095).
    """
    assert len(samples) == 5
    header = struct.pack(">H", sequence & 0xFFFF)
    # Pack five 12-bit values into 7.5 bytes (round up to 8 bytes = 10 total)
    bits = 0
    for s in samples:
        bits = (bits << 12) | (s & 0xFFF)
    # 5 × 12 = 60 bits; pad to 64 bits (8 bytes)
    bits <<= 4
    payload = bits.to_bytes(8, "big")
    return header + payload[:8]


class TestParseEegPacket:
    def test_sequence_number(self):
        packet = _pack_eeg_packet(42, [2048, 2048, 2048, 2048, 2048])
        seq, _ = _parse_eeg_packet(packet)
        assert seq == 42

    def test_zero_raw_gives_negative_voltage(self):
        """Raw value 0 → (0 - 2048) × 0.48828125 ≈ -1000 µV."""
        packet = _pack_eeg_packet(0, [0, 0, 0, 0, 0])
        _, samples = _parse_eeg_packet(packet)
        expected = (0 - 2048) * 0.48828125
        assert samples == pytest.approx([expected] * 5, abs=1.0)

    def test_midpoint_raw_gives_zero_voltage(self):
        """Raw value 2048 → 0 µV."""
        packet = _pack_eeg_packet(0, [2048, 2048, 2048, 2048, 2048])
        _, samples = _parse_eeg_packet(packet)
        assert samples == pytest.approx([0.0] * 5, abs=0.5)

    def test_max_raw_gives_positive_voltage(self):
        """Raw value 4095 → (4095 - 2048) × 0.48828125 ≈ +1000 µV."""
        packet = _pack_eeg_packet(0, [4095, 4095, 4095, 4095, 4095])
        _, samples = _parse_eeg_packet(packet)
        expected = (4095 - 2048) * 0.48828125
        assert samples == pytest.approx([expected] * 5, abs=1.0)

    def test_returns_five_samples(self):
        packet = _pack_eeg_packet(1, [1000, 2000, 3000, 2048, 500])
        _, samples = _parse_eeg_packet(packet)
        assert len(samples) == 5

    def test_output_is_float32_array(self):
        packet = _pack_eeg_packet(0, [2048, 2048, 2048, 2048, 2048])
        _, samples = _parse_eeg_packet(packet)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
