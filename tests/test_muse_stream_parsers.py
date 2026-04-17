"""Testy parserów strumieni Muse z użyciem syntetycznych pakietów BLE."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from src.muse_connector import (
    _parse_battery_payload,
    _parse_eeg_packet,
    _parse_imu_packet,
    _parse_ppg_packet,
)


def _pack_eeg_packet(sequence: int, samples: list[int]) -> bytes:
    """Buduje pakiet EEG z 5 próbkami 12-bit zgodnie z protokołem Muse."""
    assert len(samples) == 5
    header = struct.pack(">H", sequence & 0xFFFF)
    bits = 0
    for sample in samples:
        bits = (bits << 12) | (sample & 0xFFF)
    bits <<= 4
    return header + bits.to_bytes(8, "big")


def test_parse_eeg_packet_sequence_and_shape() -> None:
    """Weryfikuje dekodowanie sekwencji i liczby próbek EEG."""
    packet = _pack_eeg_packet(77, [2048, 2048, 2048, 2048, 2048])
    sequence, samples = _parse_eeg_packet(packet)
    assert sequence == 77
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5,)


def test_parse_imu_packet_returns_xyz_matrix() -> None:
    """Sprawdza, że parser IMU zwraca macierz N×3 i skaluje dane."""
    raw_values = [5, 10, -10, 20, -20, 30, 40, 50, -60]
    packet = struct.pack(">10h", 123, *raw_values)
    sequence, samples = _parse_imu_packet(packet, scale=0.5)
    assert sequence == 123
    assert samples.shape == (3, 3)
    assert samples[0].tolist() == pytest.approx([2.5, 5.0, -5.0])


def test_parse_ppg_packet_reads_24bit_values() -> None:
    """Weryfikuje parser PPG dla próbek 24-bit BE."""
    sequence = 45
    values = [100, 16_777_215, 50_000]
    payload = b"".join(v.to_bytes(3, "big") for v in values)
    packet = struct.pack(">H", sequence) + payload
    parsed_sequence, samples = _parse_ppg_packet(packet)
    assert parsed_sequence == sequence
    assert samples.tolist() == pytest.approx([100.0, 16_777_215.0, 50_000.0])


def test_parse_battery_payload_clamps_to_range() -> None:
    """Poziom baterii musi mieścić się w zakresie 0-100%."""
    assert _parse_battery_payload(bytes([85])) == 85
    assert _parse_battery_payload(bytes([255])) == 100
