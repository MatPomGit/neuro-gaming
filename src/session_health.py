"""Monitor stanu sesji EEG i logika bezpiecznego fallbacku."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SessionHealth:
    """Skonsolidowany stan jakości sesji używany przez UI i fallback."""

    battery_level: int | None
    signal_quality: float
    reconnect_count: int
    dropout_rate: float


@dataclass(slots=True)
class SessionHealthThresholds:
    """Konfigurowalne progi alarmowe dla monitoringu sesji."""

    low_battery_warning: int = 25
    low_signal_pause: float = 0.28
    reconnect_pause_count: int = 4
    dropout_warning: float = 5.0
    dropout_critical: float = 12.0


@dataclass(slots=True)
class SessionHealthAssessment:
    """Wynik pojedynczej oceny zdrowia sesji."""

    level: str
    status_label: str
    suggestion: str
    warnings: list[str]
    safe_pause: bool
    switch_to_keyboard_mode: bool


class SessionHealthMonitor:
    """Ocena ryzyka sesji EEG i rekomendacja bezpiecznych akcji.

    Klasa agreguje najważniejsze wskaźniki jakości i zwraca decyzję:
    * czy tylko ostrzegać,
    * czy aktywować tryb safe pause,
    * czy wymusić fallback do sterowania klawiaturą.
    """

    def __init__(self, thresholds: SessionHealthThresholds | None = None) -> None:
        self.thresholds = thresholds or SessionHealthThresholds()

    def evaluate(self, health: SessionHealth) -> SessionHealthAssessment:
        """Analizuje metryki sesji i buduje opis stanu dla UI."""
        warnings: list[str] = []
        suggestions: list[str] = []
        safe_pause = False
        switch_to_keyboard_mode = False

        # Niski poziom baterii sygnalizujemy ostrzeżeniem, ale bez blokowania sterowania.
        if health.battery_level is not None and health.battery_level <= self.thresholds.low_battery_warning:
            warnings.append("Niski poziom baterii opaski EEG.")
            suggestions.append("Podłącz ładowanie lub zaplanuj krótką przerwę.")

        # Niska jakość kontaktu elektrod zwiększa ryzyko błędnych komend EEG.
        if health.signal_quality < self.thresholds.low_signal_pause:
            safe_pause = True
            warnings.append("Słaby kontakt elektrod (niska jakość sygnału).")
            suggestions.append("Popraw kontakt elektrod i sprawdź ułożenie opaski.")

        # Liczne reconnecty to sygnał niestabilności transmisji BLE.
        if health.reconnect_count >= self.thresholds.reconnect_pause_count:
            safe_pause = True
            warnings.append("Częste reconnecty podczas sesji.")
            suggestions.append("Ogranicz zakłócenia Bluetooth i przybliż urządzenie.")

        # Rosnący dropout jest ostrzeżeniem, a próg krytyczny wymusza keyboard mode.
        if health.dropout_rate >= self.thresholds.dropout_warning:
            warnings.append(f"Wysoki dropout EEG: {health.dropout_rate:.1f}%.")
            suggestions.append("Pozostań nieruchomo i popraw dopasowanie opaski.")
        if health.dropout_rate >= self.thresholds.dropout_critical:
            safe_pause = True
            switch_to_keyboard_mode = True
            warnings.append("Krytyczny dropout – przejście do keyboard mode.")

        if switch_to_keyboard_mode:
            level = "CRITICAL"
            status_label = "KRYTYCZNY"
        elif safe_pause:
            level = "RISK"
            status_label = "RYZYKO"
        elif warnings:
            level = "WARNING"
            status_label = "OSTRZEŻENIE"
        else:
            level = "OK"
            status_label = "OK"
            suggestions.append("Sygnał stabilny. Kontynuuj sterowanie EEG.")

        suggestion_text = " ".join(dict.fromkeys(suggestions))
        return SessionHealthAssessment(
            level=level,
            status_label=status_label,
            suggestion=suggestion_text,
            warnings=warnings,
            safe_pause=safe_pause,
            switch_to_keyboard_mode=switch_to_keyboard_mode,
        )
