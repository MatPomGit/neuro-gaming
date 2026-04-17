"""
NeuroGaming – main Kivy application entry point.

Architecture overview
---------------------
* **MuseConnector** (background thread / asyncio loop)
  Handles BLE scanning and streaming.  Calls ``SignalProcessor.add_samples``
  directly and is safe to call from any thread.

* **SignalProcessor**
  Maintains rolling EEG buffers and computes alpha/beta band powers.
  ``get_direction()`` and ``get_metrics()`` are polled by the Kivy
  clock every 100 ms.

* **GameController**
  Applies hysteresis and maps direction constants to keyboard keys.
  Also handles manual keyboard / touch input as a fallback.

* **Kivy UI** (main thread)
  Rendered via ``neuro_gaming.kv``.  Two screens:
  - **ScanScreen**  – scan for and connect to Muse devices.
  - **GameScreen**  – real-time EEG visualisation + directional display.
"""

import logging
import math
import os
import argparse
import struct
import tempfile
import time
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    StringProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.switch import Switch
from kivy.uix.textinput import TextInput

from src.game_controller import GameController
from src.muse_connector import MuseConnector
from src.settings import AppSettings, load_settings, save_settings
from src.single_instance import acquire_lock, release_lock
from src.session_recorder import SessionRecorder
from src.session_replay import ReplayConnector, SessionReplay
from src.session_health import SessionHealth, SessionHealthMonitor
from src.signal_processor import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_LEFT,
    DIRECTION_NONE,
    DIRECTION_RIGHT,
    SignalProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

__version__ = "1.2.0"

# Dot movement speed in pixels per second
DOT_SPEED = 200

# Human-readable direction labels shown in the status bar
_DIR_LABELS: dict[str, str] = {
    DIRECTION_NONE:     "AWAITING SIGNAL...",
    DIRECTION_FORWARD:  "Forward",
    DIRECTION_BACKWARD: "Backward",
    DIRECTION_LEFT:     "Left",
    DIRECTION_RIGHT:    "Right",
}


class _UILogHandler(logging.Handler):
    """Mirror logger output into the application's console panel."""

    def __init__(self, app: "NeuroGamingApp") -> None:
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self._app.add_console_line(message)

# ──────────────────────────────────────────────────────────────────────────────
# Kivy layout (KV string) – loaded once at startup
# ──────────────────────────────────────────────────────────────────────────────

KV_FILE = os.path.join(os.path.dirname(__file__), "neuro_gaming.kv")


# ──────────────────────────────────────────────────────────────────────────────
# Screen: Scan / Connect
# ──────────────────────────────────────────────────────────────────────────────

class ScanScreen(Screen):
    status_text = StringProperty("Press 'Scan' to search for Muse devices.")
    device_names = ListProperty([])
    found_devices_count = NumericProperty(0)
    is_scanning = BooleanProperty(False)
    fsm_state = StringProperty("IDLE")
    scan_pulse = NumericProperty(1.0)
    fsm_color = ListProperty([0.6, 0.6, 0.7, 1.0])

    _pulse_anim = None

    def on_enter(self, *args):
        app = App.get_running_app()
        app.connector.start()
        app.connector.set_status_callback(self._update_status)
        
        # Auto-scan if no device is connected and we are not already scanning
        if not app.connector.is_connected and not self.is_scanning:
            Clock.schedule_once(lambda dt: self.scan(), 0.5)

    def scan(self) -> None:
        if self.is_scanning:
            return
        app = App.get_running_app()
        self.status_text = "Scanning…"
        self.fsm_state = "SCANNING"
        self.device_names = []
        self.is_scanning = True
        # Pulse the scan button while scanning
        self._pulse_anim = (
            Animation(scan_pulse=0.35, duration=0.45) +
            Animation(scan_pulse=1.0, duration=0.45)
        )
        self._pulse_anim.repeat = True
        self._pulse_anim.start(self)
        # Run the blocking BLE scan on a background thread so the UI stays responsive
        threading.Thread(target=self._do_scan, args=(app,), daemon=True).start()

    def _do_scan(self, app) -> None:  # noqa: ANN001
        try:
            app.connector.scan(timeout=5.0)
            names = [
                f"{d.name}  ({d.address})" for d in app.connector.devices
            ]
            device_names = names if names else ["No Muse devices found."]
            status = (
                f"Found {len(app.connector.devices)} device(s)."
                if app.connector.devices
                else "No Muse devices found."
            )
        except Exception as exc:
            device_names = []
            status = f"Scan error: {exc}"
        Clock.schedule_once(lambda dt: self._scan_done(device_names, status))

    def _scan_done(self, device_names: list, status: str) -> None:
        if self._pulse_anim:
            self._pulse_anim.stop(self)
            self._pulse_anim = None
        self.device_names = device_names
        # Liczymy tylko realnie wykryte urządzenia, bez placeholdera.
        self.found_devices_count = len(App.get_running_app().connector.devices)
        self.status_text = status
        self.is_scanning = False
        self.fsm_state = "FOUND" if device_names and device_names[0] != "No Muse devices found." else "IDLE"
        app = App.get_running_app()
        if app:
            if device_names and device_names[0] != "No Muse devices found.":
                app.play_found_sound()
                # Automatyczne podłączenie, jeśli znaleziono dokładnie jedno urządzenie Muse
                if len(app.connector.devices) == 1:
                    self._update_status("SINGLE DEVICE DETECTED. AUTO-CONNECTING...")
                    self.connect_device(0)
                elif len(app.connector.devices) > 1:
                    self._update_status("MULTIPLE DEVICES DETECTED. PLEASE SELECT MANUALLY.")
        self.scan_pulse = 1.0

    def connect_device(self, index: int) -> None:
        app = App.get_running_app()
        devices = app.connector.devices
        if index < 0 or index >= len(devices):
            self.status_text = "Invalid device selection."
            return
        
        device = devices[index]
        self.status_text = f"Connecting to {device.name}…"
        self.fsm_state = "CONNECTING"
        self.is_scanning = True  # Re-use flag to disable buttons during connect
        
        # Run connection in background thread to avoid UI freeze
        threading.Thread(
            target=self._do_connect, 
            args=(app, device), 
            daemon=True
        ).start()

    def _do_connect(self, app, device):
        try:
            # MuseConnector.connect expects a BLEDevice object based on its signature, 
            # but inside it uses device.address for Windows stability.
            app.connector.connect(device)
            Clock.schedule_once(lambda dt: self._connect_done(True))
        except Exception as exc:
            error_msg = str(exc)
            Clock.schedule_once(lambda dt: self._connect_done(False, error_msg))

    def _connect_done(self, success: bool, error_msg: str = "") -> None:
        self.is_scanning = False
        if success:
            app = App.get_running_app()
            self.fsm_state = "CONNECTED"
            app.play_connected_sound()
            app.root.current = "game"
        else:
            self.fsm_state = "ERROR"
            self.status_text = f"CONNECTION FAILED: {error_msg.upper()}"

    def skip_to_keyboard(self) -> None:
        """Continue in keyboard-only mode (no Muse required)."""
        App.get_running_app().root.current = "game"

    def go_to_test(self) -> None:
        """Open the interactive test screen."""
        App.get_running_app().root.current = "test"

    def open_user_guide(self) -> None:
        App.get_running_app().open_user_guide()

    def _update_status(self, msg: str) -> None:
        self.status_text = msg
        upper_msg = msg.upper()
        if "SCANNING" in upper_msg:
            self.fsm_state = "SCANNING"
        elif "CONNECTING" in upper_msg:
            self.fsm_state = "CONNECTING"
        elif "CONNECTED" in upper_msg:
            self.fsm_state = "CONNECTED"
        elif "ERROR" in upper_msg:
            self.fsm_state = "ERROR"
        self._update_fsm_color()

    def on_fsm_state(self, *_args) -> None:
        # Synchronizujemy kolor statusu po każdej zmianie stanu FSM.
        self._update_fsm_color()

    def _update_fsm_color(self) -> None:
        # Kolory pomagają szybciej odczytać etap połączenia.
        palette = {
            "IDLE": [0.6, 0.6, 0.7, 1.0],
            "SCANNING": [0.25, 0.7, 1.0, 1.0],
            "FOUND": [0.35, 0.82, 0.45, 1.0],
            "CONNECTING": [0.95, 0.75, 0.2, 1.0],
            "CONNECTED": [0.2, 0.85, 0.35, 1.0],
            "ERROR": [0.95, 0.35, 0.35, 1.0],
        }
        self.fsm_color = palette.get(self.fsm_state, palette["IDLE"])

    def close_app(self) -> None:
        App.get_running_app().shutdown_app()


# ──────────────────────────────────────────────────────────────────────────────
# Screen: Game
# ──────────────────────────────────────────────────────────────────────────────

class GameScreen(Screen):
    direction = StringProperty(DIRECTION_NONE)
    status_text = StringProperty("Ready")
    connected = BooleanProperty(False)
    is_calibrating = BooleanProperty(False)
    key_mode = StringProperty("arrow")

    # EEG metrics – dict channel → {"alpha": float, "beta": float}
    metrics = DictProperty({
        "TP9":  {"alpha": 0.0, "beta": 0.0},
        "AF7":  {"alpha": 0.0, "beta": 0.0},
        "AF8":  {"alpha": 0.0, "beta": 0.0},
        "TP10": {"alpha": 0.0, "beta": 0.0},
    })

    # Band power display values (0–1 normalised for UI bars)
    alpha_left  = NumericProperty(0.0)
    alpha_right = NumericProperty(0.0)
    beta_left   = NumericProperty(0.0)
    beta_right  = NumericProperty(0.0)

    # Signal quality per channel (0–1); 1 = excellent contact
    quality_tp9  = NumericProperty(0.0)
    quality_af7  = NumericProperty(0.0)
    quality_af8  = NumericProperty(0.0)
    quality_tp10 = NumericProperty(0.0)
    battery_text = StringProperty("Battery: --")
    sensors_text = StringProperty("Sensors: --")
    device_text = StringProperty("Device: keyboard mode")
    stream_status_text = StringProperty("Streams: --")
    motion_text = StringProperty("Motion: stable")
    eeg_quality_text = StringProperty("EEG quality: --")
    session_health_text = StringProperty("Session health: --")
    session_health_suggestion = StringProperty("Suggestion: --")

    _update_event = None
    _was_connected = False
    _last_safety_signature = ""

    def on_enter(self, *args):
        app = App.get_running_app()
        self.connected = app.connector.is_connected
        self._was_connected = self.connected
        self._last_safety_signature = ""
        self.key_mode = app.controller.key_mode
        app.controller.on_direction_change = self._on_direction_change
        self._update_event = Clock.schedule_interval(self._tick, 0.1)
        Window.bind(on_key_down=self._on_key_down)
        Window.bind(on_key_up=self._on_key_up)

    def on_leave(self, *args):
        if self._update_event:
            self._update_event.cancel()
        Window.unbind(on_key_down=self._on_key_down)
        Window.unbind(on_key_up=self._on_key_up)

    # ── periodic UI update (100 ms) ────────────────────────────────────────

    def _tick(self, dt) -> None:  # noqa: ANN001
        app = App.get_running_app()
        self.connected = app.connector.is_connected
        if self.connected and not self._was_connected:
            # Warm-up po reconnect minimalizuje fałszywe komendy na starcie streamu.
            app.processor.notify_stream_reconnected()
        self._was_connected = self.connected

        session_assessment = app.session_health_monitor.evaluate(self._collect_session_health(app))
        self._apply_session_health(app, session_assessment)

        if app.connector.is_connected and not app.safe_pause_active and not app.keyboard_fallback_forced:
            # Update EEG-driven direction
            eeg_dir = app.processor.get_direction()
            app.controller.update(eeg_dir)

            # Refresh metrics display
            m = app.processor.get_metrics()
            self.metrics = {k: dict(v) for k, v in m.items()}
            self._update_bars(m)

            # Refresh per-channel signal quality
            quality_snapshot = app.processor.get_quality_snapshot()
            sq = quality_snapshot.get("channels", {})
            self.quality_tp9  = float(sq.get("TP9",  0.0))
            self.quality_af7  = float(sq.get("AF7",  0.0))
            self.quality_af8  = float(sq.get("AF8",  0.0))
            self.quality_tp10 = float(sq.get("TP10", 0.0))
            app.connector._device_state["signal_quality"] = dict(sq)
            app.connector._device_state["global_quality_score"] = float(
                quality_snapshot.get("global_score", 0.0)
            )
            app.connector._device_state["session_quality_score"] = float(
                quality_snapshot.get("session_score", 0.0)
            )
            app.connector._device_state["processor_state"] = str(
                quality_snapshot.get("state", "UNKNOWN")
            )
            app.connector._device_state["motion_artifact"] = app.processor.is_motion_artifact_active()
        elif app.safe_pause_active or app.keyboard_fallback_forced:
            # Safe pause zatrzymuje komendy EEG, ale pozostawia ręczne sterowanie.
            app.controller.update(DIRECTION_NONE)

        # Zapis próbki do rejestratora sesji (jeśli nagrywanie jest aktywne).
        app.session_recorder.record_sample(
            now_monotonic=time.monotonic(),
            direction=app.controller.current_direction,
            connected=app.connector.is_connected,
            motion_artifact=app.processor.is_motion_artifact_active(),
            metrics=app.processor.get_metrics(),
            signal_quality=app.processor.get_signal_quality(),
            confidence=app.processor.get_direction_confidence(),
            rejected_window=(app.controller.current_direction == DIRECTION_NONE and app.connector.is_connected),
        )

        self.direction = app.controller.current_direction
        if app.safe_pause_active:
            tag = "[SAFE]"
        elif app.connector.is_connected:
            tag = "[EEG]"
        else:
            tag = "[KB]"
        label = _DIR_LABELS.get(self.direction, self.direction)
        self.status_text = f"{tag}  {label}"
        self._update_device_status(app)

    def _collect_session_health(self, app) -> SessionHealth:  # noqa: ANN001
        """Buduje agregat SessionHealth z telemetrii konektora i procesora."""
        state = app.connector.device_state
        battery = state.get("battery_level")
        global_quality = float(state.get("global_quality_score", 0.0))
        reconnect_count = 0
        dropout_rate = 0.0
        # Wykorzystujemy metryki sesji konektora, jeśli są dostępne.
        metrics = getattr(app.connector, "_session_metrics", None)
        if metrics is not None:
            reconnect_count = int(getattr(metrics, "reconnect_count", 0))
            dropout_rate = float(getattr(metrics, "dropout_percent", 0.0))
        return SessionHealth(
            battery_level=battery if isinstance(battery, int) else None,
            signal_quality=global_quality,
            reconnect_count=reconnect_count,
            dropout_rate=dropout_rate,
        )

    def _apply_session_health(self, app, assessment) -> None:  # noqa: ANN001
        """Aktualizuje UI i akcje bezpieczeństwa na podstawie oceny ryzyka."""
        warnings = " | ".join(assessment.warnings) if assessment.warnings else "Brak alarmów."
        self.session_health_text = f"Session health: {assessment.status_label} ({warnings})"
        self.session_health_suggestion = f"Suggestion: {assessment.suggestion or 'Brak dodatkowych zaleceń.'}"
        app.safe_pause_active = bool(assessment.safe_pause)

        if assessment.switch_to_keyboard_mode and not app.keyboard_fallback_forced:
            app.keyboard_fallback_forced = True
            app.safe_pause_active = True
            app.controller.set_direction(DIRECTION_NONE)
            logger.warning("Critical dropout detected. Switching to keyboard mode fallback.")
            app.session_recorder.record_safety_event(
                now_monotonic=time.monotonic(),
                event_name="forced_keyboard_mode",
                payload={"reason": "critical_dropout"},
            )

        self._log_safety_state_if_changed(app, assessment)

    def _log_safety_state_if_changed(self, app, assessment) -> None:  # noqa: ANN001
        """Zapisuje zmiany stanu bezpieczeństwa bez duplikowania wpisów co tick."""
        signature = "|".join(
            [
                assessment.level,
                str(assessment.safe_pause),
                str(assessment.switch_to_keyboard_mode),
                ",".join(assessment.warnings),
            ]
        )
        if signature == self._last_safety_signature:
            return
        self._last_safety_signature = signature
        if assessment.level == "OK":
            return
        app.session_recorder.record_safety_event(
            now_monotonic=time.monotonic(),
            event_name="session_health_alert",
            payload={
                "level": assessment.level,
                "safe_pause": assessment.safe_pause,
                "warnings": list(assessment.warnings),
                "suggestion": assessment.suggestion,
            },
        )

    def _update_bars(self, m: dict) -> None:
        """Normalise band powers to 0–1 for UI progress bars."""
        def _clamp(v: float) -> float:
            return max(0.0, min(1.0, abs(v) / 5.0))

        self.alpha_left  = _clamp(m.get("AF7", {}).get("alpha", 0.0))
        self.alpha_right = _clamp(m.get("AF8", {}).get("alpha", 0.0))
        self.beta_left   = _clamp(m.get("AF7", {}).get("beta",  0.0))
        self.beta_right  = _clamp(m.get("AF8", {}).get("beta",  0.0))

    def _update_device_status(self, app) -> None:  # noqa: ANN001
        if not app.connector.is_connected:
            self.device_text = "Device: keyboard mode"
            self.battery_text = "Battery: --"
            self.sensors_text = "Sensors: --"
            self.stream_status_text = "Streams: --"
            self.motion_text = "Motion: --"
            self.eeg_quality_text = "EEG quality: --"
            self.session_health_text = "Session health: KEYBOARD MODE"
            self.session_health_suggestion = "Suggestion: Możesz sterować ręcznie klawiaturą."
            return
        state = app.connector.device_state
        battery = state.get("battery_level")
        battery_value = f"{battery}%" if isinstance(battery, int) else "n/a"
        sensors = state.get("available_sensors") or []
        sensor_summary = ", ".join(sensors) if sensors else "unknown"
        sample_rate = state.get("sample_rate_hz", 0)

        if app.keyboard_fallback_forced:
            self.device_text = (
                f"Device: {state.get('device_name', 'Muse')}  "
                f"({sample_rate} Hz) [keyboard fallback]"
            )
        else:
            self.device_text = (
                f"Device: {state.get('device_name', 'Muse')}  "
                f"({sample_rate} Hz)"
            )
        self.battery_text = f"Battery: {battery_value}"
        self.sensors_text = f"Sensors: {sensor_summary}"
        stream_activity = state.get("stream_activity", {})
        stream_bits = [
            f"{name}:{'ON' if active else 'OFF'}"
            for name, active in stream_activity.items()
        ]
        self.stream_status_text = "Streams: " + (", ".join(stream_bits) if stream_bits else "--")
        motion = bool(state.get("motion_artifact", False))
        self.motion_text = "Motion: artifact risk" if motion else "Motion: stable"
        quality = state.get("signal_quality", {})
        global_quality = float(state.get("global_quality_score", 0.0))
        session_quality = float(state.get("session_quality_score", 0.0))
        if quality:
            processor_state = state.get("processor_state", "UNKNOWN")
            self.eeg_quality_text = (
                f"EEG quality: {int(global_quality * 100)}% "
                f"(session {int(session_quality * 100)}%, state {processor_state})"
            )
        else:
            self.eeg_quality_text = "EEG quality: --"

    # ── direction callback ─────────────────────────────────────────────────

    def _on_direction_change(self, new_direction: str) -> None:
        self.direction = new_direction

    # ── keyboard input ─────────────────────────────────────────────────────

    def _on_key_down(self, window, key, scancode, codepoint, modifiers) -> None:  # noqa: ANN001
        key_name = _keycode_to_name(key, codepoint)
        App.get_running_app().session_recorder.record_control_event(
            now_monotonic=time.monotonic(),
            event_name="key_down",
            payload={"key": key_name},
        )
        App.get_running_app().controller.handle_key_down(key_name)

    def _on_key_up(self, window, key, *args) -> None:  # noqa: ANN001
        key_name = _keycode_to_name(key, "")
        App.get_running_app().session_recorder.record_control_event(
            now_monotonic=time.monotonic(),
            event_name="key_up",
            payload={"key": key_name},
        )
        App.get_running_app().controller.handle_key_up(key_name)

    # ── button handlers ────────────────────────────────────────────────────

    def toggle_calibration(self) -> None:
        """Navigate to the dedicated calibration wizard screen."""
        App.get_running_app().root.current = "calibration"

    def toggle_key_mode(self) -> None:
        app = App.get_running_app()
        app.controller.key_mode = (
            "wasd" if app.controller.key_mode == "arrow" else "arrow"
        )
        app.settings.key_mode = app.controller.key_mode
        app.persist_settings()
        self.key_mode = app.controller.key_mode

    def open_settings_popup(self) -> None:
        app = App.get_running_app()
        settings = app.settings
        root = BoxLayout(orientation="vertical", spacing=8, padding=12)
        form = BoxLayout(orientation="vertical", spacing=6, size_hint_y=1)

        beta_input = TextInput(text=str(settings.beta_threshold), multiline=False)
        alpha_input = TextInput(text=str(settings.alpha_threshold), multiline=False)
        asym_input = TextInput(text=str(settings.asym_factor), multiline=False)
        hyst_input = TextInput(text=str(settings.hysteresis_count), multiline=False)
        active_profile_input = TextInput(text=settings.active_profile_id, multiline=False)
        default_profile_input = TextInput(text=settings.default_profile_id, multiline=False)
        forwarding_switch = Switch(active=settings.forwarding_enabled)
        eeg_switch = Switch(active=settings.stream_eeg_enabled)
        accel_switch = Switch(active=settings.stream_accelerometer_enabled)
        gyro_switch = Switch(active=settings.stream_gyroscope_enabled)
        ppg_switch = Switch(active=settings.stream_ppg_enabled)
        battery_switch = Switch(active=settings.stream_battery_enabled)

        for label_text, widget in (
            ("Beta threshold", beta_input),
            ("Alpha threshold", alpha_input),
            ("Asymmetry factor", asym_input),
            ("Hysteresis count", hyst_input),
            ("Active profile ID", active_profile_input),
            ("Default profile ID", default_profile_input),
        ):
            row = BoxLayout(size_hint_y=None, height="38dp")
            row.add_widget(Label(text=label_text, size_hint_x=0.55))
            row.add_widget(widget)
            form.add_widget(row)

        switch_row = BoxLayout(size_hint_y=None, height="38dp")
        switch_row.add_widget(Label(text="OS forwarding", size_hint_x=0.55))
        switch_row.add_widget(forwarding_switch)
        form.add_widget(switch_row)
        for row_label, row_switch in (
            ("Stream EEG", eeg_switch),
            ("Stream accelerometer", accel_switch),
            ("Stream gyroscope", gyro_switch),
            ("Stream PPG", ppg_switch),
            ("Stream battery", battery_switch),
        ):
            row = BoxLayout(size_hint_y=None, height="38dp")
            row.add_widget(Label(text=row_label, size_hint_x=0.55))
            row.add_widget(row_switch)
            form.add_widget(row)

        actions = BoxLayout(size_hint_y=None, height="44dp", spacing=8)
        popup = Popup(title="Settings", content=root, size_hint=(0.8, 0.75))

        save_btn = Button(text="Save")
        reset_btn = Button(text="Restore defaults")
        close_btn = Button(text="Close")

        save_btn.bind(
            on_release=lambda *_: self._save_settings_from_popup(
                popup,
                beta_input.text,
                alpha_input.text,
                asym_input.text,
                hyst_input.text,
                forwarding_switch.active,
                eeg_switch.active,
                accel_switch.active,
                gyro_switch.active,
                ppg_switch.active,
                battery_switch.active,
                active_profile_input.text,
                default_profile_input.text,
            )
        )
        reset_btn.bind(on_release=lambda *_: self._restore_defaults_from_popup(popup))
        close_btn.bind(on_release=lambda *_: popup.dismiss())
        actions.add_widget(save_btn)
        actions.add_widget(reset_btn)
        actions.add_widget(close_btn)

        root.add_widget(form)
        root.add_widget(actions)
        popup.open()

    def _save_settings_from_popup(
        self,
        popup: Popup,
        beta: str,
        alpha: str,
        asym: str,
        hysteresis: str,
        forwarding_enabled: bool,
        stream_eeg_enabled: bool,
        stream_accelerometer_enabled: bool,
        stream_gyroscope_enabled: bool,
        stream_ppg_enabled: bool,
        stream_battery_enabled: bool,
        active_profile_id: str,
        default_profile_id: str,
    ) -> None:
        app = App.get_running_app()
        try:
            candidate = AppSettings(
                beta_threshold=float(beta),
                alpha_threshold=float(alpha),
                asym_factor=float(asym),
                hysteresis_count=int(hysteresis),
                key_mode=app.settings.key_mode,
                forwarding_enabled=bool(forwarding_enabled),
                debug_logging=app.settings.debug_logging,
                debug_eeg_file=app.settings.debug_eeg_file,
                debug_logging_enabled=app.settings.debug_logging_enabled,
                stream_eeg_enabled=bool(stream_eeg_enabled),
                stream_accelerometer_enabled=bool(stream_accelerometer_enabled),
                stream_gyroscope_enabled=bool(stream_gyroscope_enabled),
                stream_ppg_enabled=bool(stream_ppg_enabled),
                stream_battery_enabled=bool(stream_battery_enabled),
                active_profile_id=active_profile_id.strip() or "default",
                default_profile_id=default_profile_id.strip() or "default",
            )
            candidate.validate()
        except (TypeError, ValueError) as exc:
            self.status_text = f"[SETTINGS] invalid values: {exc}"
            return

        app.settings = candidate
        app.apply_settings()
        app.persist_settings()
        self.key_mode = app.controller.key_mode
        popup.dismiss()

    def _restore_defaults_from_popup(self, popup: Popup) -> None:
        app = App.get_running_app()
        app.settings = AppSettings()
        app.apply_settings()
        app.persist_settings()
        self.key_mode = app.controller.key_mode
        self.status_text = "[SETTINGS] restored defaults"
        popup.dismiss()

    def disconnect(self) -> None:
        app = App.get_running_app()
        app.connector.disconnect()
        app.processor.reset()
        app.controller.reset()
        app.safe_pause_active = False
        app.keyboard_fallback_forced = False
        app.root.current = "scan"

    def close_app(self) -> None:
        App.get_running_app().shutdown_app()

    def open_user_guide(self) -> None:
        App.get_running_app().open_user_guide()


# ──────────────────────────────────────────────────────────────────────────────
# Screen: Test
# ──────────────────────────────────────────────────────────────────────────────

class TestScreen(Screen):
    """Interactive test screen – control a blue dot in 2-D screen space.

    Controls
    --------
    * **W / ↑** – move dot up
    * **S / ↓** – move dot down
    * **A / ←** – move dot left
    * **D / →** – move dot right
    * **Left mouse button (LMB)** – dot briefly flashes green (action pulse)
    * **Right mouse button (RMB)** – dot resets to the centre of the screen
    """

    dot_x       = NumericProperty(0.0)
    dot_y       = NumericProperty(0.0)
    dot_radius  = NumericProperty(20.0)
    dot_color   = ListProperty([0.2, 0.5, 1.0, 1.0])   # blue
    left_active  = BooleanProperty(False)
    right_active = BooleanProperty(False)
    status_text  = StringProperty(
        "W/S/↑/↓ – up/down   A/D/←/→ – left/right   LMB – action   RMB – reset"
    )
    recorder_status = StringProperty("Recorder: idle")
    replay_status = StringProperty("Replay: waiting")
    recording_active = BooleanProperty(False)

    _update_event = None
    _recorder_event = None
    _replay_event = None
    _replay_data: list[dict] = []
    _replay_index = 0

    def on_enter(self, *args) -> None:
        self._keys_held: set[str] = set()
        Clock.schedule_once(self._init_dot_position, 0)
        Window.bind(on_key_down=self._on_key_down)
        Window.bind(on_key_up=self._on_key_up)
        self._update_event = Clock.schedule_interval(self._tick, 1.0 / 60)
        self._recorder_event = Clock.schedule_interval(self._refresh_recorder_status, 0.25)

    def on_leave(self, *args) -> None:
        if self._update_event:
            self._update_event.cancel()
            self._update_event = None
        if self._recorder_event:
            self._recorder_event.cancel()
            self._recorder_event = None
        if self._replay_event:
            self._replay_event.cancel()
            self._replay_event = None
        Animation.cancel_all(self)
        Window.unbind(on_key_down=self._on_key_down)
        Window.unbind(on_key_up=self._on_key_up)
        self._keys_held = set()
        # Reset controller mouse state
        App.get_running_app().controller.reset()

    # ── initialisation ─────────────────────────────────────────────────────

    def _init_dot_position(self, dt) -> None:  # noqa: ANN001
        self.dot_x = Window.width  / 2.0
        self.dot_y = Window.height / 2.0

    # ── movement tick ──────────────────────────────────────────────────────

    def _tick(self, dt) -> None:  # noqa: ANN001
        speed = DOT_SPEED * dt
        r = self.dot_radius
        keys = self._keys_held
        if "up" in keys or "w" in keys:
            self.dot_y = min(self.dot_y + speed, Window.height - r)
        if "down" in keys or "s" in keys:
            self.dot_y = max(self.dot_y - speed, r)
        if "left" in keys or "a" in keys:
            self.dot_x = max(self.dot_x - speed, r)
        if "right" in keys or "d" in keys:
            self.dot_x = min(self.dot_x + speed, Window.width - r)

    # ── keyboard input ─────────────────────────────────────────────────────

    def _on_key_down(self, window, key, scancode, codepoint, modifiers) -> None:  # noqa: ANN001
        key_name = _keycode_to_name(key, codepoint)
        if key_name in ("up", "down", "left", "right", "w", "s", "a", "d"):
            self._keys_held.add(key_name)

    def _on_key_up(self, window, key, *args) -> None:  # noqa: ANN001
        key_name = _keycode_to_name(key, "")
        self._keys_held.discard(key_name)

    # ── mouse input ────────────────────────────────────────────────────────

    def on_touch_down(self, touch):
        # Propagate first so child buttons (e.g. Back) handle their own clicks
        if super().on_touch_down(touch):
            return True
        button = getattr(touch, "button", None)
        if button == "left":
            self._handle_left_click(touch)
        elif button == "right":
            self._handle_right_click(touch)
        return True

    def on_touch_up(self, touch):
        button = getattr(touch, "button", None)
        if button == "left":
            self.left_active = False
            App.get_running_app().controller.handle_mouse_up("left")
        elif button == "right":
            self.right_active = False
            App.get_running_app().controller.handle_mouse_up("right")
        return super().on_touch_up(touch)

    # ── mouse effects ──────────────────────────────────────────────────────

    def _handle_left_click(self, touch) -> None:
        """Left-click: dot pulses and flashes green (action pulse)."""
        self.left_active = True
        App.get_running_app().controller.handle_mouse_down("left")
        Animation.cancel_all(self, 'dot_color', 'dot_radius')
        anim = (
            Animation(dot_color=[0.2, 1.0, 0.4, 1.0], dot_radius=28.0,
                      duration=0.12, t='out_quad') +
            Animation(dot_color=[0.2, 0.5, 1.0, 1.0], dot_radius=20.0,
                      duration=0.3, t='in_out_quad')
        )
        anim.start(self)
        self.status_text = f"LMB – action at ({int(touch.x)}, {int(touch.y)})"

    def _handle_right_click(self, touch) -> None:
        """Right-click: animate dot smoothly back to screen centre."""
        self.right_active = True
        App.get_running_app().controller.handle_mouse_down("right")
        Animation.cancel_all(self, 'dot_x', 'dot_y')
        Animation(
            dot_x=Window.width / 2.0,
            dot_y=Window.height / 2.0,
            duration=0.4,
            t='out_cubic',
        ).start(self)
        self.status_text = "RMB – dot reset to centre"

    def _refresh_recorder_status(self, _dt: float) -> None:
        """Odświeża podsumowanie rejestratora pokazywane na ekranie."""
        info = App.get_running_app().session_recorder.snapshot()
        self.recording_active = bool(info.get("active", False))
        self.recorder_status = (
            f"Recorder: {'ON' if self.recording_active else 'OFF'} | "
            f"samples={info.get('samples', 0)} | "
            f"duration={info.get('duration', 0.0):.1f}s"
        )

    def toggle_recording(self) -> None:
        """Przełącza rejestrowanie sesji EEG do pamięci."""
        recorder = App.get_running_app().session_recorder
        if recorder.active:
            count = recorder.stop()
            self.replay_status = f"Replay: recorded {count} samples"
            return
        app = App.get_running_app()
        state = app.connector.device_state
        session_name = recorder.start(
            now_monotonic=time.monotonic(),
            metadata={
                "device_model": state.get("device_name", "Unknown"),
                "active_channels": state.get("available_sensors", []),
                "threshold_config": {
                    "beta_threshold": app.processor.beta_threshold,
                    "alpha_threshold": app.processor.alpha_threshold,
                    "asym_factor": app.processor.asym_factor,
                    "min_confidence": app.processor.min_confidence,
                },
                "app_version": app.version,
                "sample_rate_hz": state.get("sample_rate_hz", 256),
            },
        )
        self.replay_status = f"Replay: recording {session_name}"

    def clear_recording(self) -> None:
        """Czyści próbki sesji i zatrzymuje aktywny replay."""
        recorder = App.get_running_app().session_recorder
        recorder.stop()
        recorder.clear()
        if self._replay_event:
            self._replay_event.cancel()
            self._replay_event = None
        self._replay_data = []
        self._replay_index = 0
        self.replay_status = "Replay: cleared"

    def export_recording_csv(self) -> None:
        """Eksportuje nagraną sesję do pliku CSV/JSONL i raportu."""
        recorder = App.get_running_app().session_recorder
        info = recorder.snapshot()
        if info.get("samples", 0) == 0:
            self.replay_status = "Replay: nothing to export"
            return
        session_name = info.get("session_name") or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
        directory = Path("sessions")
        csv_path = recorder.export_csv(directory / f"{session_name}.csv")
        csv_extended_path = recorder.export_csv_extended(directory / f"{session_name}.extended.csv")
        session_path = recorder.export_session(directory / f"{session_name}.session.jsonl")
        report_path = recorder.export_report(directory / f"{session_name}.report.json")
        self.replay_status = (
            "Replay: exported "
            f"{csv_path.name}, {csv_extended_path.name}, {session_path.name}, {report_path.name}"
        )

    def start_replay(self) -> None:
        """Uruchamia odtwarzanie ostatnio nagranej sesji sterowania."""
        recorder = App.get_running_app().session_recorder
        self._replay_data = recorder.replay_data()
        if not self._replay_data:
            self.replay_status = "Replay: no data"
            return
        if self._replay_event:
            self._replay_event.cancel()
        self._replay_index = 0
        self.replay_status = "Replay: running"
        self._replay_event = Clock.schedule_interval(self._replay_tick, 1.0 / 30.0)

    def _replay_tick(self, _dt: float) -> None:
        """Wykonuje jedną klatkę replayu poprzez przesunięcie kropki."""
        if self._replay_index >= len(self._replay_data):
            if self._replay_event:
                self._replay_event.cancel()
                self._replay_event = None
            self.replay_status = "Replay: finished"
            return
        item = self._replay_data[self._replay_index]
        direction = item.get("direction", "NONE")
        step = 6.0
        r = self.dot_radius
        if direction == "FORWARD":
            self.dot_y = min(self.dot_y + step, Window.height - r)
        elif direction == "BACKWARD":
            self.dot_y = max(self.dot_y - step, r)
        elif direction == "LEFT":
            self.dot_x = max(self.dot_x - step, r)
        elif direction == "RIGHT":
            self.dot_x = min(self.dot_x + step, Window.width - r)
        self._replay_index += 1

    # ── navigation ─────────────────────────────────────────────────────────

    def go_back(self) -> None:
        App.get_running_app().root.current = "scan"


# ──────────────────────────────────────────────────────────────────────────────
# Screen: Calibration
# ──────────────────────────────────────────────────────────────────────────────

# Total recording time for a single calibration run (seconds)
CALIBRATION_STAGE_DURATION = 8

# Step descriptions shown in the calibration wizard
_CALIB_STEPS = [
    "1. Etap bazowy: usiądź spokojnie i patrz przed siebie.",
    "2. Etap koncentracji: skup się na prostym zadaniu mentalnym.",
    "3. Etap relaksacji: rozluźnij oddech i mięśnie twarzy.",
    "4. Walidacja: sprawdzamy długość i jakość sygnału.",
    "5. Zapis profilu: aktywujemy nowy profil kalibracji.",
]


class CalibrationScreen(Screen):
    """Step-by-step calibration wizard.

    Guides the user through fitting the headset, checking signal quality,
    and recording an EEG baseline used to normalise subsequent band powers.
    """

    # Current active step index (0-based, matches _CALIB_STEPS)
    current_step = NumericProperty(0)

    # Calibration state
    is_calibrating = BooleanProperty(False)
    calibration_done = BooleanProperty(False)

    # Countdown timer value (seconds remaining)
    timer_value = NumericProperty(CALIBRATION_STAGE_DURATION)

    # Human-readable status displayed below the step list
    status_text = StringProperty("Naciśnij START, aby rozpocząć kalibrację etapową.")

    # Signal quality per channel (0–1)
    quality_tp9  = NumericProperty(0.0)
    quality_af7  = NumericProperty(0.0)
    quality_af8  = NumericProperty(0.0)
    quality_tp10 = NumericProperty(0.0)

    # Step label texts (read-only; bound from the constant list)
    steps = ListProperty(_CALIB_STEPS)

    # Expose duration so the KV timer bar can reference it without a magic number
    calibration_duration = NumericProperty(CALIBRATION_STAGE_DURATION)
    stage_name = StringProperty("Etap bazowy")

    _tick_event = None
    _elapsed = 0.0

    def on_enter(self, *args) -> None:
        self._tick_event = Clock.schedule_interval(self._tick, 0.25)
        # If arriving from game screen with active calibration, stop it cleanly
        if self.is_calibrating:
            self._finish_calibration()

    def on_leave(self, *args) -> None:
        if self._tick_event:
            self._tick_event.cancel()
            self._tick_event = None
        # Cancel any ongoing calibration when navigating away
        if self.is_calibrating:
            self._finish_calibration()

    # ── periodic UI update (250 ms) ────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        app = App.get_running_app()
        sq = app.processor.get_signal_quality()
        self.quality_tp9  = sq.get("TP9",  0.0)
        self.quality_af7  = sq.get("AF7",  0.0)
        self.quality_af8  = sq.get("AF8",  0.0)
        self.quality_tp10 = sq.get("TP10", 0.0)

        # Advance step highlight based on quality / state
        if not self.is_calibrating and not self.calibration_done:
            self._auto_advance_step()

        # Countdown while calibrating
        if self.is_calibrating:
            self._elapsed += dt
            remaining = max(0.0, CALIBRATION_STAGE_DURATION - self._elapsed)
            self.timer_value = remaining
            self.status_text = f"{self.stage_name}: pozostało {int(remaining) + 1}s."
            if remaining <= 0:
                self._advance_or_finish()

    def _auto_advance_step(self) -> None:
        """Advance the highlighted step automatically as conditions are met."""
        if self.current_step < 0:
            self.current_step = 0

    # ── button handlers ────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Begin recording a new EEG baseline."""
        app = App.get_running_app()
        self.is_calibrating = True
        self.calibration_done = False
        self._elapsed = 0.0
        self.current_step = 0
        self.stage_name = "Etap bazowy (spoczynek)"
        self.timer_value = CALIBRATION_STAGE_DURATION
        self.status_text = "Rozpoczęto etap bazowy."
        app.processor.start_calibration("baseline_rest")
        logger.info("Calibration wizard: started")

    def _advance_or_finish(self) -> None:
        """Przechodzi do kolejnego etapu albo finalizuje profil."""
        app = App.get_running_app()
        if self.current_step == 0:
            app.processor.stop_calibration()
            self.current_step = 1
            self._elapsed = 0.0
            self.timer_value = CALIBRATION_STAGE_DURATION
            self.stage_name = "Etap koncentracji"
            self.status_text = "Etap koncentracji rozpoczęty."
            app.processor.start_calibration("focus_task")
            return
        if self.current_step == 1:
            app.processor.stop_calibration()
            self.current_step = 2
            self._elapsed = 0.0
            self.timer_value = CALIBRATION_STAGE_DURATION
            self.stage_name = "Etap relaksacji"
            self.status_text = "Etap relaksacji rozpoczęty."
            app.processor.start_calibration("relax_task")
            return
        self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Kończy kalibrację i zapisuje profil użytkownika."""
        app = App.get_running_app()
        app.processor.stop_calibration()
        self.is_calibrating = False
        self.timer_value = 0.0
        state = app.connector.device_state
        device_address = state.get("address") or "keyboard-mode"
        firmware = state.get("firmware_version")
        profile_id = f"{device_address.replace(':', '').lower()}_{int(time.time())}"
        try:
            profile = app.processor.finalize_calibration_profile(
                profile_id=profile_id,
                user_id="default-user",
                device_address=device_address,
                firmware_if_available=firmware,
            )
            app.settings.active_profile_id = profile.profile_id
            app.persist_settings()
            self.calibration_done = True
            self.current_step = 4
            self.status_text = "Kalibracja zakończona pomyślnie. Profil został zapisany."
            logger.info("Calibration wizard: finished with profile %s", profile.profile_id)
        except ValueError as exc:
            self.calibration_done = False
            self.current_step = 3
            self.status_text = f"Kalibracja odrzucona: {exc}"
            logger.warning("Calibration rejected: %s", exc)

    def go_back(self) -> None:
        """Return to the previous screen."""
        App.get_running_app().root.current = "game"


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

class NeuroGamingApp(App):
    """Root Kivy application."""

    version = StringProperty(__version__)
    console_output = StringProperty("")

    def __init__(self, **kwargs):
        self._instance_lock = kwargs.pop("instance_lock", None)
        self._replay_path = kwargs.pop("replay_path", None)
        super().__init__(**kwargs)
        self._log_lines: list[str] = []
        self._max_log_lines = 300
        self._log_handler = _UILogHandler(self)
        logging.getLogger().addHandler(self._log_handler)
        self._sound_paths: dict[str, str] = {}
        self._sounds: dict[str, object] = {}
        self.settings = AppSettings()
        # Flagi bezpieczeństwa używane przez ekran sterowania.
        self.safe_pause_active = False
        self.keyboard_fallback_forced = False

    def build(self):
        try:
            self.settings = load_settings()
        except Exception as exc:
            logger.warning("Failed to load settings, using defaults: %s", exc)
            self.settings = AppSettings()
        if self.settings.debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        # Core objects shared across screens
        self.processor = SignalProcessor(settings=self.settings)
        self.controller = GameController(settings=self.settings)
        # Monitor zdrowia sesji agreguje sygnały jakości i podpowiada fallback.
        self.session_health_monitor = SessionHealthMonitor()
        # Rejestrator sesji diagnostycznej używany przez ekran testowy i eksport CSV.
        self.session_recorder = SessionRecorder()

        def _on_eeg(channel: str, samples) -> None:  # noqa: ANN001
            self.processor.add_samples(channel, samples)
            self.session_recorder.record_eeg_frame(
                now_monotonic=time.monotonic(),
                channel=channel,
                samples=[float(v) for v in samples.tolist()],
            )

        def _on_imu_frame(frame) -> None:  # noqa: ANN001
            self.processor.add_imu_frame(frame.sensor, frame.samples)
            self.session_recorder.record_imu_frame(
                now_monotonic=time.monotonic(),
                sensor=frame.sensor,
                samples=[[float(v) for v in row] for row in frame.samples.tolist()],
            )

        def _on_ppg_frame(frame) -> None:  # noqa: ANN001
            self.session_recorder.record_ppg_frame(
                now_monotonic=time.monotonic(),
                channel=frame.channel,
                samples=[float(v) for v in frame.samples.tolist()],
            )

        if self._replay_path:
            replay = SessionReplay(self._replay_path)
            self.connector = ReplayConnector(
                replay,
                on_eeg=_on_eeg,
                on_imu=lambda sensor, samples: self.processor.add_imu_frame(sensor, samples),
                on_ppg=lambda _channel, _samples: None,
            )
            logger.info("Replay mode enabled from file: %s", self._replay_path)
        else:
            self.connector = MuseConnector(
                on_eeg=_on_eeg,
                on_imu=_on_imu_frame,
                on_ppg=_on_ppg_frame,
                on_status=lambda msg: logger.info("[Muse] %s", msg),
                debug_logging_enabled=self.settings.debug_logging_enabled,
                stream_config={
                    "eeg": self.settings.stream_eeg_enabled,
                    "accelerometer": self.settings.stream_accelerometer_enabled,
                    "gyroscope": self.settings.stream_gyroscope_enabled,
                    "ppg": self.settings.stream_ppg_enabled,
                    "battery": self.settings.stream_battery_enabled,
                },
            )
        self.apply_settings()
        self._prepare_sounds()

        # Load KV file if present
        if os.path.exists(KV_FILE):
            Builder.load_file(KV_FILE)

        sm = ScreenManager(transition=FadeTransition(duration=0.25))
        sm.add_widget(ScanScreen(name="scan"))
        sm.add_widget(GameScreen(name="game"))
        sm.add_widget(CalibrationScreen(name="calibration"))
        sm.add_widget(TestScreen(name="test"))
        return sm

    def on_stop(self):
        logging.getLogger().removeHandler(self._log_handler)
        self.connector.stop()
        release_lock(self._instance_lock)

    def add_console_line(self, line: str) -> None:
        def _append(*_args):
            self._log_lines.append(line)
            if len(self._log_lines) > self._max_log_lines:
                self._log_lines = self._log_lines[-self._max_log_lines:]
            self.console_output = "\n".join(self._log_lines)
        Clock.schedule_once(_append, 0)

    def shutdown_app(self) -> None:
        logger.info("Closing application requested by user.")
        try:
            self.connector.disconnect()
        except Exception as exc:
            logger.debug("Disconnect on close failed: %s", exc)
        self.stop()

    def apply_settings(self) -> None:
        loaded_profile = None
        if self.processor.profile_store.exists(self.settings.active_profile_id):
            loaded_profile = self.processor.profile_store.load(self.settings.active_profile_id)
        elif self.processor.profile_store.exists(self.settings.default_profile_id):
            loaded_profile = self.processor.profile_store.load(self.settings.default_profile_id)
            self.settings.active_profile_id = self.settings.default_profile_id
        self.processor.apply_settings(self.settings)
        if loaded_profile:
            self.processor.apply_calibration_profile(loaded_profile)
        self.controller.apply_settings(self.settings)
        self.connector.set_stream_config({
            "eeg": self.settings.stream_eeg_enabled,
            "accelerometer": self.settings.stream_accelerometer_enabled,
            "gyroscope": self.settings.stream_gyroscope_enabled,
            "ppg": self.settings.stream_ppg_enabled,
            "battery": self.settings.stream_battery_enabled,
        })

    def persist_settings(self) -> None:
        try:
            save_settings(self.settings)
        except Exception as exc:
            logger.warning("Failed to save settings: %s", exc)

    def _prepare_sounds(self) -> None:
        temp_dir = tempfile.gettempdir()
        found_path = os.path.join(temp_dir, "neuro_gaming_found.wav")
        connect_path = os.path.join(temp_dir, "neuro_gaming_connect.wav")
        if not os.path.exists(found_path):
            self._generate_tone(found_path, freq=760.0, duration=0.12, volume=0.28)
        if not os.path.exists(connect_path):
            self._generate_tone(connect_path, freq=920.0, duration=0.1, volume=0.28)
        self._sound_paths = {"found": found_path, "connect": connect_path}
        self._sounds = {
            "found": SoundLoader.load(found_path),
            "connect": SoundLoader.load(connect_path),
        }

    def _generate_tone(self, path: str, freq: float, duration: float, volume: float) -> None:
        sample_rate = 44100
        count = int(sample_rate * duration)
        with wave.open(path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for i in range(count):
                t = i / sample_rate
                sample = math.sin(2 * math.pi * freq * t)
                value = int(max(-1.0, min(1.0, sample * volume)) * 32767)
                wav_file.writeframes(struct.pack("<h", value))

    def play_found_sound(self) -> None:
        sound = self._sounds.get("found")
        if sound:
            sound.stop()
            sound.play()

    def play_connected_sound(self) -> None:
        sound = self._sounds.get("connect")
        if not sound:
            return
        sound.stop()
        sound.play()
        Clock.schedule_once(lambda *_: sound.play(), 0.18)

    def open_user_guide(self) -> None:
        guide_text = (
            "SYSTEM OPERATIONAL GUIDE\n\n"
            "1) Power on the Muse S headband and ensure all sensors have "
            "direct skin contact.\n"
            "2) Click 'SCAN FOR DEVICES', select your device from the list, "
            "and initialize connection.\n"
            "3) Once connected, verify battery level, signal quality, and "
            "sensor health in the telemetry panel.\n"
            "4) Use 'START CALIBRATION' and remain still for 5-10 seconds "
            "to establish an EEG baseline.\n"
            "5) Use concentrated focus (Beta) for FORWARD, and relaxation (Alpha) "
            "for BACKWARD/LATERAL control.\n"
            "6) Use 'DISCONNECT' to terminate the session and return to "
            "the system scan screen."
        )

        content = BoxLayout(orientation="vertical", spacing=12, padding=16)
        from kivy.graphics import Color, Rectangle
        with content.canvas.before:
            Color(0.08, 0.08, 0.12, 1)
            content_rect = Rectangle(size=content.size, pos=content.pos)
        
        def update_rect(instance, value):
            content_rect.pos = instance.pos
            content_rect.size = instance.size
        content.bind(pos=update_rect, size=update_rect)

        scroll = ScrollView()
        lbl = Label(
            text=guide_text,
            halign="left",
            valign="top",
            size_hint_y=None,
            font_size='13sp',
            line_height=1.2,
            color=(0.85, 0.85, 0.9, 1),
        )
        lbl.bind(
            width=lambda *_: setattr(lbl, "text_size", (lbl.width, None)),
            texture_size=lambda *_: setattr(lbl, "height", lbl.texture_size[1]),
        )
        scroll.add_widget(lbl)
        content.add_widget(scroll)
        close_btn = Button(
            text="CLOSE",
            size_hint_y=None,
            height="48dp",
            font_size='13sp',
            bold=True,
            background_normal='',
            background_color=(0.18, 0.55, 1.0, 1),
        )
        content.add_widget(close_btn)
        popup = Popup(
            title="SYSTEM USER GUIDE",
            content=content,
            size_hint=(0.85, 0.8),
            auto_dismiss=True,
            title_size='14sp',
            title_align='center',
            separator_color=(0.18, 0.55, 1.0, 1)
        )
        close_btn.bind(on_release=popup.dismiss)
        popup.open()


# ──────────────────────────────────────────────────────────────────────────────
# Fallback inline KV (used when neuro_gaming.kv is not found)
# The real KV file provides the same layout with nicer styling.
# ──────────────────────────────────────────────────────────────────────────────

_INLINE_KV = """
<ScanScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 10

        Label:
            text: 'NeuroGaming – Muse S Athena'
            font_size: '22sp'
            size_hint_y: None
            height: '50dp'

        Label:
            text: root.status_text
            font_size: '14sp'
            size_hint_y: None
            height: '40dp'

        Button:
            text: 'Scan for Muse devices'
            size_hint_y: None
            height: '48dp'
            on_release: root.scan()

        RecycleView:
            id: device_list
            data: [{'text': n, 'index': i} for i, n in enumerate(root.device_names)]
            viewclass: 'DeviceRow'
            RecycleBoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height

        Button:
            text: 'Continue in keyboard-only mode'
            size_hint_y: None
            height: '48dp'
            on_release: root.skip_to_keyboard()

        Button:
            text: 'Test Screen (dot control)'
            size_hint_y: None
            height: '48dp'
            on_release: root.go_to_test()

<DeviceRow@Button>:
    index: 0
    size_hint_y: None
    height: '40dp'
    on_release: app.root.get_screen('scan').connect_device(self.index)

<GameScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 8

        # Status bar
        Label:
            text: root.status_text
            font_size: '13sp'
            size_hint_y: None
            height: '36dp'

        # Directional pad
        GridLayout:
            cols: 3
            rows: 3
            size_hint_y: 0.5

            Label: text: ''
            DirectionButton:
                label: '▲'
                active: root.direction == 'FORWARD'
            Label: text: ''

            DirectionButton:
                label: '◄'
                active: root.direction == 'LEFT'
            Label: text: ''
            DirectionButton:
                label: '►'
                active: root.direction == 'RIGHT'

            Label: text: ''
            DirectionButton:
                label: '▼'
                active: root.direction == 'BACKWARD'
            Label: text: ''

        # EEG band-power bars
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: 0.25
            spacing: 4

            Label:
                text: 'Alpha (left / right)   Beta (left / right)'
                font_size: '11sp'
                size_hint_y: None
                height: '20dp'

            BoxLayout:
                spacing: 6
                ProgressBar:
                    value: root.alpha_left * 100
                    max: 100
                ProgressBar:
                    value: root.alpha_right * 100
                    max: 100
                ProgressBar:
                    value: root.beta_left * 100
                    max: 100
                ProgressBar:
                    value: root.beta_right * 100
                    max: 100

        # Control buttons
        BoxLayout:
            size_hint_y: None
            height: '48dp'
            spacing: 6

            Button:
                text: 'Calibration Wizard'
                on_release: root.toggle_calibration()

            Button:
                text: 'Mode: ' + root.key_mode
                on_release: root.toggle_key_mode()

            Button:
                text: 'Disconnect'
                on_release: root.disconnect()

<CalibrationScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 12

        Label:
            text: 'CALIBRATION WIZARD'
            font_size: '20sp'
            bold: True
            size_hint_y: None
            height: '50dp'

        Label:
            text: 'Steps:'
            font_size: '13sp'
            size_hint_y: None
            height: '28dp'

        Label:
            text: root.steps[0]
            font_size: '12sp'
            size_hint_y: None
            height: '32dp'
            color: (1, 1, 1, 1) if root.current_step == 0 else (0.5, 0.5, 0.5, 1)
        Label:
            text: root.steps[1]
            font_size: '12sp'
            size_hint_y: None
            height: '32dp'
            color: (1, 1, 1, 1) if root.current_step == 1 else (0.5, 0.5, 0.5, 1)
        Label:
            text: root.steps[2]
            font_size: '12sp'
            size_hint_y: None
            height: '32dp'
            color: (1, 1, 1, 1) if root.current_step == 2 else (0.5, 0.5, 0.5, 1)
        Label:
            text: root.steps[3]
            font_size: '12sp'
            size_hint_y: None
            height: '32dp'
            color: (1, 1, 1, 1) if root.current_step == 3 else (0.5, 0.5, 0.5, 1)
        Label:
            text: root.steps[4]
            font_size: '12sp'
            size_hint_y: None
            height: '32dp'
            color: (1, 1, 1, 1) if root.current_step == 4 else (0.5, 0.5, 0.5, 1)

        Label:
            text: root.status_text
            font_size: '12sp'
            size_hint_y: None
            height: '36dp'

        BoxLayout:
            size_hint_y: None
            height: '48dp'
            spacing: 8

            Button:
                text: 'CALIBRATING…' if root.is_calibrating else ('RECALIBRATE' if root.calibration_done else 'START CALIBRATION')
                disabled: root.is_calibrating
                on_release: root.start_calibration()

            Button:
                text: 'Back'
                on_release: root.go_back()

<DirectionButton@Label>:
    label: ''
    active: False
    text: self.label
    font_size: '32sp'
    canvas.before:
        Color:
            rgba: (0.2, 0.6, 1, 1) if self.active else (0.2, 0.2, 0.2, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [8]

<TestScreen>:
    canvas.before:
        Color:
            rgba: (0.05, 0.05, 0.1, 1)
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgba: root.dot_color
        Ellipse:
            pos: root.dot_x - root.dot_radius, root.dot_y - root.dot_radius
            size: root.dot_radius * 2, root.dot_radius * 2

    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 8

        Label:
            text: 'Test Screen'
            font_size: '18sp'
            size_hint_y: None
            height: '40dp'

        Widget:
            size_hint_y: 1

        Label:
            text: root.status_text
            font_size: '11sp'
            size_hint_y: None
            height: '28dp'

        BoxLayout:
            size_hint_y: None
            height: '48dp'
            spacing: 6

            Label:
                text: 'LMB: ' + ('ACTIVE' if root.left_active else 'idle')
            Label:
                text: 'RMB: ' + ('ACTIVE' if root.right_active else 'idle')
            Button:
                text: 'Back'
                size_hint_x: 0.4
                on_release: root.go_back()
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_KIVY_KEY_MAP: dict[int, str] = {
    273: "up",
    274: "down",
    276: "left",
    275: "right",
    119: "w",
    115: "s",
    97:  "a",
    100: "d",
}


def _keycode_to_name(key: int, codepoint: str) -> str:
    """Convert a Kivy key code to a direction-key name."""
    if key in _KIVY_KEY_MAP:
        return _KIVY_KEY_MAP[key]
    return codepoint.lower() if codepoint else ""


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroGaming launcher")
    parser.add_argument("--replay", dest="replay_path", help="Replay session JSONL file path")
    args = parser.parse_args()

    app_lock = acquire_lock("neuro_gaming")
    if app_lock is None:
        print("NeuroGaming is already running. Close the existing window and try again.")
        raise SystemExit(0)
    NeuroGamingApp(instance_lock=app_lock, replay_path=args.replay_path).run()
