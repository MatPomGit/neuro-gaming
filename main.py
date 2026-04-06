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
import os
import threading

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
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
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager

from src.game_controller import GameController
from src.muse_connector import MuseConnector
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

__version__ = "1.0.0"

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
    is_scanning = BooleanProperty(False)
    scan_pulse = NumericProperty(1.0)

    _pulse_anim = None

    def on_enter(self, *args):
        app = App.get_running_app()
        app.connector.start()
        app.connector._on_status = self._update_status  # noqa: SLF001

    def scan(self) -> None:
        if self.is_scanning:
            return
        app = App.get_running_app()
        self.status_text = "Scanning…"
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
        self.status_text = status
        self.is_scanning = False
        self.scan_pulse = 1.0

    def connect_device(self, index: int) -> None:
        app = App.get_running_app()
        devices = app.connector.devices
        if index < 0 or index >= len(devices):
            self.status_text = "Invalid device selection."
            return
        
        device = devices[index]
        self.status_text = f"Connecting to {device.name}…"
        self.is_scanning = True  # Re-use flag to disable buttons during connect
        
        # Run connection in background thread to avoid UI freeze
        threading.Thread(
            target=self._do_connect, 
            args=(app, device), 
            daemon=True
        ).start()

    def _do_connect(self, app, device):
        try:
            # Zamiast całego obiektu przekaż tylko adres i nazwę
            addr = device.address if hasattr(device, 'address') else device[0]
            name = device.name if hasattr(device, 'name') else (device[1] if len(device) > 1 else "")
            app.connector.connect((addr, name))   # teraz connect sam odświeży urządzenie
            Clock.schedule_once(lambda dt: self._connect_done(True))
        except Exception as exc:
            error_msg = str(exc)
            Clock.schedule_once(lambda dt: self._connect_done(False, error_msg))

    def _connect_done(self, success: bool, error_msg: str = "") -> None:
        self.is_scanning = False
        if success:
            app = App.get_running_app()
            app.root.current = "game"
        else:
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

    _update_event = None

    def on_enter(self, *args):
        app = App.get_running_app()
        self.connected = app.connector.is_connected
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

        if app.connector.is_connected:
            # Update EEG-driven direction
            eeg_dir = app.processor.get_direction()
            app.controller.update(eeg_dir)

            # Refresh metrics display
            m = app.processor.get_metrics()
            self.metrics = {k: dict(v) for k, v in m.items()}
            self._update_bars(m)

            # Refresh per-channel signal quality
            sq = app.processor.get_signal_quality()
            self.quality_tp9  = sq.get("TP9",  0.0)
            self.quality_af7  = sq.get("AF7",  0.0)
            self.quality_af8  = sq.get("AF8",  0.0)
            self.quality_tp10 = sq.get("TP10", 0.0)

        self.direction = app.controller.current_direction
        tag = "[EEG]" if app.connector.is_connected else "[KB]"
        label = _DIR_LABELS.get(self.direction, self.direction)
        self.status_text = f"{tag}  {label}"
        self._update_device_status(app)

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
            return
        state = app.connector.device_state
        battery = state.get("battery_level")
        battery_value = f"{battery}%" if isinstance(battery, int) else "n/a"
        sensors = state.get("available_sensors") or []
        sensor_summary = ", ".join(sensors) if sensors else "unknown"
        sample_rate = state.get("sample_rate_hz", 0)

        self.device_text = (
            f"Device: {state.get('device_name', 'Muse')}  "
            f"({sample_rate} Hz)"
        )
        self.battery_text = f"Battery: {battery_value}"
        self.sensors_text = f"Sensors: {sensor_summary}"

    # ── direction callback ─────────────────────────────────────────────────

    def _on_direction_change(self, new_direction: str) -> None:
        self.direction = new_direction

    # ── keyboard input ─────────────────────────────────────────────────────

    def _on_key_down(self, window, key, scancode, codepoint, modifiers) -> None:  # noqa: ANN001
        key_name = _keycode_to_name(key, codepoint)
        App.get_running_app().controller.handle_key_down(key_name)

    def _on_key_up(self, window, key, *args) -> None:  # noqa: ANN001
        key_name = _keycode_to_name(key, "")
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
        self.key_mode = app.controller.key_mode

    def disconnect(self) -> None:
        app = App.get_running_app()
        app.connector.disconnect()
        app.processor.reset()
        app.controller.reset()
        app.root.current = "scan"

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

    _update_event = None

    def on_enter(self, *args) -> None:
        self._keys_held: set[str] = set()
        Clock.schedule_once(self._init_dot_position, 0)
        Window.bind(on_key_down=self._on_key_down)
        Window.bind(on_key_up=self._on_key_up)
        self._update_event = Clock.schedule_interval(self._tick, 1.0 / 60)

    def on_leave(self, *args) -> None:
        if self._update_event:
            self._update_event.cancel()
            self._update_event = None
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

    # ── navigation ─────────────────────────────────────────────────────────

    def go_back(self) -> None:
        App.get_running_app().root.current = "scan"


# ──────────────────────────────────────────────────────────────────────────────
# Screen: Calibration
# ──────────────────────────────────────────────────────────────────────────────

# Total recording time for a single calibration run (seconds)
CALIBRATION_DURATION = 10

# Step descriptions shown in the calibration wizard
_CALIB_STEPS = [
    "1. Put on the Muse headband and ensure all sensors have direct skin contact.",
    "2. Verify sensor contact quality — all quality bars should be solid green.",
    "3. Sit still, relax, and look straight ahead with eyes open.",
    "4. Press START CALIBRATION and remain still for 10 seconds.",
    "5. When the timer reaches zero the baseline is saved automatically.",
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
    timer_value = NumericProperty(CALIBRATION_DURATION)

    # Human-readable status displayed below the step list
    status_text = StringProperty("Follow the steps above, then press START CALIBRATION.")

    # Signal quality per channel (0–1)
    quality_tp9  = NumericProperty(0.0)
    quality_af7  = NumericProperty(0.0)
    quality_af8  = NumericProperty(0.0)
    quality_tp10 = NumericProperty(0.0)

    # Step label texts (read-only; bound from the constant list)
    steps = ListProperty(_CALIB_STEPS)

    # Expose duration so the KV timer bar can reference it without a magic number
    calibration_duration = NumericProperty(CALIBRATION_DURATION)

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
            remaining = max(0.0, CALIBRATION_DURATION - self._elapsed)
            self.timer_value = remaining
            self.status_text = f"Calibrating… {int(remaining) + 1}s remaining — stay still."
            if remaining <= 0:
                self._finish_calibration()

    def _auto_advance_step(self) -> None:
        """Advance the highlighted step automatically as conditions are met."""
        avg_quality = (
            self.quality_tp9 + self.quality_af7 +
            self.quality_af8 + self.quality_tp10
        ) / 4.0
        if avg_quality >= 0.5 and self.current_step < 2:
            self.current_step = 2
        elif avg_quality < 0.5 and self.current_step >= 2:
            self.current_step = 1

    # ── button handlers ────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Begin recording a new EEG baseline."""
        app = App.get_running_app()
        self.is_calibrating = True
        self.calibration_done = False
        self._elapsed = 0.0
        self.timer_value = CALIBRATION_DURATION
        self.current_step = 3
        self.status_text = f"Calibrating… {CALIBRATION_DURATION}s remaining — stay still."
        app.processor.start_calibration()
        logger.info("Calibration wizard: started")

    def _finish_calibration(self) -> None:
        """Stop recording and compute the baseline."""
        app = App.get_running_app()
        app.processor.stop_calibration()
        self.is_calibrating = False
        self.calibration_done = True
        self.current_step = 4
        self.timer_value = 0.0
        self.status_text = "Calibration complete — baseline saved.  You may return to the game."
        logger.info("Calibration wizard: finished")

    def go_back(self) -> None:
        """Return to the previous screen."""
        App.get_running_app().root.current = "game"


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

class NeuroGamingApp(App):
    """Root Kivy application."""

    version = StringProperty(__version__)

    def build(self):
        # Core objects shared across screens
        self.processor = SignalProcessor()
        self.connector = MuseConnector(
            on_eeg=self.processor.add_samples,
            on_status=lambda msg: logger.info("[Muse] %s", msg),
        )
        self.controller = GameController(key_mode="arrow")

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
        self.connector.stop()

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
    NeuroGamingApp().run()
