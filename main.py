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
from kivy.uix.screenmanager import Screen, ScreenManager

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def on_enter(self, *args):
        app = App.get_running_app()
        app.connector.start()
        app.connector._on_status = self._update_status  # noqa: SLF001

    def scan(self) -> None:
        app = App.get_running_app()
        self.status_text = "Scanning…"
        self.device_names = []
        try:
            app.connector.scan(timeout=5.0)
            names = [
                f"{d.name}  ({d.address})" for d in app.connector.devices
            ]
            self.device_names = names if names else ["No Muse devices found."]
            self.status_text = (
                f"Found {len(app.connector.devices)} device(s)."
                if app.connector.devices
                else "No Muse devices found."
            )
        except Exception as exc:
            self.status_text = f"Scan error: {exc}"

    def connect_device(self, index: int) -> None:
        app = App.get_running_app()
        devices = app.connector.devices
        if index < 0 or index >= len(devices):
            self.status_text = "Invalid device selection."
            return
        device = devices[index]
        self.status_text = f"Connecting to {device.name}…"
        try:
            app.connector.connect(device)
            app.root.current = "game"
        except Exception as exc:
            self.status_text = f"Connection failed: {exc}"

    def skip_to_keyboard(self) -> None:
        """Continue in keyboard-only mode (no Muse required)."""
        App.get_running_app().root.current = "game"

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

        self.direction = app.controller.current_direction
        self.status_text = (
            f"{'[EEG] ' if app.connector.is_connected else '[Keyboard] '}"
            f"Direction: {self.direction}"
        )

    def _update_bars(self, m: dict) -> None:
        """Normalise band powers to 0–1 for UI progress bars."""
        def _clamp(v: float) -> float:
            return max(0.0, min(1.0, abs(v) / 5.0))

        self.alpha_left  = _clamp(m.get("AF7", {}).get("alpha", 0.0))
        self.alpha_right = _clamp(m.get("AF8", {}).get("alpha", 0.0))
        self.beta_left   = _clamp(m.get("AF7", {}).get("beta",  0.0))
        self.beta_right  = _clamp(m.get("AF8", {}).get("beta",  0.0))

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
        app = App.get_running_app()
        if self.is_calibrating:
            app.processor.stop_calibration()
            self.is_calibrating = False
            self.status_text = "Calibration complete."
        else:
            app.processor.start_calibration()
            self.is_calibrating = True
            self.status_text = "Calibrating… sit still with eyes open."

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


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

class NeuroGamingApp(App):
    """Root Kivy application."""

    def build(self):
        # Core objects shared across screens
        self.processor = SignalProcessor()
        self.connector = MuseConnector(
            on_eeg=self.processor.add_samples,
            on_status=lambda msg: logger.info("[Muse] %s", msg),
        )
        self.controller = GameController(key_mode="arrow")

        # Load KV file if present; otherwise use inline KV string
        if os.path.exists(KV_FILE):
            Builder.load_file(KV_FILE)
        else:
            Builder.load_string(_INLINE_KV)

        sm = ScreenManager()
        sm.add_widget(ScanScreen(name="scan"))
        sm.add_widget(GameScreen(name="game"))
        return sm

    def on_stop(self):
        self.connector.stop()


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
                text: 'Calibrate' if not root.is_calibrating else 'Stop Calibration'
                on_release: root.toggle_calibration()

            Button:
                text: 'Mode: ' + root.key_mode
                on_release: root.toggle_key_mode()

            Button:
                text: 'Disconnect'
                on_release: root.disconnect()

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
