"""
Game controller for the NeuroGaming app.

Translates directional commands (from EEG or keyboard) into key-press
events and maintains the current active direction.

Keyboard mapping
----------------
+----------+---------+---------+
| Direction| Arrow   | WASD    |
+==========+=========+=========+
| FORWARD  | ↑       | W       |
+----------+---------+---------+
| BACKWARD | ↓       | S       |
+----------+---------+---------+
| LEFT     | ←       | A       |
+----------+---------+---------+
| RIGHT    | →       | D       |
+----------+---------+---------+
| NONE     | (none)  | (none)  |
+----------+---------+---------+

Mouse button mapping
--------------------
+-------------+------------------+
| Button      | Action constant  |
+=============+==================+
| Left (LMB)  | ``"left_click"`` |
+-------------+------------------+
| Right (RMB) | ``"right_click"``|
+-------------+------------------+

Button forwarding
-----------------
When ``forwarding_enabled=True`` is passed to :class:`GameController`,
detected directions are forwarded to the OS as real key presses via
``pynput``.  This allows other running games or applications to receive
the EEG-derived inputs as actual keyboard/mouse events.  If ``pynput``
is not available (e.g. on Android), forwarding silently degrades to a
no-op.
"""

import logging
from collections.abc import Callable
from typing import Optional

from src.settings import AppSettings
from src.signal_processor import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_LEFT,
    DIRECTION_NONE,
    DIRECTION_RIGHT,
)

logger = logging.getLogger(__name__)

# Mouse button constants
MOUSE_LEFT  = "left"
MOUSE_RIGHT = "right"

# Action constants emitted by mouse button events
ACTION_LEFT_CLICK  = "left_click"
ACTION_RIGHT_CLICK = "right_click"

# Maps a direction constant to (arrow_key, wasd_key)
KEY_MAP: dict[str, tuple[str, str]] = {
    DIRECTION_FORWARD:  ("up",    "w"),
    DIRECTION_BACKWARD: ("down",  "s"),
    DIRECTION_LEFT:     ("left",  "a"),
    DIRECTION_RIGHT:    ("right", "d"),
    DIRECTION_NONE:     ("",      ""),
}

# How many consecutive identical readings are required before we accept
# a new direction (simple hysteresis to avoid jitter).
HYSTERESIS_COUNT = 3

# pynput arrow-key name → pynput Key constant (populated lazily)
_PYNPUT_ARROW_NAMES = ("up", "down", "left", "right")


class ButtonForwarder:
    """Forwards detected directions and mouse events to the OS as real input.

    Uses ``pynput`` as the OS-level input injection backend.  When
    ``pynput`` is unavailable (e.g. on Android) all public methods are
    silent no-ops so the rest of the application continues to work.

    Attributes
    ----------
    available:
        ``True`` when ``pynput`` was imported successfully and OS input
        injection is active.
    """

    def __init__(self) -> None:
        self._keyboard = None
        self._mouse = None
        self._Key = None
        self._Button = None
        try:
            from pynput import keyboard as _kb, mouse as _ms  # noqa: PLC0415
            self._keyboard = _kb.Controller()
            self._mouse = _ms.Controller()
            self._Key = _kb.Key
            self._Button = _ms.Button
            logger.debug("ButtonForwarder: pynput loaded – OS forwarding active")
        except (ImportError, OSError) as exc:  # pragma: no cover
            logger.warning("ButtonForwarder: pynput unavailable (%s) – forwarding is a no-op", exc)

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """``True`` when pynput is loaded and OS injection is possible."""
        return self._keyboard is not None

    def press_direction(self, direction: str, key_mode: str = "arrow") -> None:
        """Simulate pressing the OS key that corresponds to *direction*."""
        if not self.available or direction == DIRECTION_NONE:
            return
        key = self._direction_to_key(direction, key_mode)
        if key is not None:
            self._safe_press(key)

    def release_direction(self, direction: str, key_mode: str = "arrow") -> None:
        """Simulate releasing the OS key that corresponds to *direction*."""
        if not self.available or direction == DIRECTION_NONE:
            return
        key = self._direction_to_key(direction, key_mode)
        if key is not None:
            self._safe_release(key)

    def press_mouse_button(self, button: str) -> None:
        """Simulate pressing a mouse button at the OS level."""
        if not self.available:
            return
        btn = self._str_to_button(button)
        if btn is not None:
            try:
                self._mouse.press(btn)
            except Exception as exc:
                logger.warning("ButtonForwarder: mouse press failed: %s", exc)

    def release_mouse_button(self, button: str) -> None:
        """Simulate releasing a mouse button at the OS level."""
        if not self.available:
            return
        btn = self._str_to_button(button)
        if btn is not None:
            try:
                self._mouse.release(btn)
            except Exception as exc:
                logger.warning("ButtonForwarder: mouse release failed: %s", exc)

    # ── private helpers ────────────────────────────────────────────────────

    def _direction_to_key(self, direction: str, key_mode: str):
        """Return the pynput key object for *direction* and *key_mode*."""
        arrow, wasd = KEY_MAP.get(direction, ("", ""))
        raw = arrow if key_mode == "arrow" else wasd
        if not raw:
            return None
        if raw in _PYNPUT_ARROW_NAMES:
            return getattr(self._Key, raw)
        return raw  # single character (e.g. 'w', 'a', 's', 'd')

    def _str_to_button(self, button: str):
        """Return the pynput mouse Button for *button* name."""
        if button == MOUSE_LEFT:
            return self._Button.left
        if button == MOUSE_RIGHT:
            return self._Button.right
        return None

    def _safe_press(self, key) -> None:
        try:
            self._keyboard.press(key)
        except Exception as exc:
            logger.warning("ButtonForwarder: key press failed: %s", exc)

    def _safe_release(self, key) -> None:
        try:
            self._keyboard.release(key)
        except Exception as exc:
            logger.warning("ButtonForwarder: key release failed: %s", exc)


class GameController:
    """Manages the current directional game state.

    Attributes
    ----------
    current_direction:
        The currently active direction (one of the ``DIRECTION_*``
        constants from :mod:`src.signal_processor`).
    key_mode:
        ``"arrow"`` or ``"wasd"`` – which key style to report.
    on_direction_change:
        Optional callback ``(new_direction: str) -> None`` invoked
        whenever the active direction changes.
    left_button_pressed:
        ``True`` while the left mouse button is held down.
    right_button_pressed:
        ``True`` while the right mouse button is held down.
    on_mouse_action:
        Optional callback ``(action: str) -> None`` invoked on each
        mouse button press.  *action* is one of ``ACTION_LEFT_CLICK``
        or ``ACTION_RIGHT_CLICK``.
    forwarding_enabled:
        When ``True``, direction changes and mouse events are forwarded
        to the OS as real keyboard/mouse inputs via :class:`ButtonForwarder`.
        Defaults to ``False``.
    """

    def __init__(
        self,
        on_direction_change: Optional[Callable[[str], None]] = None,
        key_mode: str = "arrow",
        on_mouse_action: Optional[Callable[[str], None]] = None,
        forwarding_enabled: bool = False,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.current_direction: str = DIRECTION_NONE
        self.key_mode = key_mode
        self.on_direction_change = on_direction_change
        self.on_mouse_action = on_mouse_action
        self.forwarding_enabled = forwarding_enabled

        self._pending_direction: str = DIRECTION_NONE
        self._pending_count: int = 0
        self.hysteresis_count: int = HYSTERESIS_COUNT

        self.left_button_pressed: bool = False
        self.right_button_pressed: bool = False

        self._forwarder: ButtonForwarder = ButtonForwarder()
        if settings is not None:
            self.apply_settings(settings)

    # ── public API ─────────────────────────────────────────────────────────

    def update(self, new_direction: str) -> None:
        """Feed a new direction reading from the signal processor.

        Applies hysteresis: the direction must be seen
        ``HYSTERESIS_COUNT`` times in a row before it is accepted.
        """
        if new_direction == self._pending_direction:
            self._pending_count += 1
        else:
            self._pending_direction = new_direction
            self._pending_count = 1

        if self._pending_count >= self.hysteresis_count:
            if new_direction != self.current_direction:
                previous = self.current_direction
                self.current_direction = new_direction
                logger.debug("Direction changed → %s", new_direction)
                self._emit_direction_change(previous, new_direction)

    def set_direction(self, direction: str) -> None:
        """Directly set the current direction (used for keyboard overrides)."""
        if direction != self.current_direction:
            previous = self.current_direction
            self.current_direction = direction
            self._pending_direction = direction
            self._pending_count = self.hysteresis_count
            logger.debug("Direction set (manual) → %s", direction)
            self._emit_direction_change(previous, direction)

    def apply_settings(self, settings: AppSettings) -> None:
        """Apply runtime settings from the shared application settings object."""
        self.key_mode = settings.key_mode
        self.forwarding_enabled = settings.forwarding_enabled
        self.hysteresis_count = settings.hysteresis_count

    def get_active_key(self) -> str:
        """Return the keyboard key that represents the current direction.

        Returns an empty string when the direction is ``DIRECTION_NONE``.
        """
        arrow, wasd = KEY_MAP.get(self.current_direction, ("", ""))
        return arrow if self.key_mode == "arrow" else wasd

    def reset(self) -> None:
        """Reset to the neutral state."""
        if self.forwarding_enabled:
            self._forwarder.release_direction(self.current_direction, self.key_mode)
            if self.left_button_pressed:
                self._forwarder.release_mouse_button(MOUSE_LEFT)
            if self.right_button_pressed:
                self._forwarder.release_mouse_button(MOUSE_RIGHT)
        self.current_direction = DIRECTION_NONE
        self._pending_direction = DIRECTION_NONE
        self._pending_count = 0
        self.left_button_pressed = False
        self.right_button_pressed = False

    # ── private helpers ────────────────────────────────────────────────────

    def _emit_direction_change(self, previous: str, new_direction: str) -> None:
        """Forward OS key events and invoke the callback for a direction change."""
        if self.forwarding_enabled:
            self._forwarder.release_direction(previous, self.key_mode)
            self._forwarder.press_direction(new_direction, self.key_mode)
        if self.on_direction_change:
            self.on_direction_change(new_direction)

    # ── keyboard input handler ─────────────────────────────────────────────

    def handle_key_down(self, key: str) -> bool:
        """Handle a keyboard key-down event.

        Parameters
        ----------
        key:
            Kivy key name (e.g. ``"up"``, ``"w"``).

        Returns
        -------
        bool:
            ``True`` if the key was handled, ``False`` otherwise.
        """
        direction = _key_to_direction(key)
        if direction is not None:
            self.set_direction(direction)
            return True
        return False

    def handle_key_up(self, key: str) -> bool:
        """Handle a keyboard key-up event.

        Resets the direction to ``DIRECTION_NONE`` if the released key
        matches the current direction.
        """
        direction = _key_to_direction(key)
        if direction is not None and direction == self.current_direction:
            self.set_direction(DIRECTION_NONE)
            return True
        return False

    # ── mouse input handler ────────────────────────────────────────────────

    def handle_mouse_down(self, button: str) -> bool:
        """Handle a mouse button press event.

        Parameters
        ----------
        button:
            Mouse button name: ``"left"`` or ``"right"``
            (use :data:`MOUSE_LEFT` / :data:`MOUSE_RIGHT`).

        Returns
        -------
        bool:
            ``True`` if the button was handled, ``False`` otherwise.
        """
        if button == MOUSE_LEFT:
            self.left_button_pressed = True
            logger.debug("Mouse left button pressed")
            if self.forwarding_enabled:
                self._forwarder.press_mouse_button(MOUSE_LEFT)
            if self.on_mouse_action:
                self.on_mouse_action(ACTION_LEFT_CLICK)
            return True
        if button == MOUSE_RIGHT:
            self.right_button_pressed = True
            logger.debug("Mouse right button pressed")
            if self.forwarding_enabled:
                self._forwarder.press_mouse_button(MOUSE_RIGHT)
            if self.on_mouse_action:
                self.on_mouse_action(ACTION_RIGHT_CLICK)
            return True
        return False

    def handle_mouse_up(self, button: str) -> bool:
        """Handle a mouse button release event.

        Parameters
        ----------
        button:
            Mouse button name: ``"left"`` or ``"right"``
            (use :data:`MOUSE_LEFT` / :data:`MOUSE_RIGHT`).

        Returns
        -------
        bool:
            ``True`` if the button was handled, ``False`` otherwise.
        """
        if button == MOUSE_LEFT:
            self.left_button_pressed = False
            if self.forwarding_enabled:
                self._forwarder.release_mouse_button(MOUSE_LEFT)
            return True
        if button == MOUSE_RIGHT:
            self.right_button_pressed = False
            if self.forwarding_enabled:
                self._forwarder.release_mouse_button(MOUSE_RIGHT)
            return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Reverse lookup: key name → direction
_KEY_DIRECTION_MAP: dict[str, str] = {}
for _dir, (_arrow, _wasd) in KEY_MAP.items():
    if _dir == DIRECTION_NONE:
        continue
    if _arrow:
        _KEY_DIRECTION_MAP[_arrow] = _dir
    if _wasd:
        _KEY_DIRECTION_MAP[_wasd] = _dir


def _key_to_direction(key: str) -> Optional[str]:
    """Return the direction for *key*, or ``None`` if not a game key."""
    return _KEY_DIRECTION_MAP.get(key.lower())
