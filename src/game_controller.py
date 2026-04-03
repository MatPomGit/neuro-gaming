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
"""

import logging
from collections.abc import Callable
from typing import Optional

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
    """

    def __init__(
        self,
        on_direction_change: Optional[Callable[[str], None]] = None,
        key_mode: str = "arrow",
        on_mouse_action: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.current_direction: str = DIRECTION_NONE
        self.key_mode = key_mode
        self.on_direction_change = on_direction_change
        self.on_mouse_action = on_mouse_action

        self._pending_direction: str = DIRECTION_NONE
        self._pending_count: int = 0

        self.left_button_pressed: bool = False
        self.right_button_pressed: bool = False

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

        if self._pending_count >= HYSTERESIS_COUNT:
            if new_direction != self.current_direction:
                self.current_direction = new_direction
                logger.debug("Direction changed → %s", new_direction)
                if self.on_direction_change:
                    self.on_direction_change(new_direction)

    def set_direction(self, direction: str) -> None:
        """Directly set the current direction (used for keyboard overrides)."""
        if direction != self.current_direction:
            self.current_direction = direction
            self._pending_direction = direction
            self._pending_count = HYSTERESIS_COUNT
            logger.debug("Direction set (manual) → %s", direction)
            if self.on_direction_change:
                self.on_direction_change(direction)

    def get_active_key(self) -> str:
        """Return the keyboard key that represents the current direction.

        Returns an empty string when the direction is ``DIRECTION_NONE``.
        """
        arrow, wasd = KEY_MAP.get(self.current_direction, ("", ""))
        return arrow if self.key_mode == "arrow" else wasd

    def reset(self) -> None:
        """Reset to the neutral state."""
        self.current_direction = DIRECTION_NONE
        self._pending_direction = DIRECTION_NONE
        self._pending_count = 0
        self.left_button_pressed = False
        self.right_button_pressed = False

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
            if self.on_mouse_action:
                self.on_mouse_action(ACTION_LEFT_CLICK)
            return True
        if button == MOUSE_RIGHT:
            self.right_button_pressed = True
            logger.debug("Mouse right button pressed")
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
            return True
        if button == MOUSE_RIGHT:
            self.right_button_pressed = False
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
