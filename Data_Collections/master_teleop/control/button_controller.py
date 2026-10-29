from __future__ import annotations

from dataclasses import dataclass, field

from .models import MasterHandInput


@dataclass(frozen=True)
class MasterButtonEvent:
    action: str
    value: bool | None = None
    duration_s: float = 0.0


@dataclass
class MasterButtonState:
    master_locked: bool = False
    servo_enabled: bool = False
    recording: bool = False
    events: list[MasterButtonEvent] = field(default_factory=list)


class MasterButtonController:
    """Single-hand version of Xtrainer's A/B button state machine.

    Button A short press toggles the leader torque lock. Button A long press
    toggles follower alignment/servo. Button B short press toggles recording.
    Events are emitted on release, matching the reference project.
    """

    def __init__(
        self,
        short_press_max_s: float = 0.5,
        long_press_min_s: float = 1.0,
        initial_locked: bool = False,
    ) -> None:
        if short_press_max_s <= 0.0 or long_press_min_s <= short_press_max_s:
            raise ValueError("button press thresholds must satisfy 0 < short < long")
        self.short_press_max_s = float(short_press_max_s)
        self.long_press_min_s = float(long_press_min_s)
        self.state = MasterButtonState(master_locked=bool(initial_locked))
        self._last_a = False
        self._last_b = False
        self._a_pressed_at: float | None = None
        self._b_pressed_at: float | None = None

    def update(
        self,
        sample: MasterHandInput | None,
        now_s: float,
        recording: bool = False,
    ) -> MasterButtonState:
        self.state.events = []
        self.state.recording = bool(recording)
        if sample is None or not sample.tracking_valid:
            return self.state

        if sample.button_a and not self._last_a:
            self._a_pressed_at = now_s
        elif not sample.button_a and self._last_a:
            pressed_at = now_s if self._a_pressed_at is None else self._a_pressed_at
            duration = max(now_s - pressed_at, 0.0)
            if duration < self.short_press_max_s:
                self.state.master_locked = not self.state.master_locked
                self.state.events.append(
                    MasterButtonEvent("master_lock", self.state.master_locked, duration)
                )
            elif duration > self.long_press_min_s:
                self.state.servo_enabled = not self.state.servo_enabled
                self.state.events.append(
                    MasterButtonEvent("servo", self.state.servo_enabled, duration)
                )
            else:
                self.state.events.append(MasterButtonEvent("a_ignored", None, duration))
            self._a_pressed_at = None

        if sample.button_b and not self._last_b:
            self._b_pressed_at = now_s
        elif not sample.button_b and self._last_b:
            pressed_at = now_s if self._b_pressed_at is None else self._b_pressed_at
            duration = max(now_s - pressed_at, 0.0)
            if duration < self.short_press_max_s:
                self.state.events.append(MasterButtonEvent("record", None, duration))
            else:
                self.state.events.append(MasterButtonEvent("b_ignored", None, duration))
            self._b_pressed_at = None

        self._last_a = bool(sample.button_a)
        self._last_b = bool(sample.button_b)
        return self.state

    def force_servo_off(self) -> None:
        self.state.servo_enabled = False
