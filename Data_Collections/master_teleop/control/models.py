from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class MasterHandInput:
    """One USB master-hand sample in calibrated six-joint space."""

    received_monotonic: float
    timestamp: float
    joint_positions_rad: list[float]
    gripper_open: float
    button_a: bool = False
    button_b: bool = False
    buttons_raw: list[int] = field(default_factory=list)
    tracking_valid: bool = True
    profile_name: str = ""
    raw_positions: list[int] = field(default_factory=list)

    @property
    def age_s(self) -> float:
        return monotonic() - self.received_monotonic

    @property
    def right_grip(self) -> float:
        return 1.0 if self.button_a else 0.0

    @property
    def right_trigger(self) -> float:
        return 1.0 - min(max(float(self.gripper_open), 0.0), 1.0)

    # Generic recorder fields. They remain zero in joint-master mode.
    right_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    right_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    right_thumbstick: list[float] = field(default_factory=lambda: [0.0, 0.0])
