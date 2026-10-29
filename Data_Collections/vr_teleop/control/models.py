from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any


Pose6 = list[float]
Vec3 = list[float]
Quat = list[float]


@dataclass
class VRInput:
    received_monotonic: float
    timestamp: float
    right_pos: Vec3
    tracking_valid: bool = True
    right_rot: Quat = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    right_grip: float = 0.0
    right_trigger: float = 0.0
    button_a: bool = False
    button_b: bool = False
    right_thumbstick: list[float] = field(default_factory=lambda: [0.0, 0.0])
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def age_s(self) -> float:
        return monotonic() - self.received_monotonic

    @staticmethod
    def from_json(data: dict[str, Any]) -> "VRInput":
        pos = data.get("right_pos", [0.0, 0.0, 0.0])
        rot = data.get("right_rot", [0.0, 0.0, 0.0, 1.0])
        stick = data.get("right_thumbstick", [0.0, 0.0])
        return VRInput(
            received_monotonic=monotonic(),
            timestamp=float(data.get("timestamp", 0.0)),
            right_pos=[float(pos[0]), float(pos[1]), float(pos[2])],
            tracking_valid=bool(data.get("tracking_valid", True)),
            right_rot=[float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],
            right_grip=float(data.get("right_grip", data.get("grip", 0.0))),
            right_trigger=float(data.get("right_trigger", data.get("trigger", 0.0))),
            button_a=bool(data.get("button_a", False)),
            button_b=bool(data.get("button_b", False)),
            right_thumbstick=[float(stick[0]), float(stick[1])],
            raw=data,
        )


@dataclass
class RobotState:
    connected: bool = False
    enabled: bool = False
    pose_mm_deg: Pose6 = field(default_factory=lambda: [350.0, 0.0, 350.0, 180.0, 0.0, 0.0])
    q_actual_deg: list[float] = field(default_factory=list)
    q_target_deg: list[float] = field(default_factory=list)
    target_pose_mm_deg: Pose6 = field(default_factory=list)
    q_actual_velocity_deg_s: list[float] = field(default_factory=list)
    tcp_speed_actual_mm_deg_s: list[float] = field(default_factory=list)
    tcp_force_actual: list[float] = field(default_factory=list)
    robot_mode: int | None = None
    safety_state: int | None = None
    error_status: int = 0
    collision_state: int = 0
    speed_scaling: float | None = None
    last_feedback_monotonic: float = field(default_factory=monotonic)
    message: str = ""

    @property
    def feedback_age_s(self) -> float:
        return monotonic() - self.last_feedback_monotonic


@dataclass
class TeleopCommand:
    allowed: bool
    target_pose_mm_deg: Pose6 | None = None
    target_joint_positions_deg: list[float] | None = None
    reason: str = ""
    clutched: bool = False
    control_source: str = "vr"
    control_mode: str = "servo_p"
