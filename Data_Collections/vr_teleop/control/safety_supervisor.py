from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import Pose6, RobotState, VRInput


class SafetyInput(Protocol):
    @property
    def age_s(self) -> float: ...

    tracking_valid: bool
    right_grip: float


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""


def _wrapped_angle_delta(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class SafetySupervisor:
    def __init__(self, config: dict, require_robot_enabled: bool | None = None) -> None:
        safety = config["safety"]
        ws = safety["workspace"]
        self.require_deadman = bool(safety.get("require_deadman", True))
        self.deadman_grip_threshold = float(safety.get("deadman_grip_threshold", 0.65))
        self.watchdog_timeout_s = float(safety.get("watchdog_timeout_s", 0.25))
        self.max_step_mm = float(safety.get("max_step_mm", 2.0))
        self.max_step_deg = float(safety.get("max_step_deg", 1.0))
        self.max_linear_speed_mm_s = float(safety.get("max_linear_speed_mm_s", 30.0))
        self.max_angular_speed_deg_s = float(safety.get("max_angular_speed_deg_s", 10.0))
        self.max_command_lead_mm = float(safety.get("max_command_lead_mm", self.max_step_mm))
        self.max_command_lead_deg = float(safety.get("max_command_lead_deg", self.max_step_deg))
        self.stop_on_feedback_timeout = bool(safety.get("stop_on_feedback_timeout", True))
        self.feedback_timeout_s = float(safety.get("feedback_timeout_s", 0.5))
        self.stop_on_collision_state = bool(safety.get("stop_on_collision_state", True))
        self.stop_on_error_status = bool(safety.get("stop_on_error_status", True))
        configured_require_enabled = bool(safety.get("require_robot_enabled", True))
        self.require_robot_enabled = (
            configured_require_enabled if require_robot_enabled is None else require_robot_enabled
        )
        self.workspace = {
            "x": (float(ws["x_min_mm"]), float(ws["x_max_mm"])),
            "y": (float(ws["y_min_mm"]), float(ws["y_max_mm"])),
            "z": (float(ws["z_min_mm"]), float(ws["z_max_mm"])),
        }

    def check(
        self,
        vr: VRInput | SafetyInput | None,
        robot: RobotState,
        require_deadman: bool | None = None,
        source_name: str = "vr",
    ) -> SafetyDecision:
        if vr is None:
            return SafetyDecision(False, f"waiting_for_{source_name}")
        if vr.age_s > self.watchdog_timeout_s:
            return SafetyDecision(False, f"{source_name}_watchdog_timeout:{vr.age_s:.3f}s")
        if not vr.tracking_valid:
            return SafetyDecision(False, "controller_not_tracked")
        deadman_required = self.require_deadman if require_deadman is None else require_deadman
        if deadman_required and vr.right_grip < self.deadman_grip_threshold:
            return SafetyDecision(False, "deadman_released")
        if not robot.connected:
            return SafetyDecision(False, "robot_not_connected")
        if self.stop_on_feedback_timeout and robot.feedback_age_s > self.feedback_timeout_s:
            return SafetyDecision(False, f"feedback_timeout:{robot.feedback_age_s:.3f}s")
        if self.stop_on_error_status and robot.error_status:
            return SafetyDecision(False, f"robot_error_status:{robot.error_status}")
        if self.stop_on_collision_state and robot.collision_state:
            return SafetyDecision(False, f"collision_state:{robot.collision_state}")
        if self.require_robot_enabled and robot.robot_mode is not None and robot.robot_mode not in (5, 7):
            return SafetyDecision(False, f"robot_not_ready_mode:{robot.robot_mode}")
        if not self.pose_in_workspace(robot.pose_mm_deg):
            pose = ",".join(f"{value:.1f}" for value in robot.pose_mm_deg[:3])
            return SafetyDecision(False, f"current_pose_outside_workspace:{pose}")
        return SafetyDecision(True, "ok")

    def pose_in_workspace(self, pose: Pose6) -> bool:
        return (
            self.workspace["x"][0] <= pose[0] <= self.workspace["x"][1]
            and self.workspace["y"][0] <= pose[1] <= self.workspace["y"][1]
            and self.workspace["z"][0] <= pose[2] <= self.workspace["z"][1]
        )

    def clamp_pose(self, pose: Pose6) -> Pose6:
        out = list(pose)
        out[0] = min(max(out[0], self.workspace["x"][0]), self.workspace["x"][1])
        out[1] = min(max(out[1], self.workspace["y"][0]), self.workspace["y"][1])
        out[2] = min(max(out[2], self.workspace["z"][0]), self.workspace["z"][1])
        return out

    def limit_step(self, current: Pose6, target: Pose6, dt_s: float) -> Pose6:
        max_step_by_speed = self.max_linear_speed_mm_s * max(dt_s, 0.001)
        max_step = min(self.max_step_mm, max_step_by_speed)
        out = list(current)
        for i in range(3):
            delta = target[i] - current[i]
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            out[i] = current[i] + delta
        for i in range(3, 6):
            max_angle_by_speed = self.max_angular_speed_deg_s * max(dt_s, 0.001)
            max_angle_step = min(self.max_step_deg, max_angle_by_speed)
            delta = _wrapped_angle_delta(target[i], current[i])
            if delta > max_angle_step:
                delta = max_angle_step
            elif delta < -max_angle_step:
                delta = -max_angle_step
            out[i] = _normalize_angle(current[i] + delta)
        return self.clamp_pose(out)

    def limit_command_lead(self, actual: Pose6, command: Pose6) -> Pose6:
        """Keep a streamed command close enough to measured robot feedback."""
        out = list(command)
        for i in range(3):
            delta = out[i] - actual[i]
            delta = min(max(delta, -self.max_command_lead_mm), self.max_command_lead_mm)
            out[i] = actual[i] + delta
        for i in range(3, 6):
            delta = _wrapped_angle_delta(out[i], actual[i])
            delta = min(max(delta, -self.max_command_lead_deg), self.max_command_lead_deg)
            out[i] = _normalize_angle(actual[i] + delta)
        return self.clamp_pose(out)
