from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Pose6, RobotState, TeleopCommand, VRInput
from .safety_supervisor import SafetySupervisor


def _axis_value(vec: list[float], spec: str) -> float:
    sign = -1.0 if spec.startswith("-") else 1.0
    axis = spec[-1].lower()
    index = {"x": 0, "y": 1, "z": 2}[axis]
    return sign * vec[index]


def _quat_normalize(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [value / norm for value in q]


def _smoothing_alpha(dt_s: float, tau_s: float) -> float:
    if tau_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-max(dt_s, 0.0) / tau_s)


def _apply_deadband(value: float, threshold: float) -> float:
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    return math.copysign(magnitude - threshold, value)


def _quat_nlerp(current: list[float], target: list[float], alpha: float) -> list[float]:
    current_q = _quat_normalize(current)
    target_q = _quat_normalize(target)
    if sum(a * b for a, b in zip(current_q, target_q)) < 0.0:
        target_q = [-value for value in target_q]
    blended = [
        current_q[index] + (target_q[index] - current_q[index]) * alpha
        for index in range(4)
    ]
    return _quat_normalize(blended)


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _relative_rotation_vector_deg(current: list[float], reference: list[float]) -> list[float]:
    current_q = _quat_normalize(current)
    reference_q = _quat_normalize(reference)
    inverse_reference = [-reference_q[0], -reference_q[1], -reference_q[2], reference_q[3]]
    delta = _quat_normalize(_quat_multiply(current_q, inverse_reference))
    if delta[3] < 0.0:
        delta = [-value for value in delta]
    vector_norm = math.sqrt(sum(value * value for value in delta[:3]))
    if vector_norm < 1e-9:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(delta[3], 0.0))
    scale = math.degrees(angle) / vector_norm
    return [delta[index] * scale for index in range(3)]


@dataclass
class MapperState:
    calibrated: bool = False
    calibration_hand_ref: list[float] | None = None
    calibration_hand_rot_ref: list[float] | None = None
    calibration_robot_ref: Pose6 | None = None
    clutched: bool = False
    hand_ref: list[float] | None = None
    hand_rot_ref: list[float] | None = None
    robot_ref: Pose6 | None = None
    filtered_hand_pos: list[float] | None = None
    filtered_hand_rot: list[float] | None = None
    last_target: Pose6 | None = None
    use_calibration_reference: bool = False
    last_calibration_button: bool = False


class RelativeVRMapper:
    def __init__(self, config: dict, safety: SafetySupervisor) -> None:
        mapping = config["vr_mapping"]
        robot_cfg = config["robot"]
        self.scale = float(mapping.get("translation_scale_mm_per_m", 200.0))
        self.robot_x_from = str(mapping.get("robot_x_from", "+z"))
        self.robot_y_from = str(mapping.get("robot_y_from", "-x"))
        self.robot_z_from = str(mapping.get("robot_z_from", "+y"))
        self.robot_rx_from = str(mapping.get("robot_rx_from", self.robot_x_from))
        self.robot_ry_from = str(mapping.get("robot_ry_from", self.robot_y_from))
        self.robot_rz_from = str(mapping.get("robot_rz_from", self.robot_z_from))
        self.enable_orientation = bool(mapping.get("enable_orientation", False))
        self.rotation_scale = float(mapping.get("rotation_scale_deg_per_deg", 0.3))
        self.lock_controller_twist = bool(mapping.get("lock_controller_twist", False))
        self.controller_twist_axis = str(
            mapping.get("controller_twist_axis", "z")
        ).lower()
        if self.controller_twist_axis not in ("x", "y", "z"):
            raise ValueError("vr_mapping.controller_twist_axis must be x, y, or z")
        self.position_smoothing_tau_s = float(mapping.get("position_smoothing_tau_s", 0.0))
        self.rotation_smoothing_tau_s = float(mapping.get("rotation_smoothing_tau_s", 0.0))
        # Smooth the command target after mapping as well as the controller
        # input. This prevents delayed robot feedback from making the target
        # alternate around the current TCP pose.
        self.target_smoothing_tau_s = float(mapping.get("target_smoothing_tau_s", 0.0))
        self.position_deadband_m = float(mapping.get("position_deadband_m", 0.0))
        self.rotation_deadband_deg = float(mapping.get("rotation_deadband_deg", 0.0))
        self.lock_orientation = bool(robot_cfg.get("lock_orientation", True))
        calibration = config.get("calibration", {})
        self.require_calibration = bool(calibration.get("required", True))
        self.calibration_button = str(calibration.get("button", "button_a"))
        self.expected_robot_pose = [
            float(value)
            for value in calibration.get("robot_initial_pose_mm_deg", robot_cfg["default_pose_mm_deg"])
        ]
        self.calibration_position_tolerance_mm = float(
            calibration.get("position_tolerance_mm", 10.0)
        )
        self.calibration_orientation_tolerance_deg = float(
            calibration.get("orientation_tolerance_deg", 10.0)
        )
        self.safety = safety
        self.state = MapperState()

    def release_clutch(self) -> None:
        self.state.clutched = False
        self.state.hand_ref = None
        self.state.hand_rot_ref = None
        self.state.robot_ref = None
        self.state.filtered_hand_pos = None
        self.state.filtered_hand_rot = None
        self.state.last_target = None

    def map(self, vr: VRInput | None, robot: RobotState, dt_s: float) -> TeleopCommand:
        base_decision = self.safety.check(vr, robot, require_deadman=False)
        if not base_decision.allowed:
            self.release_clutch()
            return TeleopCommand(False, reason=base_decision.reason, clutched=False)
        assert vr is not None

        calibration_pressed = bool(getattr(vr, self.calibration_button, False))
        calibration_rising = calibration_pressed and not self.state.last_calibration_button
        self.state.last_calibration_button = calibration_pressed
        if calibration_rising:
            if vr.right_grip >= self.safety.deadman_grip_threshold:
                return TeleopCommand(False, reason="calibration_requires_grip_released", clutched=False)
            mismatch = self._initial_pose_mismatch(robot.pose_mm_deg)
            if mismatch:
                return TeleopCommand(False, reason=mismatch, clutched=False)
            self.state.calibrated = True
            self.state.calibration_hand_ref = list(vr.right_pos)
            self.state.calibration_hand_rot_ref = list(vr.right_rot)
            self.state.calibration_robot_ref = list(robot.pose_mm_deg)
            self.state.clutched = False
            self.state.hand_ref = list(vr.right_pos)
            self.state.hand_rot_ref = list(vr.right_rot)
            self.state.robot_ref = list(robot.pose_mm_deg)
            self.state.filtered_hand_pos = list(vr.right_pos)
            self.state.filtered_hand_rot = list(vr.right_rot)
            self.state.last_target = list(robot.pose_mm_deg)
            self.state.use_calibration_reference = True
            return TeleopCommand(False, reason="calibrated_release_A_then_hold_grip", clutched=False)

        if self.require_calibration and not self.state.calibrated:
            self.release_clutch()
            return TeleopCommand(False, reason="calibration_required_release_grip_and_press_A", clutched=False)

        decision = self.safety.check(vr, robot, require_deadman=True)
        if not decision.allowed:
            self.release_clutch()
            return TeleopCommand(False, reason=decision.reason, clutched=False)

        if not self.state.clutched:
            self.state.clutched = True
            if self.state.use_calibration_reference:
                self.state.hand_ref = list(self.state.calibration_hand_ref or vr.right_pos)
                self.state.hand_rot_ref = list(self.state.calibration_hand_rot_ref or vr.right_rot)
                self.state.robot_ref = list(self.state.calibration_robot_ref or robot.pose_mm_deg)
                self.state.last_target = list(self.state.robot_ref)
                self.state.use_calibration_reference = False
            else:
                self.state.hand_ref = list(vr.right_pos)
                self.state.hand_rot_ref = list(vr.right_rot)
                self.state.robot_ref = list(robot.pose_mm_deg)
                self.state.last_target = list(robot.pose_mm_deg)
            self.state.filtered_hand_pos = list(vr.right_pos)
            self.state.filtered_hand_rot = list(vr.right_rot)

        assert self.state.hand_ref is not None
        assert self.state.robot_ref is not None
        assert self.state.hand_rot_ref is not None
        assert self.state.filtered_hand_pos is not None
        assert self.state.filtered_hand_rot is not None

        position_alpha = _smoothing_alpha(dt_s, self.position_smoothing_tau_s)
        rotation_alpha = _smoothing_alpha(dt_s, self.rotation_smoothing_tau_s)
        self.state.filtered_hand_pos = [
            self.state.filtered_hand_pos[index]
            + (vr.right_pos[index] - self.state.filtered_hand_pos[index]) * position_alpha
            for index in range(3)
        ]
        self.state.filtered_hand_rot = _quat_nlerp(
            self.state.filtered_hand_rot,
            vr.right_rot,
            rotation_alpha,
        )

        hand_delta = [
            _apply_deadband(
                self.state.filtered_hand_pos[index] - self.state.hand_ref[index],
                self.position_deadband_m,
            )
            for index in range(3)
        ]
        raw_target = list(self.state.robot_ref)
        raw_target[0] += _axis_value(hand_delta, self.robot_x_from) * self.scale
        raw_target[1] += _axis_value(hand_delta, self.robot_y_from) * self.scale
        raw_target[2] += _axis_value(hand_delta, self.robot_z_from) * self.scale

        if self.lock_orientation or not self.enable_orientation:
            # Keep the calibrated TCP orientation fixed. The adapter separately
            # enforces the absolute J6 joint angle in ServoJ lock mode.
            raw_target[3:6] = self.expected_robot_pose[3:6]
        else:
            hand_rotation = _relative_rotation_vector_deg(
                self.state.filtered_hand_rot,
                self.state.hand_rot_ref,
            )
            if self.lock_controller_twist:
                twist_index = {"x": 0, "y": 1, "z": 2}[
                    self.controller_twist_axis
                ]
                hand_rotation[twist_index] = 0.0
            hand_rotation = [
                _apply_deadband(value, self.rotation_deadband_deg)
                for value in hand_rotation
            ]
            raw_target[3] += _axis_value(hand_rotation, self.robot_rx_from) * self.rotation_scale
            raw_target[4] += _axis_value(hand_rotation, self.robot_ry_from) * self.rotation_scale
            raw_target[5] += _axis_value(hand_rotation, self.robot_rz_from) * self.rotation_scale

        if self.state.last_target is not None and self.target_smoothing_tau_s > 0.0:
            alpha = _smoothing_alpha(dt_s, self.target_smoothing_tau_s)
            smoothed_target = [
                self.state.last_target[index]
                + (raw_target[index] - self.state.last_target[index]) * alpha
                for index in range(3)
            ]
            for index in range(3, 6):
                delta = _wrapped_angle_delta(raw_target[index], self.state.last_target[index])
                smoothed_target.append(
                    _wrapped_angle_delta(
                        self.state.last_target[index] + delta * alpha,
                        0.0,
                    )
                )
            raw_target = smoothed_target

        # Advance from the last safe streamed command instead of restarting
        # every cycle from delayed robot feedback. A separate lead limiter
        # below prevents the command from running away from the real robot.
        step_reference = self.state.last_target or robot.pose_mm_deg
        limited = self.safety.limit_step(step_reference, raw_target, dt_s)
        limited = self.safety.limit_command_lead(robot.pose_mm_deg, limited)
        self.state.last_target = limited
        return TeleopCommand(True, target_pose_mm_deg=limited, reason="ok", clutched=True)

    def _initial_pose_mismatch(self, actual: Pose6) -> str:
        position_error = max(abs(actual[index] - self.expected_robot_pose[index]) for index in range(3))
        orientation_error = max(
            abs(_wrapped_angle_delta(actual[index], self.expected_robot_pose[index]))
            for index in range(3, 6)
        )
        if position_error > self.calibration_position_tolerance_mm:
            return f"calibration_robot_position_mismatch:{position_error:.1f}mm"
        if orientation_error > self.calibration_orientation_tolerance_deg:
            return f"calibration_robot_orientation_mismatch:{orientation_error:.1f}deg"
        return ""


def _wrapped_angle_delta(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0
