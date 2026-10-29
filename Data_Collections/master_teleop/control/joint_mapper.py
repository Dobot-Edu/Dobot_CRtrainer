from __future__ import annotations

import math

from vr_teleop.control.models import RobotState, TeleopCommand
from vr_teleop.control.safety_supervisor import SafetySupervisor

from .models import MasterHandInput


class RelativeJointMapper:
    """Map one physical master hand to one CR5A in absolute or relative joint space."""

    def __init__(self, config: dict, safety: SafetySupervisor) -> None:
        master = config.get("master_hand", {})
        mapping = master.get("mapping", {})
        self.safety = safety
        self.watchdog_timeout_s = float(master.get("watchdog_timeout_s", 0.10))
        self.mode = str(mapping.get("mode", "absolute")).strip().lower()
        self.joint_signs = _six(mapping.get("joint_signs", [1] * 6), "joint_signs")
        self.joint_scales = _six(mapping.get("joint_scales", [1] * 6), "joint_scales")
        self.joint_min_deg = _six(mapping.get("joint_min_deg", [-180] * 6), "joint_min_deg")
        self.joint_max_deg = _six(mapping.get("joint_max_deg", [180] * 6), "joint_max_deg")
        self.max_step_deg = float(mapping.get("max_step_deg", 1.0))
        self.max_speed_deg_s = float(mapping.get("max_speed_deg_s", 30.0))
        self.max_command_lead_deg = float(mapping.get("max_command_lead_deg", 5.0))
        self.require_hold_button_a = bool(mapping.get("require_hold_button_a", True))
        self.alignment_tolerance_deg = float(mapping.get("alignment_tolerance_deg", 1.0))
        self.soft_alignment_max_step_deg = float(
            mapping.get("soft_alignment_max_step_deg", 0.035)
        )
        self.soft_alignment_max_speed_deg_s = float(
            mapping.get("soft_alignment_max_speed_deg_s", 1.4)
        )
        self.soft_alignment_tolerance_deg = float(
            mapping.get("soft_alignment_tolerance_deg", 0.2)
        )
        self.pre_alignment_master_settle_s = float(
            mapping.get("pre_alignment_master_settle_s", 0.30)
        )
        self.max_initial_alignment_delta_deg = float(
            mapping.get("max_initial_alignment_delta_deg", math.degrees(0.6))
        )
        self.enforce_initial_alignment_delta = bool(
            mapping.get("enforce_initial_alignment_delta", True)
        )
        if self.mode not in ("absolute", "relative"):
            raise ValueError("master_hand.mapping.mode must be absolute or relative")
        if any(low >= high for low, high in zip(self.joint_min_deg, self.joint_max_deg)):
            raise ValueError("master hand joint minimums must be below maximums")
        if min(
            self.max_step_deg,
            self.max_speed_deg_s,
            self.max_command_lead_deg,
            self.alignment_tolerance_deg,
            self.soft_alignment_max_step_deg,
            self.soft_alignment_max_speed_deg_s,
            self.soft_alignment_tolerance_deg,
            self.max_initial_alignment_delta_deg,
        ) <= 0.0:
            raise ValueError("master hand joint safety limits must be positive")
        if self.pre_alignment_master_settle_s < 0.0:
            raise ValueError("pre_alignment_master_settle_s must be non-negative")
        self._master_reference_rad: list[float] | None = None
        self._robot_reference_deg: list[float] | None = None
        self._last_target_deg: list[float] | None = None
        self._deadman_was_pressed = False
        self._active_was_enabled = False
        self.desired_target_deg: list[float] | None = None
        self.aligned = False
        self._alignment_started = False
        self.soft_alignment_active = False
        self._soft_alignment_target_deg: list[float] | None = None

    @property
    def soft_alignment_target_deg(self) -> list[float] | None:
        return (
            None
            if self._soft_alignment_target_deg is None
            else list(self._soft_alignment_target_deg)
        )

    def begin_soft_alignment(
        self,
        sample: MasterHandInput | None = None,
        robot: RobotState | None = None,
    ) -> None:
        """Freeze a leader target and use gentle limits until it is reached."""
        self.soft_alignment_active = True
        self.aligned = False
        self._alignment_started = False
        self._active_was_enabled = False
        self._last_target_deg = None
        self._soft_alignment_target_deg = None
        if sample is not None and robot is not None:
            self._soft_alignment_target_deg = self._absolute_target_deg(sample, robot)
            self._validate_initial_alignment_target(
                self._soft_alignment_target_deg,
                robot.q_actual_deg,
            )

    def end_soft_alignment(self) -> None:
        self.soft_alignment_active = False
        self._soft_alignment_target_deg = None

    def release_clutch(self) -> None:
        self._master_reference_rad = None
        self._robot_reference_deg = None
        self._last_target_deg = None
        self._deadman_was_pressed = False
        self._active_was_enabled = False
        self.desired_target_deg = None
        self.aligned = False
        self._alignment_started = False

    def map(
        self,
        sample: MasterHandInput | None,
        robot: RobotState,
        dt_s: float,
        active: bool = True,
    ) -> TeleopCommand:
        if not active:
            self.release_clutch()
            return self._hold("servo_disabled")
        decision = self.safety.check(
            sample,
            robot,
            require_deadman=self.require_hold_button_a,
            source_name="master_hand",
        )
        if not decision.allowed:
            self.release_clutch()
            return self._hold(decision.reason)
        assert sample is not None
        if sample.age_s > self.watchdog_timeout_s:
            self.release_clutch()
            return self._hold(f"master_hand_watchdog_timeout:{sample.age_s:.3f}s")
        if len(sample.joint_positions_rad) != 6 or not all(
            math.isfinite(value) for value in sample.joint_positions_rad
        ):
            self.release_clutch()
            return self._hold("invalid_master_joint_sample")
        if len(robot.q_actual_deg) != 6 or not all(
            math.isfinite(value) for value in robot.q_actual_deg
        ):
            self.release_clutch()
            return self._hold("waiting_for_robot_joint_feedback")
        rising_active = not self._active_was_enabled
        self._active_was_enabled = True
        rising_deadman = sample.button_a and not self._deadman_was_pressed
        self._deadman_was_pressed = sample.button_a
        if rising_active or rising_deadman or self._last_target_deg is None:
            self._master_reference_rad = list(sample.joint_positions_rad)
            self._robot_reference_deg = list(robot.q_actual_deg)
            self._last_target_deg = list(robot.q_actual_deg)

        assert self._master_reference_rad is not None
        assert self._robot_reference_deg is not None
        assert self._last_target_deg is not None
        if self.soft_alignment_active and self._soft_alignment_target_deg is not None:
            # X-trainer snapshots agent.act() once before dynamic_approach().
            # Do not let an unlocked/noisy leader turn pre-alignment into live follow.
            desired = list(self._soft_alignment_target_deg)
        elif self.mode == "absolute":
            desired = self._absolute_target_deg(sample, robot)
        else:
            desired = []
            for master, reference, robot_reference, sign, scale in zip(
                sample.joint_positions_rad,
                self._master_reference_rad,
                self._robot_reference_deg,
                self.joint_signs,
                self.joint_scales,
            ):
                delta_deg = math.degrees(_wrapped_radian_delta(master, reference))
                desired.append(robot_reference + delta_deg * sign * scale)
        self.desired_target_deg = list(desired)

        for index, (value, low, high) in enumerate(
            zip(desired, self.joint_min_deg, self.joint_max_deg), start=1
        ):
            if value < low or value > high:
                return self._hold(
                    f"joint_limit:J{index}={value:.2f} outside [{low:.2f},{high:.2f}]"
                )

        if not self._alignment_started:
            deltas = [
                abs(target - actual)
                for target, actual in zip(desired, robot.q_actual_deg)
            ]
            worst_index = max(range(6), key=deltas.__getitem__)
            if (
                self.enforce_initial_alignment_delta
                and deltas[worst_index] > self.max_initial_alignment_delta_deg
            ):
                self._last_target_deg = list(robot.q_actual_deg)
                return self._hold(
                    "alignment_mismatch:"
                    f"J{worst_index + 1} delta={deltas[worst_index]:.1f}deg "
                    f"exceeds {self.max_initial_alignment_delta_deg:.1f}deg"
                )
            self._alignment_started = True
            self._last_target_deg = list(robot.q_actual_deg)

        configured_step = (
            self.soft_alignment_max_step_deg
            if self.soft_alignment_active
            else self.max_step_deg
        )
        configured_speed = (
            self.soft_alignment_max_speed_deg_s
            if self.soft_alignment_active
            else self.max_speed_deg_s
        )
        max_step = min(configured_step, configured_speed * max(dt_s, 0.001))
        limited = [
            previous + min(max(target - previous, -max_step), max_step)
            for previous, target in zip(self._last_target_deg, desired)
        ]
        lead_limited = [
            actual + min(
                max(target - actual, -self.max_command_lead_deg),
                self.max_command_lead_deg,
            )
            for actual, target in zip(robot.q_actual_deg, limited)
        ]
        self._last_target_deg = lead_limited
        alignment_tolerance = (
            self.soft_alignment_tolerance_deg
            if self.soft_alignment_active
            else self.alignment_tolerance_deg
        )
        self.aligned = max(
            abs(target - actual) for target, actual in zip(desired, robot.q_actual_deg)
        ) <= alignment_tolerance
        target_pose = (
            list(robot.target_pose_mm_deg)
            if len(robot.target_pose_mm_deg) == 6
            else list(robot.pose_mm_deg)
        )
        return TeleopCommand(
            allowed=True,
            target_pose_mm_deg=target_pose,
            target_joint_positions_deg=lead_limited,
            reason="ok" if self.aligned else "aligning",
            clutched=True,
            control_source="master_hand",
            control_mode="servo_j",
        )

    def _absolute_target_deg(
        self,
        sample: MasterHandInput,
        robot: RobotState,
    ) -> list[float]:
        raw_desired = [
            math.degrees(master) * sign * scale
            for master, sign, scale in zip(
                sample.joint_positions_rad,
                self.joint_signs,
                self.joint_scales,
            )
        ]
        anchors = (
            robot.q_actual_deg
            if not self._alignment_started or self._last_target_deg is None
            else self._last_target_deg
        )
        return [
            _closest_equivalent_deg(target, anchor)
            for target, anchor in zip(raw_desired, anchors)
        ]

    def _validate_initial_alignment_target(
        self,
        desired: list[float],
        actual: list[float],
    ) -> None:
        if len(actual) != 6 or not all(math.isfinite(value) for value in actual):
            raise RuntimeError("waiting_for_robot_joint_feedback")
        for index, (value, low, high) in enumerate(
            zip(desired, self.joint_min_deg, self.joint_max_deg), start=1
        ):
            if value < low or value > high:
                raise RuntimeError(
                    f"joint_limit:J{index}={value:.2f} outside [{low:.2f},{high:.2f}]"
                )
        deltas = [abs(target - current) for target, current in zip(desired, actual)]
        worst_index = max(range(6), key=deltas.__getitem__)
        if (
            self.enforce_initial_alignment_delta
            and deltas[worst_index] > self.max_initial_alignment_delta_deg
        ):
            raise RuntimeError(
                "alignment_mismatch:"
                f"J{worst_index + 1} delta={deltas[worst_index]:.1f}deg "
                f"exceeds {self.max_initial_alignment_delta_deg:.1f}deg"
            )

    @staticmethod
    def _hold(reason: str) -> TeleopCommand:
        return TeleopCommand(
            allowed=False,
            reason=reason,
            control_source="master_hand",
            control_mode="servo_j",
        )


def _six(values, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 6 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"master_hand.mapping.{name} must contain six finite values")
    return result


def _wrapped_radian_delta(target: float, reference: float) -> float:
    return (target - reference + math.pi) % (2.0 * math.pi) - math.pi


def _closest_equivalent_deg(target: float, reference: float) -> float:
    return target + 360.0 * round((reference - target) / 360.0)
