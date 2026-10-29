from __future__ import annotations

import math
import time
from typing import Any

from master_teleop.control.button_controller import MasterButtonController, MasterButtonEvent
from master_teleop.control.joint_mapper import RelativeJointMapper
from vr_teleop.control.models import TeleopCommand
from vr_teleop.core.app import TeleopRuntimeConfig
from vr_teleop.devices.camera import CameraFrame
from vr_teleop.devices.dobot_adapter import BaseRobotAdapter
from vr_teleop.devices.gripper_controller import TriggerGripperController
from vr_teleop.recording.dataset_recorder import EpisodeRecorder, RecorderEvent


class MasterHandTeleopApp:
    """Single USB master hand to single CR5A control loop."""

    def __init__(
        self,
        master: Any,
        robot: BaseRobotAdapter,
        mapper: RelativeJointMapper,
        runtime: TeleopRuntimeConfig,
        buttons: MasterButtonController,
        recorder: EpisodeRecorder | None = None,
        gripper: TriggerGripperController | None = None,
        camera: Any = None,
        recording_status_window: Any = None,
        sync_trace: Any = None,
        predict_target_workspace: bool = False,
    ) -> None:
        self.master = master
        self.robot = robot
        self.mapper = mapper
        self.runtime = runtime
        self.buttons = buttons
        self.recorder = recorder
        self.gripper = gripper
        self.camera = camera
        self.recording_status_window = recording_status_window
        self.sync_trace = sync_trace
        self.predict_target_workspace = bool(predict_target_workspace)
        self._was_commanding = False
        self._last_hold_reason = ""
        self._last_log = 0.0
        self._last_camera_preview = 0.0
        self._alignment_ready = False
        self._dynamic_approach_active = False
        self._unlock_after_approach = False
        # Connecting the Modbus gripper runs its initialization cycle. Defer
        # that physical action until long A explicitly starts alignment.
        self._gripper_connected = False
        self._last_fk_duration_s = 0.0
        self._last_predicted_pose_mm_deg: list[float] | None = None

    def run_forever(self) -> None:
        try:
            self.master.connect()
            self.master.set_locked(self.buttons.state.master_locked)
            if self.camera is not None:
                self.camera.start()
            if self.recording_status_window is not None:
                self.recording_status_window.show(
                    bool(self.recorder is not None and self.recorder.recording)
                )
            self.robot.connect()
            if self.runtime.execute and self.runtime.enable_robot:
                enable_robot = getattr(self.robot, "enable_robot", None)
                if callable(enable_robot):
                    enable_robot()
            if self.runtime.execute and self.runtime.speed_factor_percent is not None:
                self.robot.set_speed_factor(self.runtime.speed_factor_percent)
                print(f"Dobot SpeedFactor set to {self.runtime.speed_factor_percent}%")
            if self.gripper is not None:
                print(
                    "Follower gripper is idle; initialization and position alignment "
                    "will start only after long-press A.",
                    flush=True,
                )
            print(
                "Single-master teleoperation ready. Short-press A toggles the leader "
                "lock; long-press A starts/stops X-trainer dynamic approach and "
                "follower servo; B short=record start/stop; "
                f"execute={self.runtime.execute}. Leader starts locked. Recommended "
                "sequence: long A once -> wait for [master aligned] -> short A once. "
                f"Predictive FK workspace guard={self.predict_target_workspace}.",
                flush=True,
            )
            last = time.monotonic()
            next_tick = last
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.monotonic()
                dt_s = now - last
                last = now
                self.tick(dt_s)
                next_tick += self.runtime.servo_period_s
                finished = time.monotonic()
                if next_tick <= finished:
                    next_tick = finished + self.runtime.servo_period_s
        except KeyboardInterrupt:
            print("Stopping single-master teleoperation...")
        finally:
            if self.runtime.execute and self._was_commanding:
                self.robot.stop()
            self._lock_master_for_exit()
            if self.gripper is not None:
                self.gripper.close()
                self._gripper_connected = False
            if self.recording_status_window is not None:
                self.recording_status_window.close()
            if self.camera is not None:
                self.camera.stop()
            if self.recorder is not None:
                event = self.recorder.close()
                if event is not None:
                    self._print_recorder_event(event)
            if self.sync_trace is not None:
                self.sync_trace.close()
            self.robot.close()

    def _lock_master_for_exit(self) -> None:
        """Hold the leader's current pose after this process closes its serial port."""
        try:
            self.master.set_locked(True)
            print(
                "Leader locked at current pose; torque remains enabled after exit.",
                flush=True,
            )
        except Exception as exc:
            print(f"[master warning] could not lock leader during exit: {exc}", flush=True)
        try:
            self.master.close(release_torque=False)
        except TypeError:
            # Compatibility with a third-party/mock master exposing close() only.
            self.master.close()

    def tick(self, dt_s: float) -> None:
        sample = self.master.latest()
        state = self.robot.state()
        button_state = self.buttons.update(
            sample,
            time.monotonic(),
            recording=bool(self.recorder is not None and self.recorder.recording),
        )
        for event in button_state.events:
            self._handle_button_event(event, sample)
        if button_state.events:
            # Refresh asynchronous inputs after button handling so safety checks
            # use the newest master and robot feedback.
            sample = self.master.latest()
            state = self.robot.state()
        self._update_gripper(sample, button_state.servo_enabled)
        camera_frames = self._latest_camera_frames()

        # Match X-trainer: long A runs a locked, rate-limited approach. Once
        # aligned, ordinary following runs only while the logical leader lock
        # is released by short A.
        mapping_active = bool(
            button_state.servo_enabled
            and (
                self._dynamic_approach_active
                or (self._alignment_ready and not button_state.master_locked)
            )
        )
        command = self.mapper.map(sample, state, dt_s, active=mapping_active)
        if command.allowed and command.target_joint_positions_deg is not None:
            self._apply_predictive_workspace_guard(command)
        if self.sync_trace is not None:
            self.sync_trace.record_tick(
                sample=sample,
                robot=state,
                command=command,
                mapper=self.mapper,
                button_state=button_state,
                alignment_ready=self._alignment_ready,
                approach_active=self._dynamic_approach_active,
                execute=self.runtime.execute,
                force=bool(button_state.events),
            )
        if not command.allowed or command.target_joint_positions_deg is None:
            self._record(sample, state, command, dt_s, camera_frames)
            hold_reason = command.reason
            if (
                hold_reason == "servo_disabled"
                and button_state.servo_enabled
                and self._alignment_ready
                and button_state.master_locked
            ):
                hold_reason = (
                    "waiting_for_short_a_unlock: follower servo is ON; "
                    "short-press A once to follow (do not long-press A again)"
                )
            reason_key = hold_reason.split(":", 1)[0]
            if reason_key != self._last_hold_reason:
                print(f"[master hold] {hold_reason}", flush=True)
                self._last_hold_reason = reason_key
            if self.runtime.execute and self._was_commanding:
                self.robot.stop()
            self._was_commanding = False
            self._show_previews()
            return

        self._last_hold_reason = ""
        if self.mapper.aligned and self._dynamic_approach_active:
            if not self._complete_dynamic_approach():
                if self.runtime.execute and self._was_commanding:
                    self.robot.stop()
                self._was_commanding = False
                self._show_previews()
                return
        if self.runtime.execute:
            self.robot.servo_j(
                command.target_joint_positions_deg,
                t=self.runtime.servo_t_s,
                aheadtime=self.runtime.servo_aheadtime,
                gain=self.runtime.servo_gain,
            )
            self._was_commanding = True
        self._record(sample, state, command, dt_s, camera_frames)
        now = time.monotonic()
        if now - self._last_log >= 0.5:
            source = ", ".join(f"{value:+.2f}" for value in sample.joint_positions_rad)
            target = ", ".join(f"{value:+.1f}" for value in command.target_joint_positions_deg)
            desired_values = self.mapper.desired_target_deg or []
            desired = ", ".join(f"{value:+.1f}" for value in desired_values)
            mode = "EXEC" if self.runtime.execute else "DRY"
            phase = "APPROACH" if self._dynamic_approach_active else "FOLLOW"
            tracking_error = max(
                (
                    abs(target_value - actual_value)
                    for target_value, actual_value in zip(
                        desired_values,
                        state.q_actual_deg,
                    )
                ),
                default=0.0,
            )
            print(
                f"[master {mode} {phase}] q_master_rad=[{source}] "
                f"q_desired_deg=[{desired}] q_command_deg=[{target}] "
                f"tracking_error_deg={tracking_error:.2f} "
                f"fk_ms={self._last_fk_duration_s * 1000.0:.1f}"
            )
            self._last_log = now
        self._show_previews()

    def _apply_predictive_workspace_guard(self, command: TeleopCommand) -> None:
        """Fail closed when the controller predicts an out-of-box TCP target."""
        if not self.predict_target_workspace:
            self._last_fk_duration_s = 0.0
            self._last_predicted_pose_mm_deg = None
            return
        assert command.target_joint_positions_deg is not None
        started = time.monotonic()
        try:
            pose = self.robot.positive_kinematics(command.target_joint_positions_deg)
        except Exception as exc:
            self._last_fk_duration_s = time.monotonic() - started
            self._last_predicted_pose_mm_deg = None
            command.allowed = False
            command.reason = f"positive_kinematics_failed:{exc}"
            return
        self._last_fk_duration_s = time.monotonic() - started
        self._last_predicted_pose_mm_deg = list(pose)
        command.target_pose_mm_deg = list(pose)
        if not self.mapper.safety.pose_in_workspace(pose):
            xyz = ",".join(f"{value:.1f}" for value in pose[:3])
            command.allowed = False
            command.reason = f"predicted_target_pose_outside_workspace:{xyz}"

    def _handle_button_event(self, event: MasterButtonEvent, sample=None) -> None:
        sync_trace = getattr(self, "sync_trace", None)
        if sync_trace is not None:
            sync_trace.record_button_event(
                event,
                sample,
                self.robot.state(),
                frozen_target_deg=self.mapper.soft_alignment_target_deg,
                alignment_ready=self._alignment_ready,
                approach_active=self._dynamic_approach_active,
            )
        if event.action == "master_lock":
            desired_locked = bool(event.value)
            if self._dynamic_approach_active:
                self._unlock_after_approach = not desired_locked
                print(
                    f"[button A short {event.duration_s:.2f}s] leader "
                    f"{'LOCK' if desired_locked else 'UNLOCK'} queued until "
                    "dynamic approach completes",
                    flush=True,
                )
                return
            try:
                self.master.set_locked(desired_locked)
                state = "LOCKED" if desired_locked else "UNLOCKED"
                print(
                    f"[button A short {event.duration_s:.2f}s] leader {state}",
                    flush=True,
                )
            except Exception as exc:
                self.buttons.force_servo_off()
                print(f"[master hold] leader torque change failed: {exc}", flush=True)
        elif event.action == "servo":
            state = "DYNAMIC APPROACH/START" if event.value else "STOP"
            print(f"[button A long {event.duration_s:.2f}s] follower {state}", flush=True)
            self._alignment_ready = False
            self._dynamic_approach_active = False
            self._unlock_after_approach = False
            self.mapper.end_soft_alignment()
            if event.value:
                self._unlock_after_approach = not self.buttons.state.master_locked
                try:
                    robot_state = self.robot.state()
                    self._require_alignment_ready(sample, robot_state)
                    # X-trainer holds the leader rigid for dynamic_approach(),
                    # without changing the short-A logical lock state.
                    self.master.set_locked(True)
                    self.mapper.begin_soft_alignment(sample, robot_state)
                    self._require_frozen_target_workspace()
                    if sync_trace is not None:
                        sync_trace.write(
                            "alignment_target_frozen",
                            {
                                "frozen_target_deg": self.mapper.soft_alignment_target_deg,
                                "robot_joint_actual_deg": robot_state.q_actual_deg,
                            },
                        )
                    # Do not initialize or move the independent Modbus gripper
                    # until arm feedback and the frozen six-joint target pass.
                    self._align_gripper_to_master(sample)
                    self._dynamic_approach_active = True
                    self._print_alignment_target(sample)
                    print(
                        "[master dynamic approach] leader physically LOCKED; follower "
                        "gripper and six joints are approaching the current leader pose",
                        flush=True,
                    )
                except Exception as exc:
                    self._dynamic_approach_active = False
                    self._unlock_after_approach = False
                    self.mapper.end_soft_alignment()
                    self.buttons.force_servo_off()
                    if sync_trace is not None:
                        sync_trace.write(
                            "alignment_start_failed",
                            {"error": str(exc)},
                        )
                    print(
                        f"[master hold] gripper/joint dynamic approach start "
                        f"failed: {exc}",
                        flush=True,
                    )
                    return
            if not event.value:
                try:
                    self.master.set_locked(self.buttons.state.master_locked)
                except Exception as exc:
                    print(
                        f"[master hold] failed to restore leader lock state: {exc}",
                        flush=True,
                    )
                if self.runtime.execute and self._was_commanding:
                    self.robot.stop()
                    self._was_commanding = False
        elif event.action == "record":
            if self.recorder is None:
                print("[button B short] recording is disabled; use --record-data", flush=True)
            elif self.recorder.recording:
                self._print_recorder_event(self.recorder.stop_episode("button_b_short"))
            elif not self.buttons.state.servo_enabled:
                print("[button B short] ignored until follower servo is started", flush=True)
            else:
                self._print_recorder_event(self.recorder.start_episode())
        elif event.action == "a_ignored":
            print(
                f"[button A {event.duration_s:.2f}s] ignored: use <0.5s or >1.0s",
                flush=True,
            )
        elif event.action == "b_ignored":
            print(f"[button B {event.duration_s:.2f}s] ignored: use a short press", flush=True)

    def _complete_dynamic_approach(self) -> bool:
        """Finish X-trainer approach and apply the queued leader lock state."""
        desired_locked = not self._unlock_after_approach
        try:
            self.master.set_locked(desired_locked)
        except Exception as exc:
            self.buttons.state.master_locked = True
            self.buttons.force_servo_off()
            self._dynamic_approach_active = False
            self._unlock_after_approach = False
            self.mapper.end_soft_alignment()
            print(
                f"[master hold] leader lock restore after approach failed: {exc}",
                flush=True,
            )
            return False
        self.buttons.state.master_locked = desired_locked
        self._dynamic_approach_active = False
        self._unlock_after_approach = False
        self.mapper.end_soft_alignment()
        self._alignment_ready = True
        if desired_locked:
            print(
                "[master aligned] follower reached the leader pose; leader remains "
                "LOCKED and follower servo remains ON. NEXT: short-press A once to "
                "unlock and follow. Do not long-press A again, because that stops servo",
                flush=True,
            )
        else:
            print(
                "[master aligned] follower reached the leader pose; leader UNLOCKED "
                "and fast following started",
                flush=True,
            )
        return True

    def _require_alignment_ready(self, sample, robot_state) -> None:
        decision = self.mapper.safety.check(
            sample,
            robot_state,
            require_deadman=False,
            source_name="master_hand",
        )
        if not decision.allowed:
            raise RuntimeError(decision.reason)

    def _require_frozen_target_workspace(self) -> None:
        target = self.mapper.soft_alignment_target_deg
        if target is None:
            raise RuntimeError("alignment_target_is_unavailable")
        pose = self.robot.positive_kinematics(target)
        if not self.mapper.safety.pose_in_workspace(pose):
            xyz = ",".join(f"{value:.1f}" for value in pose[:3])
            raise RuntimeError(
                f"alignment_target_pose_outside_workspace:{xyz}"
            )

    def _update_gripper(self, sample, servo_enabled: bool) -> None:
        if self.gripper is None or not self._gripper_is_connected():
            return
        # Match X-trainer dynamic_approach(): the gripper follows the locked
        # leader while the six arm joints are converging, then keeps following
        # during ordinary unlocked teleoperation.
        valid = (
            sample is not None
            and sample.tracking_valid
            and servo_enabled
            and (
                self._dynamic_approach_active
                or (self._alignment_ready and not self.buttons.state.master_locked)
            )
        )
        trigger = sample.right_trigger if valid else 0.0
        self.gripper.update(trigger, input_valid=valid)

    def _align_gripper_to_master(self, sample) -> None:
        """Start follower gripper alignment from the locked leader's current pose."""
        if self.gripper is None:
            return
        if sample is None or not sample.tracking_valid:
            raise RuntimeError("master gripper sample is unavailable")

        self._ensure_gripper_connected()

        leader_open = min(max(float(sample.gripper_open), 0.0), 1.0)
        desired_closed = leader_open < 0.5
        close_trigger = 1.0 if self.gripper.config.pressed_closes else 0.0
        open_trigger = 0.0 if self.gripper.config.pressed_closes else 1.0
        trigger = close_trigger if desired_closed else open_trigger
        target = "CLOSE" if desired_closed else "OPEN"
        print(
            f"[master gripper align] leader_open={leader_open:.3f} -> {target}; "
            "motion continues concurrently with six-joint approach",
            flush=True,
        )
        self.gripper.update(
            trigger,
            input_valid=True,
            allow_initial_open=True,
        )

    def _gripper_is_connected(self) -> bool:
        # object.__new__ based unit-test fixtures predate lazy connection and
        # represent an already connected fake gripper.
        return bool(getattr(self, "_gripper_connected", True))

    def _ensure_gripper_connected(self) -> None:
        if self.gripper is None or self._gripper_is_connected():
            return
        try:
            self.gripper.connect(getattr(self.robot, "dashboard", None))
        except Exception:
            self.gripper.close()
            self._gripper_connected = False
            raise
        self._gripper_connected = True

    def _print_alignment_target(self, sample) -> None:
        if sample is None or len(sample.joint_positions_rad) != 6:
            return
        state = self.robot.state()
        desired = self.mapper.soft_alignment_target_deg
        if desired is None:
            desired = [math.degrees(value) for value in sample.joint_positions_rad]
        actual = list(state.q_actual_deg)
        if len(actual) != 6:
            return
        worst = max(abs(target - current) for target, current in zip(desired, actual))
        desired_text = ", ".join(f"{value:+.1f}" for value in desired)
        actual_text = ", ".join(f"{value:+.1f}" for value in actual)
        print(
            f"[master alignment target] leader_deg=[{desired_text}] "
            f"follower_deg=[{actual_text}] max_delta_deg={worst:.1f}",
            flush=True,
        )
        if worst <= self.mapper.soft_alignment_tolerance_deg:
            print(
                "[master alignment already satisfied] leader and follower joint "
                "coordinates are already within the configured tolerance; no initial "
                "joint motion is required",
                flush=True,
            )

    def _record(self, sample, state, command, dt_s, camera_frames) -> None:
        if self.recorder is None:
            return
        self.recorder.record(
            sample,
            state,
            command,
            dt_s,
            self.runtime.execute,
            gripper_state=self.gripper.snapshot() if self.gripper is not None else None,
            camera_frames=camera_frames,
        )

    def _latest_camera_frames(self) -> dict[str, CameraFrame | None]:
        if self.camera is None:
            return {}
        self.camera.raise_if_failed()
        latest_frames = getattr(self.camera, "latest_frames", None)
        if callable(latest_frames):
            return latest_frames()
        return {self.camera.config.name: self.camera.latest()}

    def _show_previews(self) -> None:
        if self.camera is None and self.recording_status_window is None:
            return
        fps = max(float(getattr(getattr(self.camera, "config", None), "fps", 30)), 1.0)
        now = time.monotonic()
        if now - self._last_camera_preview >= 1.0 / fps:
            if self.recording_status_window is not None:
                self.recording_status_window.show(
                    bool(self.recorder is not None and self.recorder.recording)
                )
            show = getattr(self.camera, "show_previews", None)
            if callable(show):
                show()
            self._last_camera_preview = now

    @staticmethod
    def _print_recorder_event(event: RecorderEvent) -> None:
        if event.action == "started":
            print(f"[record] START {event.episode_dir}")
        elif event.action == "stopped":
            validity = "valid" if event.valid else "too_short_or_incomplete"
            print(
                f"[record] STOP frames={event.frame_count} status={validity} "
                f"path={event.episode_dir}"
            )
