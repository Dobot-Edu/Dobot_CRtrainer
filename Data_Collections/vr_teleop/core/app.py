from __future__ import annotations

import time
from dataclasses import dataclass

from ..control.mapper import RelativeVRMapper
from ..devices.camera import CameraFrame, RealSenseCamera
from ..devices.dobot_adapter import BaseRobotAdapter
from ..devices.gripper_controller import TriggerGripperController
from ..network.vr_udp_receiver import VRUDPReceiver
from ..recording.dataset_recorder import EpisodeRecorder, RecorderEvent


@dataclass
class TeleopRuntimeConfig:
    servo_period_s: float
    servo_t_s: float
    servo_aheadtime: float
    servo_gain: float
    speed_factor_percent: int | None = None
    execute: bool = False
    enable_robot: bool = False
    auto_restore_after_episode: bool = False
    initial_pose_mm_deg: list[float] | None = None
    restore_acceleration_percent: int = 10
    restore_velocity_percent: int = 10
    restore_timeout_s: float = 60.0
    restore_position_tolerance_mm: float = 1.5
    restore_orientation_tolerance_deg: float = 2.0


class TeleopApp:
    def __init__(
        self,
        receiver: VRUDPReceiver,
        robot: BaseRobotAdapter,
        mapper: RelativeVRMapper,
        runtime: TeleopRuntimeConfig,
        recorder: EpisodeRecorder | None = None,
        gripper: TriggerGripperController | None = None,
        camera: RealSenseCamera | None = None,
    ) -> None:
        self.receiver = receiver
        self.robot = robot
        self.mapper = mapper
        self.runtime = runtime
        self.recorder = recorder
        self.gripper = gripper
        self.camera = camera
        self._last_stop_reason = ""
        self._last_gripper_error = ""
        self._was_commanding = False
        self._last_servo_log_monotonic = 0.0
        self._last_camera_preview_monotonic = 0.0

    def run_forever(self) -> None:
        self.receiver.start()
        try:
            if self.camera is not None:
                try:
                    self.camera.start()
                except Exception as exc:
                    self.camera.stop()
                    if self.camera.config.required:
                        raise
                    print(f"[camera disabled] startup failed: {exc}", flush=True)
            self.robot.connect()
            if self.runtime.execute and self.runtime.enable_robot:
                enable_robot = getattr(self.robot, "enable_robot", None)
                if callable(enable_robot):
                    enable_robot()
            if self.runtime.execute and self.runtime.speed_factor_percent is not None:
                self.robot.set_speed_factor(self.runtime.speed_factor_percent)
                print(f"Dobot SpeedFactor set to {self.runtime.speed_factor_percent}%")
            if self.gripper is not None:
                try:
                    self.gripper.connect(getattr(self.robot, "dashboard", None))
                except Exception as exc:
                    # A gripper fault must not bring down the independent arm servo loop.
                    print(f"[gripper disabled] startup failed: {exc}", flush=True)
                    self.gripper.close()
            print(
                f"VR teleop listening on UDP {self.receiver.listen_ip}:{self.receiver.listen_port}. "
                f"execute={self.runtime.execute}"
            )
            if self.recorder is not None:
                print(
                    f"Data recording enabled. Press Quest {self.recorder.record_button} "
                    "once to start an episode and once again to stop it."
                )
                if self.runtime.auto_restore_after_episode:
                    print(
                        "Automatic initial-pose restore is enabled after each "
                        "recorded episode."
                    )
            last = time.monotonic()
            next_tick = last
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.monotonic()
                dt = now - last
                last = now
                self.tick(dt)
                next_tick += self.runtime.servo_period_s
                finished = time.monotonic()
                if next_tick <= finished:
                    next_tick = finished + self.runtime.servo_period_s
        except KeyboardInterrupt:
            print("Stopping teleop...")
        finally:
            if self.runtime.execute and self._was_commanding:
                self.robot.stop()
            if self.gripper is not None:
                self.gripper.close()
            if self.camera is not None:
                self.camera.stop()
            if self.recorder is not None:
                event = self.recorder.close()
                if event is not None:
                    self._print_recorder_event(event)
            self.receiver.stop()
            self.robot.close()

    def tick(self, dt_s: float) -> None:
        vr = self.receiver.latest()
        self._update_gripper(vr)
        camera_frames = self._latest_camera_frames()
        state = self.robot.state()
        recorder_event = self._handle_record_button(vr)
        if (
            recorder_event is not None
            and recorder_event.action == "stopped"
            and self.runtime.auto_restore_after_episode
        ):
            self._restore_after_episode()
            return

        cmd = self.mapper.map(vr, state, dt_s)
        if not cmd.allowed or cmd.target_pose_mm_deg is None:
            self._record_tick(vr, state, cmd, dt_s, camera_frames)
            reason_key = cmd.reason.split(":", 1)[0]
            if reason_key != self._last_stop_reason:
                print(f"[hold] {cmd.reason}")
                self._last_stop_reason = reason_key
            if self.runtime.execute and self._was_commanding:
                self.robot.stop()
            self._was_commanding = False
            self._update_camera_previews()
            return

        self._last_stop_reason = ""
        if self.runtime.execute:
            try:
                self.robot.servo_p(
                    cmd.target_pose_mm_deg,
                    t=self.runtime.servo_t_s,
                    aheadtime=self.runtime.servo_aheadtime,
                    gain=self.runtime.servo_gain,
                )
            except Exception as exc:
                # Keep the original traceback, but expose the robot state and
                # command that caused the first motion failure.
                print(
                    "[robot motion error] ServoP/ServoJ command was rejected: "
                    f"{exc}",
                    flush=True,
                )
                print(
                    "[robot motion error] "
                    f"mode={state.robot_mode} enabled={state.enabled} "
                    f"error={state.error_status} collision={state.collision_state} "
                    f"pose={state.pose_mm_deg}",
                    flush=True,
                )
                print(
                    "[robot motion error] target="
                    f"{cmd.target_pose_mm_deg} "
                    f"j6_lock={getattr(self.robot, 'lock_j6', None)}",
                    flush=True,
                )
                raise
            self._was_commanding = True
        self._record_tick(vr, state, cmd, dt_s, camera_frames)
        # Do not write to the Windows console at the servo rate. Console I/O
        # can block the 50 Hz loop and show up as physical motion jitter.
        now = time.monotonic()
        if now - self._last_servo_log_monotonic >= 0.5:
            pose_text = ", ".join(f"{v:.1f}" for v in cmd.target_pose_mm_deg)
            print(f"[servo{' EXEC' if self.runtime.execute else ' DRY'}] {pose_text}")
            self._last_servo_log_monotonic = now
        self._update_camera_previews()

    def _update_camera_previews(self) -> None:
        if self.camera is None:
            return
        show_previews = getattr(self.camera, "show_previews", None)
        if not callable(show_previews):
            return
        preview_fps = max(float(getattr(self.camera.config, "fps", 30)), 1.0)
        now = time.monotonic()
        if now - self._last_camera_preview_monotonic < 1.0 / preview_fps:
            return
        show_previews()
        self._last_camera_preview_monotonic = now

    def _record_tick(
        self,
        vr,
        state,
        cmd,
        dt_s: float,
        camera_frames: dict[str, CameraFrame | None],
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.record(
            vr,
            state,
            cmd,
            dt_s,
            self.runtime.execute,
            gripper_state=self.gripper.snapshot() if self.gripper is not None else None,
            camera_frames=camera_frames,
        )

    def _handle_record_button(self, vr) -> RecorderEvent | None:
        if self.recorder is None:
            return None
        event = self.recorder.handle_record_button(vr)
        if event is not None:
            self._print_recorder_event(event)
        return event

    def _restore_after_episode(self) -> None:
        if not self.runtime.auto_restore_after_episode:
            return

        target = self.runtime.initial_pose_mm_deg
        if target is None or len(target) != 6:
            raise RuntimeError(
                "Automatic restore requires a six-value initial_pose_mm_deg"
            )

        self.mapper.release_clutch()
        if not self.runtime.execute:
            print("[restore] DRY RUN: automatic initial-pose motion skipped.")
            return

        print(
            "[restore] Episode saved. Stopping VR servo and returning to "
            f"initial pose: {[round(value, 3) for value in target]}",
            flush=True,
        )
        if self._was_commanding:
            self.robot.stop()
            # Give the controller a short transition interval before issuing
            # the queued joint move that returns to the home pose.
            time.sleep(0.1)
        self._was_commanding = False

        state = self.robot.state()
        position_error, orientation_error = _pose_error(state.pose_mm_deg, target)
        if not _pose_reached(position_error, orientation_error, self.runtime):
            self.robot.move_j_pose(
                target,
                acceleration=self.runtime.restore_acceleration_percent,
                velocity=self.runtime.restore_velocity_percent,
            )

            deadline = time.monotonic() + self.runtime.restore_timeout_s
            next_status_log = 0.0
            while time.monotonic() < deadline:
                time.sleep(0.1)
                state = self.robot.state()
                position_error, orientation_error = _pose_error(
                    state.pose_mm_deg,
                    target,
                )
                if state.error_status or state.collision_state:
                    raise RuntimeError(
                        "Automatic restore stopped by robot state: "
                        f"error={state.error_status}, "
                        f"collision={state.collision_state}"
                    )
                if _pose_reached(position_error, orientation_error, self.runtime):
                    break
                now = time.monotonic()
                if now >= next_status_log:
                    print(
                        "[restore] moving "
                        f"position_error={position_error:.2f}mm "
                        f"orientation_error={orientation_error:.2f}deg",
                        flush=True,
                    )
                    next_status_log = now + 1.0
            else:
                self.robot.stop()
                raise RuntimeError(
                    "Automatic initial-pose restore timed out after "
                    f"{self.runtime.restore_timeout_s:.1f}s"
                )

        self.mapper.release_clutch()
        print(
            "[restore] Initial pose reached. Release Grip, reposition the "
            "controller, then press B to start the next episode.",
            flush=True,
        )

    def _update_gripper(self, vr) -> None:
        if self.gripper is None:
            return
        try:
            self.gripper.update(
                vr.right_trigger if vr is not None else 0.0,
                input_valid=vr is not None and bool(vr.tracking_valid),
            )
        except Exception as exc:
            message = str(exc)
            if message != self._last_gripper_error:
                print(f"[gripper error] {message}", flush=True)
                self._last_gripper_error = message

    def _latest_camera_frames(self) -> dict[str, CameraFrame | None]:
        if self.camera is None:
            return {}
        try:
            self.camera.raise_if_failed()
        except Exception as exc:
            if self.camera.config.required:
                raise
            print(f"[camera error] {exc}", flush=True)
            return {}
        latest_frames = getattr(self.camera, "latest_frames", None)
        if callable(latest_frames):
            return latest_frames()
        return {self.camera.config.name: self.camera.latest()}

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


def _pose_error(actual: list[float], target: list[float]) -> tuple[float, float]:
    position_error = max(abs(actual[index] - target[index]) for index in range(3))
    orientation_error = max(
        abs((actual[index] - target[index] + 180.0) % 360.0 - 180.0)
        for index in range(3, 6)
    )
    return position_error, orientation_error


def _pose_reached(
    position_error: float,
    orientation_error: float,
    runtime: TeleopRuntimeConfig,
) -> bool:
    return (
        position_error <= runtime.restore_position_tolerance_mm
        and orientation_error <= runtime.restore_orientation_tolerance_deg
    )
