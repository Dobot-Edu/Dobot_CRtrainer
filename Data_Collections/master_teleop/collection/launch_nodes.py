from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Data_Collections"))

# 数据集任务名：日常采集只需修改这一行（建议使用英文、数字或下划线）。
DATASET_TASK_NAME = "cosmetics_quality"

from master_teleop.collection.workflow import (
    create_master,
    create_robot,
    format_six,
    wait_master_sample,
    wait_robot_state,
)
from master_teleop.core.config import load_master_project_config
from vr_teleop.devices.camera import (
    CameraConfig,
    CameraGroup,
    configure_preview_display,
    create_camera,
)
from vr_teleop.devices.gripper_controller import GripperConfig, TriggerGripperController


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: connect the single CR5A, raise it through the Xtrainer "
            "waypoint, and return it to the initial pose."
        )
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--no-master", action="store_true")
    parser.add_argument("--check-camera", action="store_true")
    parser.add_argument("--camera-preview", action="store_true")
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Skip the automatic startup gripper OPEN -> CLOSE test.",
    )
    parser.add_argument("--monitor-seconds", type=float, default=0.0)
    parser.add_argument("--speed-factor", type=int, choices=range(1, 101), default=20)
    parser.add_argument(
        "--skip-pose-init",
        action="store_true",
        help="Read-only: do not raise/restore CR5A or run the gripper test.",
    )
    args = parser.parse_args()
    if args.monitor_seconds < 0.0:
        parser.error("--monitor-seconds must be non-negative")
    config = load_master_project_config(args.config_dir)
    if args.check_camera and args.camera_preview:
        configure_preview_display(config)
    dataset_output_dir = ROOT / "datasets" / DATASET_TASK_NAME / "episodes"
    print(f"[dataset] B-button episodes will be saved under: {dataset_output_dir}")
    master = None if args.no_master else create_master(config)
    robot = create_robot(config, args.robot_ip)
    gripper = None
    cameras = _create_cameras(config, args.camera_preview) if args.check_camera else None
    try:
        if master is not None:
            master.connect()
            sample = wait_master_sample(master)
            print(
                f"[OK] master profile={sample.profile_name} "
                f"joints_rad={format_six(sample.joint_positions_rad)} "
                f"buttons={sample.buttons_raw}"
            )
        robot.connect()
        state = wait_robot_state(robot)
        print(
            f"[OK] CR5A connected={state.connected} enabled={state.enabled} "
            f"mode={state.robot_mode} errors={state.error_status} collision={state.collision_state}"
        )
        print(f"     joints_deg={format_six(state.q_actual_deg, 2)}")
        print(f"     pose_mm_deg={format_six(state.pose_mm_deg, 2)}")
        if not args.skip_pose_init:
            _ensure_robot_ready(robot, state)
            robot.set_speed_factor(args.speed_factor)
            print(f"CR5A SpeedFactor set to {args.speed_factor}%")
            pose_init = config["master_hand"].get("robot_pose_init", {})
            raised = _six_joint_values(
                pose_init.get("raised_joint_deg", [90, -30, 110, -20, -90, -90]),
                "raised_joint_deg",
            )
            home = _six_joint_values(
                pose_init.get("home_joint_deg", [90, 0, 90, 0, -90, -90]),
                "home_joint_deg",
            )
            print(f"[pose init 1/2] raising CR5A to {format_six(raised, 1)} deg")
            _stream_joint_waypoint(robot, raised, config, pose_init)
            print(f"[pose init 2/2] restoring CR5A to {format_six(home, 1)} deg")
            _stream_joint_waypoint(robot, home, config, pose_init)
            final_state = wait_robot_state(robot)
            print(f"[OK] CR5A initial pose restored: {format_six(final_state.q_actual_deg, 2)}")
            gripper_values = config.get("gripper", {})
            if not args.no_gripper and bool(gripper_values.get("enabled", False)):
                gripper = TriggerGripperController(
                    GripperConfig.from_mapping(gripper_values),
                    robot_ip=args.robot_ip or config["robot_network"]["ip"],
                    execute=True,
                )
                gripper.connect(getattr(robot, "dashboard", None))
                _run_gripper_open_close_cycle(gripper)
        if cameras is not None:
            cameras.start()
            deadline = time.monotonic() + 12.0
            frames = {}
            while time.monotonic() < deadline:
                cameras.raise_if_failed()
                frames = cameras.latest_frames()
                if frames and all(frame is not None for frame in frames.values()):
                    break
                time.sleep(0.05)
            missing = [name for name, frame in frames.items() if frame is None]
            if not frames or missing:
                raise RuntimeError(f"camera preflight timed out; missing={missing or 'all'}")
            print(f"[OK] cameras={list(frames)}")
            if args.camera_preview:
                cameras.show_previews()

        if args.monitor_seconds:
            print(f"Monitoring read-only state for {args.monitor_seconds:.1f}s...")
            deadline = time.monotonic() + args.monitor_seconds
            last_log = 0.0
            while time.monotonic() < deadline:
                now = time.monotonic()
                state = robot.state()
                if now - last_log >= 1.0:
                    print(
                        f"mode={state.robot_mode} q={format_six(state.q_actual_deg, 2)} "
                        f"feedback_age={state.feedback_age_s:.3f}s"
                    )
                    last_log = now
                if cameras is not None:
                    cameras.raise_if_failed()
                    if args.camera_preview:
                        cameras.show_previews()
                time.sleep(0.02)
        if args.skip_pose_init:
            print("Read-only preflight passed; no robot motion command was sent.")
        else:
            print(
                "Xtrainer-style pose initialization and automatic gripper test "
                "complete. Run run_control.py next."
            )
    except KeyboardInterrupt:
        print("Stopping CR5A pose initialization...")
        robot.stop()
    except Exception:
        if not args.skip_pose_init:
            try:
                robot.stop()
            except Exception:
                pass
        raise
    finally:
        if gripper is not None:
            gripper.close()
        if cameras is not None:
            cameras.stop()
        if master is not None:
            master.close()
        robot.close()


def _run_gripper_open_close_cycle(gripper) -> None:
    open_trigger = 0.0 if gripper.config.pressed_closes else 1.0
    close_trigger = 1.0 if gripper.config.pressed_closes else 0.0
    print("[gripper test] OPEN -> CLOSE (final state: CLOSED)", flush=True)
    gripper.update(open_trigger, input_valid=True)
    _wait_gripper_command(gripper, False)
    open_actual = _wait_gripper_endpoint(gripper, expected_closed=False)
    time.sleep(gripper.config.test_cycle_hold_s)
    gripper.update(close_trigger, input_valid=True)
    _wait_gripper_command(gripper, True)
    close_actual = _wait_gripper_endpoint(gripper, expected_closed=True)
    print(
        "[OK] gripper startup cycle complete; follower gripper remains CLOSED "
        f"(open_actual={open_actual}, close_actual={close_actual})",
        flush=True,
    )


def _wait_gripper_command(gripper, expected_closed: bool) -> None:
    deadline = time.monotonic() + gripper.config.test_cycle_command_timeout_s
    while time.monotonic() < deadline:
        snapshot = gripper.snapshot()
        if snapshot.get("error"):
            raise RuntimeError(f"gripper test failed: {snapshot['error']}")
        if snapshot.get("applied_closed") is expected_closed:
            return
        time.sleep(0.01)
    raise RuntimeError(
        f"gripper test command timed out: expected_closed={expected_closed}"
    )


def _wait_gripper_endpoint(gripper, expected_closed: bool) -> int:
    target = (
        int(gripper.config.close_position)
        if expected_closed
        else int(gripper.config.open_position)
    )
    tolerance = int(gripper.config.test_cycle_position_tolerance)
    started = time.monotonic()
    deadline = started + float(gripper.config.test_cycle_command_timeout_s)
    minimum_travel_s = float(gripper.config.test_cycle_travel_s)
    last_position: int | None = None
    stable_samples = 0
    while time.monotonic() < deadline:
        snapshot = gripper.snapshot()
        if snapshot.get("error"):
            raise RuntimeError(f"gripper endpoint check failed: {snapshot['error']}")
        actual_value = snapshot.get("actual_position")
        if actual_value is not None:
            actual = int(actual_value)
            if abs(actual - target) <= tolerance:
                return actual
            elapsed = time.monotonic() - started
            if elapsed >= minimum_travel_s and last_position is not None:
                stable_samples = stable_samples + 1 if abs(actual - last_position) <= 2 else 0
                if stable_samples >= 3:
                    state = "CLOSED" if expected_closed else "OPEN"
                    print(
                        f"[gripper test] {state} mechanical endpoint settled: "
                        f"target={target} actual={actual}",
                        flush=True,
                    )
                    return actual
            last_position = actual
        time.sleep(min(float(gripper.config.feedback_poll_interval_s), 0.10))
    state = "closed" if expected_closed else "open"
    raise RuntimeError(
        f"gripper did not reach the {state} endpoint: "
        f"target={target} actual={last_position} tolerance={tolerance}"
    )


def _create_cameras(config: dict, preview: bool) -> CameraGroup | None:
    values: list[dict] = []
    primary = dict(config.get("camera", {}))
    primary.setdefault("name", "top")
    values.append(primary)
    for name, camera_values in config.get("cameras", {}).items():
        item = dict(camera_values)
        item.setdefault("name", str(name))
        values.append(item)
    cameras = []
    for item in values:
        if not bool(item.get("enabled", False)):
            continue
        if preview:
            item["preview"] = True
        cameras.append(create_camera(CameraConfig.from_mapping(item)))
    return CameraGroup(cameras) if cameras else None


def _ensure_robot_ready(robot, state, timeout_s: float = 5.0) -> None:
    if state.collision_state:
        raise RuntimeError(f"CR5A has collision_state={state.collision_state}")

    if state.error_status:
        error_ids = _read_robot_error_ids(robot)
        suffix = f"; GetErrorID={error_ids}" if error_ids is not None else ""
        print(
            f"[robot recovery] CR5A alarm detected: "
            f"error_status={state.error_status}, mode={state.robot_mode}{suffix}",
            flush=True,
        )
        print("[robot recovery] sending ClearError()...", flush=True)
        robot.clear_error()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = robot.state()
            if state.collision_state:
                raise RuntimeError(
                    f"CR5A entered collision protection while clearing alarm: "
                    f"collision_state={state.collision_state}"
                )
            if not state.error_status:
                print("[OK] CR5A alarm cleared", flush=True)
                break
            time.sleep(0.05)
        else:
            error_ids = _read_robot_error_ids(robot)
            suffix = f"; GetErrorID={error_ids}" if error_ids is not None else ""
            raise RuntimeError(
                f"CR5A alarm did not clear within {timeout_s:.1f}s: "
                f"error_status={state.error_status}, mode={state.robot_mode}{suffix}. "
                "Release the emergency stop and resolve the alarm in DobotStudio, "
                "then run launch_nodes.py again."
            )

    if state.robot_mode not in (5, 7):
        print(
            f"[robot recovery] enabling CR5A from mode={state.robot_mode}...",
            flush=True,
        )
        robot.enable_robot()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = robot.state()
            if state.error_status or state.collision_state:
                raise RuntimeError(
                    "CR5A protection triggered while enabling: "
                    f"error={state.error_status} collision={state.collision_state} "
                    f"mode={state.robot_mode}"
                )
            if state.robot_mode in (5, 7):
                print(f"[OK] CR5A enabled in mode={state.robot_mode}", flush=True)
                return
            time.sleep(0.05)
        raise RuntimeError(f"CR5A did not enter ready mode; mode={state.robot_mode}")


def _read_robot_error_ids(robot):
    dashboard = getattr(robot, "dashboard", None)
    get_error_id = getattr(dashboard, "GetErrorID", None)
    if not callable(get_error_id):
        return None
    try:
        return get_error_id()
    except Exception as exc:
        return f"unavailable ({exc})"


def _stream_joint_waypoint(robot, target_deg: list[float], config: dict, pose_init: dict) -> None:
    robot_values = config["robot"]
    period_s = float(robot_values.get("servo_period_s", 0.02))
    paths = _joint_path(
        robot.state().q_actual_deg,
        target_deg,
        interpolation_step_rad=float(pose_init.get("interpolation_step_rad", 0.01)),
        max_steps=int(pose_init.get("max_interpolation_steps", 100)),
    )
    for waypoint in paths:
        state = robot.state()
        if state.error_status or state.collision_state:
            robot.stop()
            raise RuntimeError(
                f"CR5A protection during pose initialization: "
                f"error={state.error_status} collision={state.collision_state}"
            )
        robot.servo_j(
            waypoint,
            t=float(robot_values.get("servo_t_s", period_s)),
            aheadtime=float(robot_values.get("servo_aheadtime", 35)),
            gain=float(robot_values.get("servo_gain", 450)),
        )
        time.sleep(period_s)

    tolerance = float(pose_init.get("final_tolerance_deg", 1.5))
    deadline = time.monotonic() + float(pose_init.get("settle_timeout_s", 3.0))
    while time.monotonic() < deadline:
        state = robot.state()
        error = max(abs(target - actual) for target, actual in zip(target_deg, state.q_actual_deg))
        if error <= tolerance:
            return
        if state.error_status or state.collision_state:
            robot.stop()
            raise RuntimeError(
                f"CR5A protection while settling: error={state.error_status} "
                f"collision={state.collision_state}"
            )
        robot.servo_j(
            target_deg,
            t=float(robot_values.get("servo_t_s", period_s)),
            aheadtime=float(robot_values.get("servo_aheadtime", 35)),
            gain=float(robot_values.get("servo_gain", 450)),
        )
        time.sleep(period_s)
    raise RuntimeError(
        f"CR5A did not reach waypoint within tolerance: target={target_deg} "
        f"actual={robot.state().q_actual_deg}"
    )


def _joint_path(
    current_deg: list[float],
    target_deg: list[float],
    interpolation_step_rad: float = 0.01,
    max_steps: int = 100,
) -> list[list[float]]:
    current = _six_joint_values(current_deg, "current joints")
    target = _six_joint_values(target_deg, "target joints")
    if interpolation_step_rad <= 0.0 or max_steps < 2:
        raise ValueError("pose initialization interpolation settings are invalid")
    maximum_rad = max(math.radians(abs(end - start)) for start, end in zip(current, target))
    steps = max(2, min(int(maximum_rad / interpolation_step_rad), max_steps))
    return [
        [start + (end - start) * index / (steps - 1) for start, end in zip(current, target)]
        for index in range(1, steps)
    ]


def _six_joint_values(values, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 6 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain six finite joint values")
    return result


if __name__ == "__main__":
    main()
