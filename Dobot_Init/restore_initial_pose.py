from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from vr_teleop.core.config import load_project_config
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter, RobotAdapterError
from vr_teleop.devices.gripper_controller import (
    GripperConfig,
    GripperError,
    TriggerGripperController,
)

ROBOT_MODE_NAMES = {
    1: "initializing",
    2: "brake_open",
    4: "disabled",
    5: "enabled_idle",
    6: "backdrive",
    7: "running",
    8: "drag_recording",
    9: "error",
    10: "paused",
    11: "jogging",
}


def max_pose_error(actual: list[float], target: list[float]) -> tuple[float, float]:
    position_error = max(abs(actual[index] - target[index]) for index in range(3))
    orientation_error = max(
        abs((actual[index] - target[index] + 180.0) % 360.0 - 180.0)
        for index in range(3, 6)
    )
    return position_error, orientation_error


def format_pose(label: str, pose: list[float]) -> str:
    values = ", ".join(f"{value:.3f}" for value in pose)
    return f"{label}=[{values}]"


def max_joint_error(actual: list[float], target: list[float]) -> float:
    if len(actual) < 6 or len(target) != 6:
        raise ValueError("Joint restore requires six actual and target joint values")
    return max(abs(actual[index] - target[index]) for index in range(6))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 Dobot 通过关节模式 MovJ 恢复到 config/robot.yaml 中的绝对初始关节姿态。"
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument(
        "--robot-ip",
        default=None,
        help="Override config/network.yaml robot_network.ip.",
    )
    parser.add_argument("--acceleration", type=int, default=100, choices=range(1, 101))
    parser.add_argument("--velocity", type=int, default=100, choices=range(1, 101))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--joint-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取并显示当前/目标关节和 TCP 位姿，不发送 MovJ。",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="跳过 config/gripper.yaml 中配置的夹爪初始化。",
    )
    # Keep the old flags accepted for compatibility. Restoration is now the
    # default action; use --dry-run when a non-moving preview is required.
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--confirm", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    execute = not args.dry_run

    config = load_project_config(args.config_dir)
    calibration = config["calibration"]
    target_joints = [
        float(value) for value in calibration["robot_initial_joints_deg"]
    ]
    target_pose = [
        float(value) for value in calibration["robot_initial_pose_mm_deg"]
    ]
    if len(target_joints) != 6 or len(target_pose) != 6:
        raise ValueError("Initial joint and TCP targets must each contain six values")
    if args.joint_tolerance <= 0.0:
        parser.error("--joint-tolerance must be positive")
    network = config["robot_network"]
    robot_config = config["robot"]
    robot = DobotTcpAdapter(
        args.robot_ip or network["ip"],
        int(network["dashboard_port"]),
        int(network["move_port"]),
        int(network["feedback_port"]),
        user_index=int(robot_config.get("user_index", 0)),
        tool_index=int(robot_config.get("tool_index", 0)),
    )
    gripper = None
    gripper_values = config.get("gripper", {})
    if not args.no_gripper and bool(gripper_values.get("enabled", False)):
        gripper_config = replace(
            GripperConfig.from_mapping(gripper_values),
            require_initial_press_before_open=False,
        )
        gripper = TriggerGripperController(
            gripper_config,
            robot_ip=args.robot_ip or network["ip"],
            execute=execute,
        )
    moving = False
    try:
        robot.connect()
        time.sleep(0.4)
        state = robot.state()
        if len(state.q_actual_deg) < 6:
            raise RobotAdapterError("Robot feedback does not contain six joint values")
        print(format_pose("current_pose_mm_deg", state.pose_mm_deg))
        print(format_pose("target_pose_mm_deg", target_pose))
        print(format_pose("current_joints_deg", state.q_actual_deg[:6]))
        print(format_pose("target_joints_deg", target_joints))
        position_error, orientation_error = max_pose_error(state.pose_mm_deg, target_pose)
        joint_error = max_joint_error(state.q_actual_deg, target_joints)
        print(
            "current_error: "
            f"joint={joint_error:.3f} deg, "
            f"position={position_error:.3f} mm, "
            f"orientation={orientation_error:.3f} deg"
        )

        if joint_error <= args.joint_tolerance:
            print("Robot is already at the absolute initial joint configuration; skipping MovJ.")
            return 0

        if not execute:
            print("Dry run only. Without --dry-run, the script sends MovJ.")
            return 0

        if state.error_status or state.collision_state:
            raise RobotAdapterError(
                "Robot safety state blocks restoration: "
                f"error_status={state.error_status}, "
                f"collision_state={state.collision_state}. "
                "Remove the physical cause and reset the collision/alarm in "
                "DobotStudio before enabling motion."
            )

        if state.robot_mode not in (5, 7):
            mode_name = ROBOT_MODE_NAMES.get(state.robot_mode, "unknown")
            raise RobotAdapterError(
                "Robot is not ready for MovJ: "
                f"robot_mode={state.robot_mode} ({mode_name}). "
                "Exit Jog/drag/pause, release any physical emergency stop, "
                "and enable the robot until it reaches mode 5 (enabled idle)."
            )

        print(
            f"Sending MovJ(joint) at acceleration={args.acceleration}% "
            f"velocity={args.velocity}%. Keep the emergency stop reachable."
        )
        if gripper is not None:
            print("Initializing configured gripper before restoring initial pose.")
            gripper.connect(robot.dashboard)
        robot.move_j_joint(target_joints, args.acceleration, args.velocity)
        moving = True

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            time.sleep(0.2)
            state = robot.state()
            if len(state.q_actual_deg) < 6:
                raise RobotAdapterError("Robot feedback lost six-joint state during restore")
            joint_error = max_joint_error(state.q_actual_deg, target_joints)
            position_error, orientation_error = max_pose_error(
                state.pose_mm_deg, target_pose
            )
            print(
                f"moving mode={state.robot_mode} "
                f"joint_error={joint_error:.2f}deg "
                f"position_error={position_error:.2f}mm "
                f"orientation_error={orientation_error:.2f}deg"
            )
            if joint_error <= args.joint_tolerance:
                moving = False
                print(format_pose("restored_joints_deg", state.q_actual_deg[:6]))
                print(format_pose("restored_pose_mm_deg", state.pose_mm_deg))
                print("Absolute initial joint configuration restored.")
                return 0
            if state.error_status or state.collision_state:
                raise RobotAdapterError(
                    f"Motion stopped by robot state: error={state.error_status}, "
                    f"collision={state.collision_state}"
                )

        raise RobotAdapterError("Timed out while waiting for the initial pose")
    except KeyboardInterrupt:
        if moving:
            print("Stopping the queued motion...", file=sys.stderr)
            robot.stop()
        return 130
    except RobotAdapterError as exc:
        print(f"Restore failed: {exc}", file=sys.stderr)
        return 2
    except GripperError as exc:
        print(f"Gripper initialization failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if gripper is not None:
            gripper.close()
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
