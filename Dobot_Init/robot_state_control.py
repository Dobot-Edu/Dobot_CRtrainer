from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from vr_teleop.core.config import load_project_config
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter, RobotAdapterError


STATE_CHANGING_ACTIONS = {"enable", "disable", "clear-error", "stop", "pause"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dobot 状态控制台：不启动 VR，只操作控制器状态。"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=["status", "enable", "disable", "clear-error", "stop", "pause"],
        help="状态操作；省略时默认执行 enable。",
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="确认执行有副作用的状态命令。",
    )
    args = parser.parse_args()
    action = args.action or "enable"

    # A bare invocation is the project startup shortcut for enabling the robot.
    # Keep explicit state-changing actions protected by --confirm.
    if action in STATE_CHANGING_ACTIONS and args.action is not None and not args.confirm:
        parser.error(f"{action} 会改变机械臂状态，请追加 --confirm")

    config = load_project_config(args.config_dir)
    network = config["robot_network"]
    robot = DobotTcpAdapter(
        args.robot_ip or network["ip"],
        int(network["dashboard_port"]),
        int(network["move_port"]),
        int(network["feedback_port"]),
    )
    try:
        robot.connect()
        if action == "status":
            time.sleep(0.4)
            state = robot.state()
            print(f"connected={state.connected}")
            print(f"enabled={state.enabled}")
            print(f"robot_mode={state.robot_mode}")
            print(f"pose_mm_deg={state.pose_mm_deg}")
            print(f"q_actual_deg={state.q_actual_deg}")
            print(f"safety_state={state.safety_state}")
            print(f"error_status={state.error_status}")
            print(f"collision_state={state.collision_state}")
            print(f"speed_scaling={state.speed_scaling}")
        elif action == "enable":
            robot.enable_robot()
            print("EnableRobot sent")
        elif action == "disable":
            robot.disable_robot()
            print("DisableRobot sent")
        elif action == "clear-error":
            robot.clear_error()
            print("ClearError sent")
        elif action == "stop":
            robot.stop()
            print("Stop sent")
        elif action == "pause":
            robot.pause()
            print("Pause sent")
        return 0
    except RobotAdapterError as exc:
        print(f"Dobot command failed: {exc}", file=sys.stderr)
        return 2
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
