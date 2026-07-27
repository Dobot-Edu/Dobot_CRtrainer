from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from vr_teleop.core.app import TeleopApp, TeleopRuntimeConfig
from vr_teleop.devices.camera import CameraConfig, CameraGroup, create_camera
from vr_teleop.core.config import load_project_config
from vr_teleop.collection.vr_teleop import create_vr_teleop_collection
from vr_teleop.control.mapper import RelativeVRMapper
from vr_teleop.control.safety_supervisor import SafetySupervisor
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter, MockRobotAdapter
from vr_teleop.devices.gripper_controller import GripperConfig, TriggerGripperController
from vr_teleop.network.host_discovery import VRHostDiscoveryResponder
from vr_teleop.network.vr_udp_receiver import VRUDPReceiver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument(
        "--robot-mode",
        choices=["mock", "dobot"],
        default="dobot",
        help="Robot backend. Defaults to the real Dobot robot.",
    )
    parser.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        default=True,
        help="Actually send ServoP commands to the robot. Enabled by default.",
    )
    parser.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Do not send motion commands; useful for mock or link testing.",
    )
    parser.add_argument(
        "--enable-robot",
        dest="enable_robot",
        action="store_true",
        default=True,
        help="Send EnableRobot before teleoperation. Requires --execute and --robot-mode dobot.",
    )
    parser.add_argument(
        "--no-enable-robot",
        dest="enable_robot",
        action="store_false",
        help="Do not send EnableRobot automatically at startup.",
    )
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument(
        "--no-j6-lock",
        action="store_true",
        help="Baseline test only: use direct ServoP instead of J6-lock IK + ServoJ.",
    )
    parser.add_argument(
        "--speed-factor",
        type=int,
        choices=range(1, 101),
        default=None,
        help="Override the Dobot global SpeedFactor percentage in execute mode.",
    )
    parser.add_argument(
        "--record-data",
        action="store_true",
        help="Enable synchronized VR and robot episode recording; press Quest B to start/stop.",
    )
    parser.add_argument(
        "--no-auto-restore",
        action="store_true",
        help="Do not return to the configured initial pose after stopping an episode.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Override the dataset output directory.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task name stored in episode metadata and the dataset directory.",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Disable the configured PGE gripper for this run.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable the configured camera for this run.",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Show the continuously captured camera RGB stream.",
    )
    args = parser.parse_args()
    if args.enable_robot and args.robot_mode == "dobot" and not args.execute:
        parser.error("--enable-robot requires --execute when using --robot-mode dobot")

    config = load_project_config(args.config_dir)
    network = config["robot_network"]
    robot_cfg = config["robot"]
    calibration_cfg = config.get("calibration", {})
    collection_cfg = config.get("data_collection", {})
    vr_cfg = config["vr"]
    host_network = config.get("host_network", {})
    listen_ip = str(host_network.get("listen_ip", vr_cfg["listen_ip"]))
    listen_port = int(host_network.get("listen_port", vr_cfg["listen_port"]))
    local_ip = str(host_network.get("local_ip", "")).strip()
    discovery_enabled = bool(host_network.get("discovery_enabled", True))
    discovery_port = int(host_network.get("discovery_port", 5006))

    receiver = VRUDPReceiver(listen_ip, listen_port)
    discovery = None
    if discovery_enabled:
        discovery = VRHostDiscoveryResponder(listen_ip, discovery_port, listen_port)
        discovery.start()
        print(f"VR host discovery listening on UDP {listen_ip}:{discovery_port}")
    if local_ip:
        print(
            f"Quest 3 automatic discovery enabled; manual fallback target is "
            f"{local_ip}:{listen_port}"
        )
    safety = SafetySupervisor(config, require_robot_enabled=args.execute)
    mapper = RelativeVRMapper(config, safety)
    runtime = TeleopRuntimeConfig(
        servo_period_s=float(robot_cfg.get("servo_period_s", 0.05)),
        servo_t_s=float(robot_cfg.get("servo_t_s", 0.1)),
        servo_aheadtime=float(robot_cfg.get("servo_aheadtime", 50)),
        servo_gain=float(robot_cfg.get("servo_gain", 500)),
        speed_factor_percent=int(
            args.speed_factor
            if args.speed_factor is not None
            else robot_cfg.get("speed_factor_percent", 100)
        ),
        execute=args.execute,
        enable_robot=args.enable_robot,
        auto_restore_after_episode=bool(
            args.record_data
            and collection_cfg.get("auto_restore_after_episode", True)
            and not args.no_auto_restore
        ),
        initial_pose_mm_deg=[
            float(value)
            for value in calibration_cfg.get(
                "robot_initial_pose_mm_deg",
                robot_cfg["default_pose_mm_deg"],
            )
        ],
        restore_acceleration_percent=int(
            collection_cfg.get("restore_acceleration_percent", 10)
        ),
        restore_velocity_percent=int(
            collection_cfg.get("restore_velocity_percent", 10)
        ),
        restore_timeout_s=float(collection_cfg.get("restore_timeout_s", 60.0)),
        restore_position_tolerance_mm=float(
            collection_cfg.get("restore_position_tolerance_mm", 1.5)
        ),
        restore_orientation_tolerance_deg=float(
            collection_cfg.get("restore_orientation_tolerance_deg", 2.0)
        ),
    )

    if args.robot_mode == "mock":
        robot = MockRobotAdapter(robot_cfg["default_pose_mm_deg"])
    else:
        robot = DobotTcpAdapter(
            robot_ip=args.robot_ip or network["ip"],
            dashboard_port=int(network["dashboard_port"]),
            move_port=int(network["move_port"]),
            feedback_port=int(network["feedback_port"]),
            lock_j6=bool(robot_cfg.get("lock_j6", False) and not args.no_j6_lock),
        )

    gripper = None
    gripper_values = config.get("gripper", {})
    if not args.no_gripper and bool(gripper_values.get("enabled", False)):
        gripper = TriggerGripperController(
            GripperConfig.from_mapping(gripper_values),
            robot_ip=args.robot_ip or network["ip"],
            execute=bool(args.execute and args.robot_mode == "dobot"),
        )

    camera = None
    camera_values = dict(config.get("camera", {}))
    camera_values.setdefault("name", "top")
    additional_camera_values = config.get("cameras", {})
    if not isinstance(additional_camera_values, dict):
        parser.error("config/camera.yaml: cameras must be a mapping of camera names")
    all_camera_values = [camera_values]
    for name, values in additional_camera_values.items():
        if not isinstance(values, dict):
            parser.error(f"config/camera.yaml: cameras.{name} must be a mapping")
        camera_value = dict(values)
        camera_value.setdefault("name", str(name))
        all_camera_values.append(camera_value)
    if args.camera_preview:
        for values in all_camera_values:
            values["preview"] = True
    if not args.no_camera:
        cameras = [
            create_camera(CameraConfig.from_mapping(values))
            for values in all_camera_values
            if bool(values.get("enabled", False))
        ]
        if cameras:
            camera = CameraGroup(cameras)

    if args.robot_mode == "dobot" and not args.execute:
        print("Dobot mode without --execute will connect and compute targets, but it will not send ServoP.")

    recorder = None
    if args.record_data:
        recorder = create_vr_teleop_collection(
            config=config,
            project_root=ROOT,
            control_period_s=runtime.servo_period_s,
            dataset_dir=args.dataset_dir,
            task=args.task,
        )

    app = TeleopApp(
        receiver,
        robot,
        mapper,
        runtime,
        recorder=recorder,
        gripper=gripper,
        camera=camera,
    )
    try:
        app.run_forever()
    finally:
        if discovery is not None:
            discovery.stop()


if __name__ == "__main__":
    main()
