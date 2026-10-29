from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Data_Collections"))

from vr_teleop.collection.vr_teleop import create_vr_teleop_collection
from vr_teleop.control.safety_supervisor import SafetySupervisor
from vr_teleop.core.app import TeleopRuntimeConfig
from vr_teleop.devices.camera import (
    CameraConfig,
    CameraGroup,
    configure_preview_display,
    create_camera,
)
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter, MockRobotAdapter
from vr_teleop.devices.gripper_controller import GripperConfig, TriggerGripperController
from master_teleop.control.button_controller import MasterButtonController
from master_teleop.collection.launch_nodes import DATASET_TASK_NAME
from master_teleop.control.joint_mapper import RelativeJointMapper
from master_teleop.core.app import MasterHandTeleopApp
from master_teleop.core.config import load_master_project_config
from master_teleop.core.sync_trace import MasterSyncTrace
from master_teleop.devices.master_hand import (
    DynamixelMasterHand,
    MasterHandConfig,
    MockMasterHand,
)
from scripts.recording_status_window import RecordingStatusWindow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Single USB master hand to one Dobot CR5A teleoperation."
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--master-mode", choices=["dynamixel", "mock"], default="dynamixel")
    parser.add_argument("--robot-mode", choices=["dobot", "mock"], default="dobot")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--probe-only", action="store_true", help="Read the USB master only.")
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=0.5,
        help="How long to monitor the master in probe-only mode.",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute", action="store_true", help="Actually send ServoJ commands."
    )
    execution.add_argument(
        "--dry-run", action="store_true", help="Do not send robot or gripper commands."
    )
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--speed-factor", type=int, choices=range(1, 101), default=100)
    parser.add_argument("--record-data", action="store_true")
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "datasets"),
        help="Episode output root (default: <project>/datasets).",
    )
    parser.add_argument("--task", default=DATASET_TASK_NAME)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument(
        "--no-sync-log",
        action="store_true",
        help="Disable the low-rate master/follower alignment diagnostic JSONL.",
    )
    camera_mode = parser.add_mutually_exclusive_group()
    camera_mode.add_argument("--no-camera", action="store_true")
    camera_mode.add_argument(
        "--with-camera", action="store_true", help="Enable configured collection cameras."
    )
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--camera-preview", dest="camera_preview", action="store_true")
    preview.add_argument(
        "--no-camera-preview",
        dest="camera_preview",
        action="store_false",
        help="Capture configured cameras without opening preview windows.",
    )
    parser.set_defaults(camera_preview=False)
    args = parser.parse_args(argv)
    if args.enable_robot and not args.execute:
        parser.error("--enable-robot requires --execute")
    if args.execute and args.robot_mode != "dobot":
        parser.error("--execute requires --robot-mode dobot")
    if args.probe_seconds <= 0.0:
        parser.error("--probe-seconds must be positive")

    config = load_master_project_config(args.config_dir)
    if args.camera_preview and not args.no_camera:
        configure_preview_display(config)
    robot_cfg = config["robot"]
    network = config["robot_network"]
    master_cfg = MasterHandConfig.from_mapping(config.get("master_hand"))
    master = (
        MockMasterHand()
        if args.master_mode == "mock"
        else DynamixelMasterHand(master_cfg)
    )
    if args.probe_only:
        master.connect()
        try:
            deadline = time.monotonic() + max(args.probe_seconds, 3.0)
            sample = None
            while sample is None and time.monotonic() < deadline:
                time.sleep(0.05)
                sample = master.latest()
            if sample is None:
                raise SystemExit("No master-hand sample received within 3 seconds")
            print(f"profile={sample.profile_name}")
            monitor_deadline = time.monotonic() + args.probe_seconds
            last_signature = None
            while time.monotonic() < monitor_deadline:
                sample = master.latest()
                if sample is not None:
                    signature = (
                        tuple(round(value, 3) for value in sample.joint_positions_rad),
                        round(sample.gripper_open, 2),
                        tuple(sample.buttons_raw),
                    )
                    if signature != last_signature:
                        joints = ", ".join(
                            f"{value:+.3f}" for value in sample.joint_positions_rad
                        )
                        print(
                            f"joints_rad=[{joints}] gripper_open={sample.gripper_open:.3f} "
                            f"buttons_raw={sample.buttons_raw} "
                            f"A={sample.button_a} B={sample.button_b}",
                            flush=True,
                        )
                        last_signature = signature
                time.sleep(0.05)
            return
        finally:
            master.close()

    safety = SafetySupervisor(config, require_robot_enabled=args.execute)
    mapper = RelativeJointMapper(config, safety)
    button_values = config.get("master_hand", {}).get("buttons", {})
    buttons = MasterButtonController(
        short_press_max_s=float(button_values.get("short_press_max_s", 0.5)),
        long_press_min_s=float(button_values.get("long_press_min_s", 1.0)),
        initial_locked=bool(button_values.get("initial_locked", True)),
    )
    mapping_values = config.get("master_hand", {}).get("mapping", {})
    runtime = TeleopRuntimeConfig(
        servo_period_s=float(robot_cfg.get("servo_period_s", 0.02)),
        servo_t_s=float(robot_cfg.get("servo_t_s", 0.02)),
        servo_aheadtime=float(
            mapping_values.get("servo_aheadtime", robot_cfg.get("servo_aheadtime", 35))
        ),
        servo_gain=float(
            mapping_values.get("servo_gain", robot_cfg.get("servo_gain", 450))
        ),
        speed_factor_percent=args.speed_factor,
        execute=args.execute,
        enable_robot=args.enable_robot,
    )
    robot = (
        MockRobotAdapter(robot_cfg["default_pose_mm_deg"])
        if args.robot_mode == "mock"
        else DobotTcpAdapter(
            robot_ip=args.robot_ip or network["ip"],
            dashboard_port=int(network["dashboard_port"]),
            move_port=int(network["move_port"]),
            feedback_port=int(network["feedback_port"]),
            user_index=int(robot_cfg.get("user_index", 0)),
            tool_index=int(robot_cfg.get("tool_index", 0)),
        )
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
    if not args.no_camera:
        values = _camera_values(config, args.camera_preview)
        cameras = [
            create_camera(CameraConfig.from_mapping(item))
            for item in values
            if bool(item.get("enabled", False))
        ]
        if cameras:
            camera = CameraGroup(cameras)

    recorder = None
    if args.record_data:
        recorder = create_vr_teleop_collection(
            config=config,
            project_root=ROOT,
            control_period_s=runtime.servo_period_s,
            dataset_dir=args.dataset_dir,
            task=args.task,
        )

    recording_status_window = None
    if args.camera_preview and camera is not None:
        recording_status_window = RecordingStatusWindow(
            width=int(camera.config.width),
            height=int(camera.config.height),
        )

    sync_trace = None
    if not args.no_sync_log:
        sync_trace = MasterSyncTrace(
            ROOT / "outputs" / "master_sync_diagnostics",
            config,
        )
        print(f"master_sync_diagnostic={sync_trace.path}", flush=True)

    MasterHandTeleopApp(
        master=master,
        robot=robot,
        mapper=mapper,
        runtime=runtime,
        buttons=buttons,
        recorder=recorder,
        gripper=gripper,
        camera=camera,
        recording_status_window=recording_status_window,
        sync_trace=sync_trace,
        predict_target_workspace=bool(
            mapping_values.get("predict_target_workspace", False)
        ),
    ).run_forever()


def _camera_values(config: dict, preview: bool) -> list[dict]:
    primary = dict(config.get("camera", {}))
    primary.setdefault("name", "top")
    result = [primary]
    additional = config.get("cameras", {})
    if not isinstance(additional, dict):
        raise ValueError("config/camera.yaml: cameras must be a mapping")
    for name, values in additional.items():
        if not isinstance(values, dict):
            raise ValueError(f"config/camera.yaml: cameras.{name} must be a mapping")
        item = dict(values)
        item.setdefault("name", str(name))
        result.append(item)
    if preview:
        for item in result:
            item["preview"] = True
    return result


if __name__ == "__main__":
    main()
