from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from master_teleop.core.config import MASTER_RUNTIME_CONFIG, _deep_merge
from master_teleop.devices.master_hand import (
    DynamixelMasterHand,
    MasterHandConfig,
)
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def serial_candidates(explicit: Iterable[str] = ()) -> list[str]:
    candidates = [str(Path(value)) for value in explicit if value]
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ):
        candidates.extend(str(path) for path in sorted(Path("/").glob(pattern.lstrip("/"))))
    result: list[str] = []
    resolved: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            identity = str(path.resolve())
        except OSError:
            identity = str(path)
        if identity not in resolved:
            resolved.add(identity)
            result.append(str(path))
    return result


def probe_master(
    master_values: dict[str, Any],
    device: str,
    baud_rate: int,
    profile_name: str,
    wait_s: float = 1.0,
) -> tuple[DynamixelMasterHand, Any]:
    values = copy.deepcopy(master_values)
    values["device"] = device
    values["baud_rate"] = int(baud_rate)
    values["baud_rate_candidates"] = [int(baud_rate)]
    values["side"] = profile_name
    master = DynamixelMasterHand(MasterHandConfig.from_mapping(values))
    try:
        master.connect()
        deadline = time.monotonic() + wait_s
        sample = None
        while sample is None and time.monotonic() < deadline:
            time.sleep(0.02)
            sample = master.latest()
        if sample is None:
            raise RuntimeError("master connected but no joint sample was received")
        return master, sample
    except Exception:
        master.close()
        raise


def create_master(config: dict[str, Any]) -> DynamixelMasterHand:
    return DynamixelMasterHand(MasterHandConfig.from_mapping(config.get("master_hand")))


def create_robot(config: dict[str, Any], robot_ip: str | None = None) -> DobotTcpAdapter:
    network = config["robot_network"]
    robot = config["robot"]
    return DobotTcpAdapter(
        robot_ip=robot_ip or network["ip"],
        dashboard_port=int(network["dashboard_port"]),
        move_port=int(network["move_port"]),
        feedback_port=int(network["feedback_port"]),
        user_index=int(robot.get("user_index", 0)),
        tool_index=int(robot.get("tool_index", 0)),
    )


def wait_master_sample(master: DynamixelMasterHand, timeout_s: float = 3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = master.latest()
        if sample is not None:
            return sample
        time.sleep(0.02)
    raise RuntimeError(f"no master-hand sample received within {timeout_s:.1f}s")


def wait_robot_state(robot: DobotTcpAdapter, timeout_s: float = 3.0):
    deadline = time.monotonic() + timeout_s
    state = robot.state()
    while time.monotonic() < deadline:
        state = robot.state()
        if state.connected and len(state.q_actual_deg) == 6:
            return state
        time.sleep(0.05)
    raise RuntimeError(f"no CR5A feedback received within {timeout_s:.1f}s")


def calibration_from_raw(
    raw_positions: list[float],
    target_joint_deg: list[float],
    joint_signs: Iterable[int],
    position_units_per_pi: float,
) -> dict[str, list[float] | float]:
    if len(raw_positions) < 7 or len(target_joint_deg) != 6:
        raise ValueError("calibration needs six robot joints and seven master positions")
    signs = [int(value) for value in joint_signs]
    if len(signs) != 6 or any(value not in (-1, 1) for value in signs):
        raise ValueError("calibration joint signs must contain six +/-1 values")
    raw_rad = [value / position_units_per_pi * math.pi for value in raw_positions]
    target_rad = [math.radians(value) for value in target_joint_deg]
    offsets = [
        raw - target * sign
        for raw, target, sign in zip(raw_rad[:6], target_rad, signs)
    ]
    gripper_deg = raw_positions[6] / position_units_per_pi * 180.0
    return {
        "joint_offsets_rad": [round(value, 6) for value in offsets],
        "start_joints_rad": [round(value, 6) for value in target_rad],
        "gripper_closed_deg": round(gripper_deg + 30.0, 3),
        "gripper_open_deg": round(gripper_deg - 0.2, 3),
    }


def update_runtime_master(updates: dict[str, Any]) -> Path:
    runtime: dict[str, Any] = {}
    if MASTER_RUNTIME_CONFIG.exists():
        with MASTER_RUNTIME_CONFIG.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"runtime config is not a mapping: {MASTER_RUNTIME_CONFIG}")
        runtime = loaded
    runtime = _deep_merge(runtime, {"master_hand": updates})
    MASTER_RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporary = MASTER_RUNTIME_CONFIG.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by master_teleop collection setup scripts.\n")
        yaml.safe_dump(runtime, handle, allow_unicode=True, sort_keys=False)
    temporary.replace(MASTER_RUNTIME_CONFIG)
    return MASTER_RUNTIME_CONFIG


def format_six(values: Iterable[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):+.{precision}f}" for value in values) + "]"
